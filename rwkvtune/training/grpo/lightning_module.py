"""
GRPO Lightning Module - Core Training Logic
"""
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import Dict, Any, Optional, List
import os
import json
import time
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import numpy as np

# DeepSpeed is an optional dependency
try:
  from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
  DEEPSPEED_AVAILABLE = True
except ImportError:
  DEEPSPEED_AVAILABLE = False
  DeepSpeedCPUAdam = None
  FusedAdam = None

from rwkvtune.models.rwkv7 import RWKV7Model
from rwkvtune.training.grpo.loss import get_loss_function
from rwkvtune.training.grpo.utils import selective_log_softmax


# ========== Memory and Time Monitoring Tools ==========

def get_gpu_memory_info(device=None):
  """
  Get GPU memory information.
  
  Returns:
    dict: Contains allocated, reserved, max_allocated, total (in GB)
  """
  if not torch.cuda.is_available():
    return None
  
  if device is None:
    device = torch.cuda.current_device()
  
  torch.cuda.synchronize(device)
  
  allocated = torch.cuda.memory_allocated(device) / (1024 ** 3) # GB
  reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
  max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3) # GB
  
  props = torch.cuda.get_device_properties(device)
  total = props.total_memory / (1024 ** 3) # GB
  
  return {
    'allocated': allocated,
    'reserved': reserved,
    'max_allocated': max_allocated,
    'total': total,
    'free': total - reserved,
    'usage_percent': (reserved / total) * 100 if total > 0 else 0
  }


@contextmanager
def monitor_memory_and_time(rank, stage_name, device=None):
  """
  Context manager for monitoring memory usage and execution time.
  
  Usage:
    with monitor_memory_and_time(rank, "generation_phase"):
      # execute code
      pass
  """
  if device is None and torch.cuda.is_available():
    device = torch.cuda.current_device()
  
  # Record starting state
  start_time = time.time()
  start_mem = get_gpu_memory_info(device) if torch.cuda.is_available() else None
  
  # Reset peak memory stats (if possible)
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
  
  try:
    yield
  finally:
    # Record ending state
    end_time = time.time()
    end_mem = get_gpu_memory_info(device) if torch.cuda.is_available() else None
    
    elapsed_time = end_time - start_time
    
    # Print monitoring info
    if start_mem and end_mem:
      mem_delta = end_mem['allocated'] - start_mem['allocated']
      peak_mem = end_mem['max_allocated']
      
      print(f"[Rank {rank}] [STATS] {stage_name}:")
      print(f" [TIME] Elapsed: {elapsed_time:.2f}s")
      print(f" [MEM] Memory change: {mem_delta:+.2f} GB (start: {start_mem['allocated']:.2f} GB -> end: {end_mem['allocated']:.2f} GB)")
      print(f" [PEAK] Peak memory: {peak_mem:.2f} GB")
      print(f" [USAGE] Memory usage: {end_mem['usage_percent']:.1f}% ({end_mem['reserved']:.2f} GB / {end_mem['total']:.2f} GB)")
    else:
      print(f"[Rank {rank}] [STATS] {stage_name}: [TIME] Elapsed: {elapsed_time:.2f}s")


class GRPOLightningModule(pl.LightningModule):
  """
  GRPO Lightning Module
  
  Core responsibilities:
  1. Define GRPO training steps
  2. Compute advantage functions
  3. Compute GRPO loss
  4. Configure optimizer and learning rate schedule
  """
  
  def __init__(self, config, use_peft: bool = False, loss_mask_func=None):
    super().__init__()
    self.config = config
    self.save_hyperparameters()
    
    # PEFT flag
    self._use_peft = use_peft
    
    # Loss Mask func (optional): generates custom mask to control which tokens participate in loss calculation
    self.loss_mask_func = loss_mask_func
    
    # Create RWKV7 policy model - need to construct correct RWKV7Config
    from rwkvtune.models.rwkv7 import RWKV7Config
    
    # [FIX] Calculate actual ctx_len based on GRPO config
    # ctx_len needs to accommodate prompt + completion total length
    required_ctx_len = config.max_prompt_length + config.max_completion_length
    
    # Round up to multiple of 16 (RWKV7 training operator requirement)
    CHUNK_LEN = 16
    actual_ctx_len = ((required_ctx_len + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    
    print(f"[INFO] Setting model ctx_len:")
    print(f"  max_prompt_length: {config.max_prompt_length}")
    print(f"  max_completion_length: {config.max_completion_length}")
    print(f"  required_ctx_len: {required_ctx_len}")
    print(f"  actual_ctx_len (aligned to 16): {actual_ctx_len}")
    
    model_config = RWKV7Config(
      n_layer=config.n_layer,
      n_embd=config.n_embd,
      vocab_size=config.vocab_size,
      ctx_len=actual_ctx_len,
      head_size_a=config.head_size_a,
      dim_att_lora=config.dim_att_lora,
      dim_gate_lora=config.dim_gate_lora,
      dim_mv_lora=config.dim_mv_lora,
    )
    self.model = RWKV7Model(model_config)
    
    # Create reference model (if KL penalty is needed)
    # Note: If using PEFT, no separate reference model needed - use disable_adapter instead
    self.ref_model = None
    if config.beta > 0.0 and not use_peft:
      # Use same model_config (with correct ctx_len)
      self.ref_model = RWKV7Model(model_config)
      # Reference model does not participate in gradient updates
      for param in self.ref_model.parameters():
        param.requires_grad = False
      self.ref_model.eval()
    
    if use_peft and config.beta > 0.0:
      print("[OK] Using LoRA: reference model via disable_adapter, saving memory")
    
    # Get loss function
    self.loss_fn = get_loss_function(config.loss_type, config)
    
    # Detect DeepSpeed offload
    self.deepspeed_offload = (
      hasattr(config, 'strategy') and 
      'deepspeed' in str(config.strategy).lower() and
      'offload' in str(config.strategy).lower()
    )
    
    # Training step counter
    self.training_step_count = 0
    
    # Checkpoint resume state
    self.skip_until_step = 0
    self.resume_checkpoint_path = None
    
    # If LoRA is enabled, disable strict loading
    if use_peft:
      self.strict_loading = False

    # Track last rollout save step (for“next time after crossing threshold rollout save”）
    # Note：rollout data only generated in on_train_batch_start “generation phase”generated，thereforesavecan only happen at rollout time。
    # to make save_rollout_steps=100 semantics more intuitive（instead of requiring exact division），using“distance from lasttimessaveexceeds threshold thensave”strategy。
    self._last_saved_rollout_training_step_count: Optional[int] = None

    # Cache for GRPO data (generation, rewards, advantages)
    self.current_grpo_data = None
    self._buffered_grpo_data = []
    self._buffer_step = 0
    self._generation_step = 0
    self._active_prompt_batch_signature: Optional[tuple] = None
    self._active_rollout_generation_step: Optional[int] = None
    self.generator = None
    self.reward_functions = None
    self.reward_weights = None
    self.reward_func_names = None
    self.completion_postprocess_fn = None  # optional hook: called after rollout, before reward
    self._metrics = defaultdict(list)
    self._rollout_buffer = []

  def set_skip_steps(self, steps: int):
    """Set steps to skip (for checkpoint resume)"""
    self.skip_until_step = steps
    
  def set_resume_checkpoint_path(self, path: str):
    """Set checkpoint path for resume (for delayed optimizer loading)"""
    self.resume_checkpoint_path = path

  def on_save_checkpoint(self, checkpoint):
    """Save additional state to checkpoint"""
    # Save total samples processed for elastic resume
    # [NOTE] GRPO special logic:
    # GRPO uses RepeatBatchSampler, same Prompt Batch is repeated generate_every times.
    # generate_every = steps_per_generation * num_iterations
    # 
    # Lightning's global_step corresponds to accumulate_grad_batches training_steps.
    # Usually accumulate_grad_batches == steps_per_generation.
    # 
    # Therefore, each global_step only consumes micro_bsz * devices / num_iterations Prompts.
    # (if num_iterations=1, consumes micro_bsz * devices Prompts)
    
    if hasattr(self.config, 'micro_bsz') and hasattr(self.config, 'devices'):
      # Get generate_every related parameters
      steps_per_generation = int(getattr(self.config, "steps_per_generation", self.config.accumulate_grad_batches))
      num_iterations = int(getattr(self.config, "num_iterations", 1))
      generate_every = steps_per_generation * num_iterations
      
      # Calculate actual Prompt consumption
      # real_training_steps = global_step * accumulate_grad_batches
      # batches_seen = real_training_steps / generate_every
      # samples_seen = batches_seen * micro_bsz * devices
      
      # Simplified formula：
      # samples_seen = (global_step * accumulate_grad_batches * micro_bsz * devices) / generate_every
      #       = (global_step * micro_bsz * devices) / num_iterations
      
      # Note: assumes accumulate_grad_batches == steps_per_generation
      # If not equal, formula needs adjustment. For generality, we use the full formula.
      
      accumulate_grad_batches = self.config.accumulate_grad_batches
      micro_bsz = self.config.micro_bsz
      devices = self.config.devices
      
      samples_seen = (self.global_step * accumulate_grad_batches * micro_bsz * devices) // generate_every
      
      checkpoint['samples_seen'] = samples_seen
      checkpoint['grpo_generate_every'] = generate_every # save this parameter for reference

  def _reload_optimizer_state(self):
    """Delayed loading of optimizer state"""
    if not self.resume_checkpoint_path:
      return
      
    print(f"\n[SWITCH] [Resume] Re-loading optimizer state from: {self.resume_checkpoint_path}")
    
    # Check if this is a pure weights file (e.g. adapter_model.bin)
    try:
      if not os.path.isdir(self.resume_checkpoint_path):
        checkpoint = torch.load(self.resume_checkpoint_path, map_location='cpu')
        if 'optimizer_states' not in checkpoint:
          print("[WARN] [Resume] Checkpoint lacks 'optimizer_states'. Skipping optimizer load (Weights only mode).")
          self.resume_checkpoint_path = None
          return
    except Exception as e:
      print(f"[WARN] [Resume] Error checking checkpoint type: {e}. Proceeding with standard load...")

    try:
      # Use Strategy to load checkpoint (compatible with DeepSpeed and normal mode)
      self.trainer.strategy.load_checkpoint(self.resume_checkpoint_path)
      print("[OK] [Resume] Optimizer state loaded successfully!")
    except Exception as e:
      print(f"[ERROR] [Resume] Failed to load optimizer state: {e}")
      # attempt fallback
      try:
        checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
        if "optimizer_states" in checkpoint:
          self.trainer.strategy.load_optimizer_state_dict(checkpoint)
          print("[OK] [Resume] Optimizer state loaded manually!")
      except Exception as e2:
        print(f"[ERROR] [Resume] Manual load failed too: {e2}")
    
    self.resume_checkpoint_path = None  # Prevent duplicate loading

  
  def setup(self, stage: str):
    """
    Lightning setup hook - indistributedtraininginitializeaftercalluse
    
    Note：at this point DeepSpeed not yet wrapped，placeinnot created here generator
    """
    if stage == 'fit':
      # load tokenizer
      # firstuse Trainer injected tokenizer（ensure ChatML/EOS withtrainingscriptthisalign）
      if not hasattr(self, 'tokenizer') or self.tokenizer is None:
        from rwkvtune.data.tokenizers import get_tokenizer
        self.tokenizer = get_tokenizer()
      
      # [INFO] fix：not created here generator
      # reason：at this pointmodelalsonotwas DeepSpeed wrap，devicecancannotcorrect
      # modifyforinfirst use time（on_train_batch_start）create
      self.generator = None
      
      # Verify reward functions are set (should be set in GRPOTrainer.__init__)
      if not hasattr(self, 'reward_functions') or self.reward_functions is None:
        raise RuntimeError(
          "Reward functions not found! Please provide reward_funcs parameter when creating GRPOTrainer.\n"
          "Reference example:examples/train_grpo_st_cpu.py"
        )
      
      print(f"[OK] [Rank {self.global_rank}] use {len(self.reward_functions)} reward functions")

      if self.config.save_rollout_steps > 0:
        os.makedirs(self.config.save_rollout_path, exist_ok=True)
        if self.global_rank == 0:
          print(f"[OK] Rollout data will be saved to: {self.config.save_rollout_path}")

      if self._use_peft:
        try:
          from rwkvtune.peft.lora.layer import LoraLinear

          if not hasattr(self, '_lora_forward_calls'):
            self._lora_forward_calls = 0
          if not hasattr(self, '_lora_hooked_modules'):
            self._lora_hooked_modules = 0
          if not hasattr(self, '_lora_forward_hook_handles'):
            self._lora_forward_hook_handles = []

          def _lora_forward_hook(_module, _inputs, _outputs):
            self._lora_forward_calls += 1

          for m in self.model.modules():
            if isinstance(m, LoraLinear):
              self._lora_forward_hook_handles.append(m.register_forward_hook(_lora_forward_hook))
              self._lora_hooked_modules += 1

          if self.global_rank == 0:
            print(
              f"[Rank 0] [LORA_HOOK_DIAG] hooked_modules={self._lora_hooked_modules}, "
              f"forward_calls={self._lora_forward_calls}"
            )
        except Exception as e:
          if self.global_rank == 0:
            print(f"[Rank 0] [LORA_HOOK_DIAG] hook failed: {e}")
      
      print(f"[OK] [Rank {self.global_rank}] GRPO groupconditioninitialized")
  
  def forward(self, idx):
    """forwardpropagation"""
    return self.model(idx)

  def _get_lora_diag_sample(self):
    if not getattr(self, '_use_peft', False):
      return None
    try:
      from rwkvtune.peft.lora.layer import LoraLinear

      for m in self.model.modules():
        if isinstance(m, LoraLinear):
          return m
    except Exception:
      return None
    return None

  def on_after_backward(self, *args, **kwargs):
    if not getattr(self, '_use_peft', False):
      return
    if self.trainer is None or not getattr(self.trainer, 'is_global_zero', False):
      return
    counter = int(getattr(self, '_loss_log_counter', 0))
    freq = 50
    if counter != 1 and (counter % freq) != 0:
      return
    sample = self._get_lora_diag_sample()
    if sample is None:
      return
    try:
      a_req = bool(getattr(sample.lora_A, 'requires_grad', False))
      b_req = bool(getattr(sample.lora_B, 'requires_grad', False))
      a_grad = getattr(sample.lora_A, 'grad', None)
      b_grad = getattr(sample.lora_B, 'grad', None)
      a_gmax = float(a_grad.detach().abs().max().item()) if a_grad is not None else None
      b_gmax = float(b_grad.detach().abs().max().item()) if b_grad is not None else None
      print(
        f"[Rank {self.trainer.global_rank}] [LORA_GRAD_DIAG] after_backward(step#{counter}) "
        f"A_req={a_req}, B_req={b_req}, A_grad_absmax={a_gmax}, B_grad_absmax={b_gmax}"
      )
    except Exception:
      pass

  def on_before_backward(self, loss, *args, **kwargs):
    if not getattr(self, '_use_peft', False):
      return
    if self.trainer is None or not getattr(self.trainer, 'is_global_zero', False):
      return
    counter = int(getattr(self, '_loss_log_counter', 0))
    freq = 50
    if counter != 1 and (counter % freq) != 0:
      return
    sample = self._get_lora_diag_sample()
    if sample is None:
      return
    try:
      g = torch.autograd.grad(
        loss,
        sample.lora_B,
        retain_graph=True,
        allow_unused=True,
      )[0]
      gmax = float(g.detach().abs().max().item()) if g is not None else None
      print(
        f"[Rank {self.trainer.global_rank}] [LORA_AUTOGRAD_DIAG] before_backward(step#{counter}) "
        f"lora_B_grad_absmax={gmax}"
      )
    except Exception as e:
      print(
        f"[Rank {self.trainer.global_rank}] [LORA_AUTOGRAD_DIAG] before_backward(step#{counter}) failed: {e}"
      )

  def on_before_optimizer_step(self, optimizer, *args, **kwargs):
    if not getattr(self, '_use_peft', False):
      return
    if self.trainer is None or not getattr(self.trainer, 'is_global_zero', False):
      return
    counter = int(getattr(self, '_loss_log_counter', 0))
    freq = 50
    if counter != 1 and (counter % freq) != 0:
      return
    sample = self._get_lora_diag_sample()
    if sample is None:
      return
    try:
      a_max = float(sample.lora_A.detach().abs().max().item())
      b_max = float(sample.lora_B.detach().abs().max().item())
      print(
        f"[Rank {self.trainer.global_rank}] [LORA_PARAM_DIAG_PRE_OPT] before_opt(step#{counter}) "
        f"lora_A_absmax={a_max:.6e}, lora_B_absmax={b_max:.6e}"
      )
    except Exception:
      pass
  
  def _compute_rewards(
    self,
    prompts: List[str],
    completions: List[str],
    extra_fields: Dict[str, Any]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rewards (follows trl-main design)
    
    calluseplacehaverewardfuncnumberandenterrowsweighted sumand。
    
    Args:
      prompts: List of prompts
      completions: List of completions
      extra_fields: Extra fields from the dataset
    
    Returns:
      total_rewards: Tensor of weighted rewards [B*G]
      rewards_per_func: Tensor of rewards per func [B*G, num_funcs]
    """
    rewards_list = []
    
    for i, (reward_func, weight) in enumerate(zip(self.reward_functions, self.reward_weights)):
      # calluserewardfuncnumber（trl-main style）
      result = reward_func(
        prompts=prompts,
        completions=completions,
        **extra_fields
      )
      
      # convertfor tensor
      if isinstance(result, torch.Tensor):
        rewards = result
      elif isinstance(result, list):
        rewards = torch.tensor(result, dtype=torch.float32)
      else:
        raise ValueError(
          f"rewardfuncnumber {i} returnbacknotsupporttype: {type(result)}\n"
          f"please returnback List[float] or torch.Tensor"
        )
      
      # ensureincorrectdeviceon
      if self.device.type == 'cuda' and not rewards.is_cuda:
        rewards = rewards.to(self.device)
      
      rewards_list.append(rewards)
    
    # Stack all rewards: [B*G, num_funcs]
    rewards_per_func = torch.stack(rewards_list, dim=1)
    
    # Compute weighted sum: [B*G]
    weights_tensor = torch.tensor(self.reward_weights, device=rewards_per_func.device)
    total_rewards = (rewards_per_func * weights_tensor).sum(dim=1)
    
    return total_rewards, rewards_per_func
  
  @staticmethod
  def _make_prompt_batch_signature(batch: Dict[str, Any]) -> tuple:
    """
    Generate a lightweight, stable batch signature to check if batch changed during repeat period.

    We deliberately avoid Python hash(prompt) (may be randomized across processes), instead using reproducible structural info:
    - prompts count
    - length, first token, last token of each sample input_ids
    """
    prompts = batch.get("prompts", None)
    input_ids_list = batch.get("input_ids", None)
    if prompts is None or input_ids_list is None:
      return ("invalid_batch",)

    sig_items = [("B", len(prompts))]
    for ids in input_ids_list:
      if ids is None or len(ids) == 0:
        sig_items.append(("ids", 0, None, None))
      else:
        sig_items.append(("ids", len(ids), int(ids[0]), int(ids[-1])))
    return tuple(sig_items)

  def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int) -> None:
    """
    GRPO Rollout phase - align TRL buffering mechanism（strict semantic version）
    """
    # Dummy Forward: skip Rollout
    if self.global_step < self.skip_until_step:
      return

    config = self.config
    
    # ========== Buffer managelogic ==========
    
    # ref：trl/trainer/grpo_trainer.py#_prepare_inputs
    #
    # core idea：
    # 1) every prompt batch first rollout once，get micro_bsz*num_generations samples（every completion one row）
    # 2) split these samples into steps_per_generation parts write to buffer
    # 3) same prompt batch will in DataLoader be repeated generate_every times（generate_every=steps_per_generation*num_iterations）
    #  - 1 timesrepeat：trigger rollout andwrite to buffer
    #  - subsequentrepeat：onlyconsume buffer slicedotraining（forward/backward），no longer rollout
    # 4) this waycan reduceeverystepsvisiblestore/save（everystepsonlyconsume 1 partsslice），and notwillskip prompt
    #
    # example（steps_per_generation=4, num_generations=4）：
    # - prompt batch A（repeat 4 times）：
    #  * 1 times：rollout get 1 prompt × 4 completions = 4 samples，split into 4 parts (each 1 samples）
    #  * after 3 times：dependtimesuse buffer[1], buffer[2], buffer[3] dotraining
    # - enterenterunderone prompt batch B，repeat againondescribeflow
    #
    # memory optimizationeffectresult：
    # - Without buffer: each step processes B×G samples (e.g. 1×4=4）
    # - With buffer + gradient accumulation: each step processes B×G/steps_per_generation samples (e.g. 4/4=1）
    # - Memory saving: ~75%（for steps_per_generation=4）
    
    config = self.config
    
    # ========== Buffer managelogic ==========
    # judgewhetherneedre-generate（ref TRL）
    # align TRL：steps_per_generation withgradientaccumulatedecouple（defaultcaninexternal settingfor accumulate_grad_batches inkeephistoryrowsfor）
    steps_per_generation = getattr(config, 'steps_per_generation', config.accumulate_grad_batches)
    generate_every = steps_per_generation * config.num_iterations
    # group_offset tableshowwhenbefore training_step in“complex/repeatusewindow”insideoffset：
    # - group_offset==0：thisroundcomplex/repeatusewindowone step，must rollout and fill buffer
    # - group_offset!=0：complex/repeatusewindowinsidesubsequent step，onlycanconsume buffer，should not advancetonew prompts

    # ====== strict validation：sameoneround buffer complex/repeatuseperiod batch must not change（otherwise samples will be skipped）======
    current_sig = self._make_prompt_batch_signature(batch)
    group_offset = self._generation_step % generate_every
    
    should_generate = (
      self._generation_step % generate_every == 0 or 
      len(self._buffered_grpo_data) == 0
    )
    
    if should_generate:
      # If not group_offset==0 but still triggers generate (e.g., buffer empty from checkpoint resume), allow but warn
      if group_offset != 0 and self.trainer.global_rank == 0:
        print(
          f"[Rank {self.trainer.global_rank}] [WARN] buffer empty causing at non-boundary re- rollout："
          f"generation_step={self._generation_step}, generate_every={generate_every}, group_offset={group_offset}"
        )

      # [INFO] key fix：inevery Rollout beforeclean CUDA state，prevent fragmentation and state accumulation
      if self._generation_step > 0: # skiponce
        if self.trainer.global_rank == 0:
          print(f"[Rank {self.trainer.global_rank}] [CLEAN] clean CUDA buffer（preventvisiblestore/savefragmentizationandstateaccumulate）...")
        # force clean memory fragments
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # samestepsplacehave CUDA operatework（ensureofbeforeoperateworkallcomplete）
        if torch.cuda.is_available():
          torch.cuda.synchronize()
      
      print(f"[Rank {self.trainer.global_rank}] [SWITCH] start Rollout phase (generation_step={self._generation_step})...")
      print(f"[Rank {self.trainer.global_rank}]  - steps_per_generation={steps_per_generation}")
      print(f"[Rank {self.trainer.global_rank}]  - num_iterations={config.num_iterations}")
      print(f"[Rank {self.trainer.global_rank}]  - generate_every=steps_per_generation*num_iterations={generate_every}")
      
      # clearold buffer
      self._buffered_grpo_data = []
      self._buffer_step = 0

      # Record this rollout prompt batch signature (for subsequent repeat validation)
      self._active_prompt_batch_signature = current_sig
      self._active_rollout_generation_step = self._generation_step
      
      # ========== generateandwrite to buffer ==========
      # Note：here batch is“prompt batch”（micro_bsz prompt）。
      # align TRL：same prompt batch will in DataLoader be repeated generate_every times，
      # thereforeweonlyneedin 1 timesrepeattime rollout once，and putresultssplit into steps_per_generation partsput into buffer。
      self._generate_and_buffer_data(batch, batch_idx, steps_per_generation)
    else:
      # non rollout steps：mustat“sameone prompt batch repeat window”inside，batch insidecontainmust not change
      if self._active_prompt_batch_signature is None:
        raise RuntimeError(
          "GRPO buffering semantic error: in non-rollout step but no active prompt batch signature recorded.\n"
          "This usually means DataLoader didn't repeat batch, or buffer state was unexpectedly cleared during training."
        )
      if current_sig != self._active_prompt_batch_signature:
        raise RuntimeError(
          "GRPO buffering semantic error：detecttoinsameoneround rollout/buffer complex/repeatuseperiod DataLoader batch happenchangeization，"
          "thiswillcausecausemanysamplesfromnot rollout/fromnottraining。\n"
          f" generation_step={self._generation_step}\n"
          f" generate_every={generate_every}\n"
          f" group_offset(generation_step%generate_every)={group_offset}\n"
          f" active_rollout_generation_step={self._active_rollout_generation_step}\n"
          "fixsuggest：ensure DataLoader toevery prompt batch repeat generate_every times（repeat_batch_count=generate_every），"
          "andin Lightning Trainer set in use_distributed_sampler=False（avoidauto-replaced sampler）。"
        )
    
    # ========== from Buffer getdata ==========
    # getwhenbeforestepscorresponddata
    buffer_idx = self._buffer_step % len(self._buffered_grpo_data)
    self.current_grpo_data = self._buffered_grpo_data[buffer_idx]
    
    print(
      f"[Rank {self.trainer.global_rank}] [DATA] use buffer[{buffer_idx}/{len(self._buffered_grpo_data)}] "
      f"(group_offset={group_offset})"
    )
    
    # updatestepsnumber
    self._buffer_step += 1
    self._generation_step += 1
  
  def _generate_and_buffer_data(
    self, 
    batch: Dict[str, Any], 
    batch_idx: int, 
    steps_per_generation: int
  ) -> None:
    """
    generateandbuffer GRPO data（ref TRL）
    
    will current batch B×G samplessplit into steps_per_generation parts
    Each part contains B*G/steps_per_generation samples
    
    Args:
      batch: whenbefore batch（contains prompts etc）
      batch_idx: batch index
      steps_per_generation: split intomanyfewparts
    """
    config = self.config
    
    # round Rollout phasemonitor
    device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
    with monitor_memory_and_time(self.trainer.global_rank, "[SWITCH] complete Rollout phase（generate+reward+advantage+LogProbs）", device):
      # ========== original Rollout logic ==========
      # switchto eval mode
      print(f"[Rank {self.trainer.global_rank}] [SWITCH] switchto eval mode...")
      self.eval()
      
      # [INFO] [IMPORTANT] All ranks switch mode simultaneously (Lightning handles DeepSpeed sync automatically)
      # No manual broadcast or sync needed, Lightning's DDP/DeepSpeed strategy handles it
      print(f"[Rank {self.trainer.global_rank}] [SWITCH] switchto eval mode...")
      self.eval()
    
      # [INFO] [FIX] All GPUs execute generation (each processes its own batch)
      # This avoids complex broadcast logic and is more compatible with DeepSpeed
      print(f"[Rank {self.trainer.global_rank}] [SWITCH] start Rollout phase (batch_idx={batch_idx})...")
      print(f"[Rank {self.trainer.global_rank}]  - Prompts count: {len(batch['prompts'])}")
      print(f"[Rank {self.trainer.global_rank}]  - every prompt generatenumber: {config.num_generations}")
      # [INFO] debug：print batch key
      if self.trainer.global_rank == 0:
        print(f"[Rank {self.trainer.global_rank}]  - Batch key: {list(batch.keys())}")
        if 'ground_truth_answer' in batch:
          print(f"[Rank {self.trainer.global_rank}]  - ground_truth_answer store/savein: {len(batch['ground_truth_answer']) if isinstance(batch['ground_truth_answer'], list) else 'N/A'} ")
      
      # [INFO] delay create generator（first use time）
      # at this point DeepSpeed alreadyalreadycompletewrap，modelincorrectdeviceon
      if self.generator is None:
        # [RUN] performanceization：use GRPOBatchGenerator replace BatchGenerator
        # GRPOBatchGenerator specialized for GRPO trainingscenarioization，performance improvement 70%+
        from rwkvtune.inference.grpo_batch_generator import GRPOBatchGenerator
        
        # [INFO] Safer model extraction for generation
        # DeepSpeed enginewilloriginalmodelstorein .module attributein
        if hasattr(self.model, 'module'):
          generation_model = self.model.module
          print(f"[Rank {self.trainer.global_rank}] use DeepSpeed unwrapped model")
        elif hasattr(self.model, 'model'):
          # havetimeLightningwillextrawraponelayer
          generation_model = self.model.model
          print(f"[Rank {self.trainer.global_rank}] use Lightning unwrapped model")
        else:
          generation_model = self.model
          print(f"[Rank {self.trainer.global_rank}] use direct model")
        
        # [INFO] ensure model in correct mode
        generation_model.eval()
        
        # [INFO] Verify model state (RWKV7 requires sequence length to be multiple of 16)
        try:
          # Create dummy input meeting RWKV7 requirement (length multiple of 16)
          dummy_input = torch.tensor([[1] * 16], device=next(generation_model.parameters()).device)
          with torch.no_grad():
            _ = generation_model(dummy_input)
          print(f"[Rank {self.trainer.global_rank}] Model forward test passed")
        except Exception as e:
          print(f"[ERROR] Model forward test failed: {e}")
          raise
        
        print(f"[Rank {self.trainer.global_rank}] [RUN] create GRPOBatchGenerator（highperformanceversionthis）...")
        print(f"[Rank {self.trainer.global_rank}]  - model type: {type(generation_model)}")
        print(f"[Rank {self.trainer.global_rank}]  - model device: {next(generation_model.parameters()).device}")
        
        # Get max_prefill_batch_size and max_decode_batch_size config (if any)
        max_prefill_batch_size = getattr(config, 'max_prefill_batch_size', -1)
        max_decode_batch_size = getattr(config, 'max_decode_batch_size', -1)
        
        self.generator = GRPOBatchGenerator(
          model=generation_model,
          tokenizer=self.tokenizer,
          chunk_size=config.prefill_chunk_size,
          max_prefill_batch_size=max_prefill_batch_size,
          max_decode_batch_size=max_decode_batch_size,
          eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
        )
        print(f"[Rank {self.trainer.global_rank}] [OK] GRPOBatchGenerator created")
        print(f"[Rank {self.trainer.global_rank}]  - chunk_size: {config.prefill_chunk_size}")
        print(f"[Rank {self.trainer.global_rank}]  - max_prefill_batch_size: {max_prefill_batch_size} (-1=no limit)")
        print(f"[Rank {self.trainer.global_rank}]  - max_decode_batch_size: {max_decode_batch_size} (-1=no limit)")
      
      # 1. Generate completions (all ranks execute, each processes own batch)
      print(f"[Rank {self.trainer.global_rank}] [RUN] startgenerate completions（use GRPOBatchGenerator）...")
      print(f"[Rank {self.trainer.global_rank}]  - model type: {type(self.model)}")
      print(f"[Rank {self.trainer.global_rank}]  - whetherfor DeepSpeed engine: {hasattr(self.model, 'module')}")
      print(f"[Rank {self.trainer.global_rank}]  - [INFO] generation config details:")
      print(f"[Rank {self.trainer.global_rank}]   * Prompts count: {len(batch['prompts'])}")
      print(f"[Rank {self.trainer.global_rank}]   * max_new_tokens: {config.max_completion_length}")
      print(f"[Rank {self.trainer.global_rank}]   * num_generations: {config.num_generations}")
      print(f"[Rank {self.trainer.global_rank}]   * temperature: {config.temperature}")
      print(f"[Rank {self.trainer.global_rank}]   * top_p: {config.top_p}")
      print(f"[Rank {self.trainer.global_rank}]   * top_k: {config.top_k}")
      print(f"[Rank {self.trainer.global_rank}]   * repetition_penalty: {getattr(config, 'repetition_penalty', 1.0)}")
      total_sequences = len(batch['prompts']) * config.num_generations
      print(f"[Rank {self.trainer.global_rank}] [Rollout] [GRPOBatchGenerator] [RUN] startgenerate {total_sequences} sequence... (startcountedtime)")
      
      # generate（alreadyalreadyin eval mode）
      # [INFO] use input_ids generate（avoid repeat tokenize）
      device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
      with monitor_memory_and_time(self.trainer.global_rank, "[SWITCH] Rollout generation phase", device):
        with torch.no_grad():
          gen_start_ts = time.time()
          # [RUN] use GRPOBatchGenerator directlyinterface（not needed GenerationConfig）
          generation_results = self.generator.generate(
            input_ids=batch['input_ids'],  # already tokenized input
            max_new_tokens=config.max_completion_length,
            num_generations=config.num_generations,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=getattr(config, 'repetition_penalty', 1.0),
            logit_bias=getattr(config, 'logit_bias', None),
          )
          gen_end_ts = time.time()
          elapsed = gen_end_ts - gen_start_ts
          avg_per_seq = elapsed / max(total_sequences, 1)
          print(
            f"[Rank {self.trainer.global_rank}] [Rollout] [GRPOBatchGenerator] [OK] generatecomplete，use time {elapsed:.2f}s，"
            f"avgeverysequence {avg_per_seq:.3f}s"
          )
      print(f"[Rank {self.trainer.global_rank}] [OK] generatecomplete，total {len(generation_results['completions'])} completions")
      
      # [INFO] adddetailedcompletionlengthstatisticscounted
      if len(generation_results['completions']) > 0:
        completion_lengths = [len(c) for c in generation_results['completions']]
        completion_token_lengths = [len(self.tokenizer.encode(c)) for c in generation_results['completions']]
        print(f"[Rank {self.trainer.global_rank}] [STATS] Completion lengthstatisticscounted:")
        print(f"[Rank {self.trainer.global_rank}]  - charlength: min={min(completion_lengths)}, max={max(completion_lengths)}, avg={sum(completion_lengths)/len(completion_lengths):.1f}")
        print(f"[Rank {self.trainer.global_rank}]  - Tokenlength: min={min(completion_token_lengths)}, max={max(completion_token_lengths)}, avg={sum(completion_token_lengths)/len(completion_token_lengths):.1f}")
        print(f"[Rank {self.trainer.global_rank}]  - preperiodmax_length: {config.max_completion_length}")
        if max(completion_token_lengths) < config.max_completion_length * 0.5:
          print(f"[Rank {self.trainer.global_rank}]  [WARN] warn: mostlargetokenlength({max(completion_token_lengths)})much smaller than configuredmax_length({config.max_completion_length})!")
      
      # printone completion workforexample（onlyinone batch and rank 0）
      if batch_idx == 0 and self.trainer.is_global_zero and len(generation_results['completions']) > 0:
        print(f"[Rank 0] [LOG] generateexample:")
        print(f"   Prompt (before100char): {batch['prompts'][0][:100]}...")
        comp1 = generation_results['completions'][0]
        comp1_tokens = len(self.tokenizer.encode(comp1))
        print(f"   Completion 1 (length: {len(comp1)} chars, {comp1_tokens} tokens): {comp1[:200]}...")
        if len(comp1) > 200:
          print(f"   Completion 1 (end100char): ...{comp1[-100:]}")
        if len(generation_results['completions']) > 1:
          comp2 = generation_results['completions'][1]
          comp2_tokens = len(self.tokenizer.encode(comp2))
          print(f"   Completion 2 (length: {len(comp2)} chars, {comp2_tokens} tokens): {comp2[:200]}...")
          if len(comp2) > 200:
            print(f"   Completion 2 (end100char): ...{comp2[-100:]}")
      
      # 2. Compute rewards (all ranks compute their own)
      # Collect extra fields (ref trl-main: extract all non-prompt fields from batch)
      # These fields are auto-passed to reward functions, supporting custom reward functions using extra info from dataset
      extra_fields = {}
      for key, value in batch.items():
        if key not in ['prompts', 'prompt_ids', 'input_ids']: # [INFO] fix：skip input_ids（not neededpass to reward function）
          # Expand to [B*G]（each prompt generates num_generations completions）
          if isinstance(value, list):
            expanded_value = []
            for v in value:
              for _ in range(config.num_generations):
                expanded_value.append(v)
            extra_fields[key] = expanded_value
          else:
            # If not a list, pass directly (will be broadcast)
            extra_fields[key] = value
      
      # [INFO] debug：print extra_fields key（onlyin rank 0）
      if self.trainer.global_rank == 0:
        print(f"[Rank {self.trainer.global_rank}]  extra_fields key: {list(extra_fields.keys())}")
        if 'ground_truth_answer' in extra_fields:
          gt_ans = extra_fields['ground_truth_answer']
          print(f"[Rank {self.trainer.global_rank}]  ground_truth_answer type: {type(gt_ans)}, length: {len(gt_ans) if isinstance(gt_ans, list) else 'N/A'}")
          if isinstance(gt_ans, list) and len(gt_ans) > 0:
            print(f"[Rank {self.trainer.global_rank}]  one ground_truth_answer preview: {str(gt_ans[0])[:100]}...")
      
      # expand prompts to [B*G]
      expanded_prompts = []
      for prompt in batch['prompts']:
        for _ in range(config.num_generations):
          expanded_prompts.append(prompt)
      
      # [INFO] [FIX] Check dimension alignment
      expected_completions = len(batch['prompts']) * config.num_generations
      actual_completions = len(generation_results['completions'])
      if actual_completions != expected_completions:
        raise RuntimeError(
          f"Completions count mismatch!"
          f"Expected: {expected_completions} (prompts={len(batch['prompts'])}, num_generations={config.num_generations}), "
          f"Actual: {actual_completions}。"
          f"This usually means generation phase had issues."
        )
      
      # [INFO] [FIX] Check extra_fields length after expansion
      for key, value in extra_fields.items():
        if isinstance(value, list):
          if len(value) != expected_completions:
            raise RuntimeError(
              f"extra_fields['{key}']Length mismatch!"
              f"Expected: {expected_completions}, Actual: {len(value)}"
            )
      
      # ========== Optional: Completion Post-Processing Hook ==========
      # Called after rollout generation and before reward computation.
      # The hook receives completions, completion_ids, masks, tokenizer, etc.
      # and is fully responsible for returning the modified versions.
      # See completion_postprocess_fn docstring for the input/output contract.
      if self.completion_postprocess_fn is not None:
        print(f"[Rank {self.trainer.global_rank}] [POSTPROCESS] Calling completion_postprocess_fn...")
        postprocess_result = self.completion_postprocess_fn(
          prompts=expanded_prompts,
          completions=list(generation_results['completions']),
          completion_ids=generation_results['completion_ids'],
          masks=generation_results['masks'],
          tokenizer=self.tokenizer,
          **extra_fields
        )
        if not isinstance(postprocess_result, dict):
          raise TypeError(
            f"completion_postprocess_fn must return a dict with keys "
            f"'completions', 'completion_ids', 'masks'. Got {type(postprocess_result).__name__}"
          )
        for required_key in ('completions', 'completion_ids', 'masks'):
          if required_key not in postprocess_result:
            raise KeyError(
              f"completion_postprocess_fn return dict missing required key '{required_key}'. "
              f"Got keys: {list(postprocess_result.keys())}"
            )
        generation_results['completions'] = postprocess_result['completions']
        generation_results['completion_ids'] = postprocess_result['completion_ids']
        generation_results['masks'] = postprocess_result['masks']
        print(f"[Rank {self.trainer.global_rank}] [POSTPROCESS] [OK] Post-processing applied")

      # Compute rewards (follows trl-main design)
      print(f"[Rank {self.trainer.global_rank}] [REWARD] [INFO] Start computing rewards...")
      # [INFO] [DEBUG] Print extra_fields keys to help locate data passing issues
      if self.trainer.global_rank == 0 and len(extra_fields) > 0:
        print(f"[Rank {self.trainer.global_rank}]  extra_fields key: {list(extra_fields.keys())}")
        if 'ground_truth_answer' in extra_fields:
          gt_ans = extra_fields['ground_truth_answer']
          print(f"[Rank {self.trainer.global_rank}]  ground_truth_answer type: {type(gt_ans)}, length: {len(gt_ans) if isinstance(gt_ans, list) else 'N/A'}")
      device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
      with monitor_memory_and_time(self.trainer.global_rank, "[REWARD] Reward Computation Phase", device):
        local_rewards, local_rewards_per_func = self._compute_rewards(
          prompts=expanded_prompts,
          completions=generation_results['completions'],
          extra_fields=extra_fields
        )
      
      # [INFO] [FIX] Check if rewards count is divisible by num_generations
      if local_rewards.shape[0] % config.num_generations != 0:
        error_msg = (
          f"[ERROR] ERROR: local rewards count({local_rewards.shape[0]})must be divisible bynum_generations({config.num_generations}) !\n"
          f"  This will cause advantage computation to fail."
        )
        print(f"[Rank {self.trainer.global_rank}] {error_msg}")
        raise RuntimeError(error_msg)
      
      # [INFO] [DIAG] Statistics of local rewards distribution (frequency limited)
      if not hasattr(self, '_reward_log_counter'):
        self._reward_log_counter = 0
      self._reward_log_counter += 1
      reward_log_frequency = 20 # every20timesrolloutprintonce
      
      if self._reward_log_counter % reward_log_frequency == 0 or self._reward_log_counter == 1:
        unique_rewards, counts = torch.unique(local_rewards, return_counts=True)
        reward_dist = {u.item(): c.item() for u, c in zip(unique_rewards, counts)}
        print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG] Local reward statistics (rollout#{self._reward_log_counter}):")
        print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG]  - Total samples: {local_rewards.shape[0]}")
        print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG]  - Reward mean: {local_rewards.mean().item():.4f}, std: {local_rewards.std().item():.4f}")
        print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG]  - Reward range: [{local_rewards.min().item():.4f}, {local_rewards.max().item():.4f}]")
        print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG]  - Unique reward values count: {len(unique_rewards)}")
        if len(unique_rewards) <= 3:
          print(f"[Rank {self.trainer.global_rank}] [REWARD_DIAG]  [WARN] [WARN] Rewards are mostly binary distribution, may cause advantages=0")
      
      print(f"[Rank {self.trainer.global_rank}] [OK] Reward computation complete, local mean: {local_rewards.mean().item():.4f}")
      
      # 3. [INFO] Gather all rewards (for computing global advantages)
      # This is key to GRPO: advantages need to be computed based on global rewards distribution
      device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
      if self.trainer.world_size > 1:
        print(f"[Rank {self.trainer.global_rank}] [SYNC] Gathering rewards from all ranks...")
        
        # [INFO] [FIX] Check if local_rewards sizes are consistent (all_gather requires all tensors same size)
        local_rewards_size = local_rewards.shape[0]
        # Use all_reduce to collect all rank sizes, check consistency
        size_tensor = torch.tensor([local_rewards_size], dtype=torch.long, device=local_rewards.device)
        all_sizes = [torch.zeros_like(size_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(all_sizes, size_tensor)
        all_sizes_list = [s.item() for s in all_sizes]
        
        if len(set(all_sizes_list)) > 1:
          error_msg = (
            f"[ERROR] ERROR: Distributed training error: local_rewards sizes differ across ranks!\n"
            f"  Rank sizes: {all_sizes_list}\n"
            f"  thiswillcausecauseall_gatherfailedorgroupinsiderelationshipwasbreak。\n"
            f"  Possible cause: different ranks processed different numbers of prompts."
          )
          print(f"[Rank {self.trainer.global_rank}] {error_msg}")
          raise RuntimeError(error_msg)
        
        if self.trainer.global_rank == 0:
          print(f"[Rank 0] [OK] All ranks have consistent local_rewards size: {local_rewards_size}")
        
        with monitor_memory_and_time(self.trainer.global_rank, "[SYNC] All-Gather Phase", device):
          # use all_gather collect all ranks rewards
          all_rewards_list = [torch.zeros_like(local_rewards) for _ in range(self.trainer.world_size)]
          torch.distributed.all_gather(all_rewards_list, local_rewards)
          # concatintoglobal rewards
          global_rewards = torch.cat(all_rewards_list, dim=0)
          
          # samelikeprocess rewards_per_func (forlogrecord）
          all_rewards_per_func_list = [torch.zeros_like(local_rewards_per_func) for _ in range(self.trainer.world_size)]
          torch.distributed.all_gather(all_rewards_per_func_list, local_rewards_per_func)
          global_rewards_per_func = torch.cat(all_rewards_per_func_list, dim=0)
        
        print(f"[Rank {self.trainer.global_rank}] [OK] Gathered rewards: global_size={global_rewards.shape[0]}, global_mean={global_rewards.mean().item():.4f}")
      else:
        # single GPU training
        global_rewards = local_rewards
        global_rewards_per_func = local_rewards_per_func
      
      # 4. countedcomputeadvantage（based onglobal rewards）
      print(f"[Rank {self.trainer.global_rank}] [STATS] [INFO] Start computing advantages...")
      
      # [INFO] [DIAG] Compute group statistics, help locate advantages=0 issue (frequency limited)
      if self.trainer.global_rank == 0:
        if not hasattr(self, '_advantage_log_counter'):
          self._advantage_log_counter = 0
        self._advantage_log_counter += 1
        advantage_log_frequency = 20 # every20timesrolloutprintonce
        
        if self._advantage_log_counter % advantage_log_frequency == 0 or self._advantage_log_counter == 1:
          from rwkvtune.training.grpo.advantage import AdvantageCalculator
          temp_calc = AdvantageCalculator(
            scale_rewards=config.scale_rewards,
            num_generations=config.num_generations
          )
          group_stats = temp_calc.compute_group_statistics(global_rewards)
          num_groups = len(group_stats['group_mean_rewards'])
          zero_std_groups = (group_stats['group_std_rewards'] < 1e-6).sum().item()
          small_std_groups = (group_stats['group_std_rewards'] < 0.01).sum().item()
          
          print(f"[Rank 0] [ADVANTAGE_DIAG] Group statistics diagnostic (rollout#{self._advantage_log_counter}):")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Total groups: {num_groups}")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Zero std dev groups: {zero_std_groups} ({100*zero_std_groups/num_groups:.1f}%)")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Small std dev groups (<0.01): {small_std_groups} ({100*small_std_groups/num_groups:.1f}%)")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Group std range: [{group_stats['group_std_rewards'].min().item():.6f}, {group_stats['group_std_rewards'].max().item():.6f}]")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Group mean range: [{group_stats['group_mean_rewards'].min().item():.4f}, {group_stats['group_mean_rewards'].max().item():.4f}]")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Global reward mean: {group_stats['global_mean_reward'].item():.4f}, std: {group_stats['global_std_reward'].item():.4f}")
          
          if zero_std_groups > num_groups * 0.5:
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [ERROR] Over 50% of groups have zero rewards std dev, this will cause advantages=0!")
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [ERROR] Possible cause: reward func returns binary results, causing same rewards within group")
          elif zero_std_groups > 0:
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [WARN] {zero_std_groups}groups have zero rewards std dev, these groups will have advantages=0")
      
      with monitor_memory_and_time(self.trainer.global_rank, "[STATS] Advantage Computation Phase", device):
        from rwkvtune.training.grpo.advantage import AdvantageCalculator
        advantage_calculator = AdvantageCalculator(
          scale_rewards=config.scale_rewards,
          num_generations=config.num_generations,
          advantage_clip=getattr(config, 'advantage_clip', None),
          low_reward_threshold=getattr(config, 'low_reward_threshold', None),
          low_reward_scale=getattr(config, 'low_reward_scale', 0.01),
        )
        global_advantages = advantage_calculator.compute_advantages(global_rewards)
      
      # [INFO] [FIX] Verify global advantages count is correct
      expected_global_size = local_rewards.shape[0] * self.trainer.world_size if self.trainer.world_size > 1 else local_rewards.shape[0]
      if global_advantages.shape[0] != expected_global_size:
        error_msg = (
          f"[ERROR] ERROR: Global advantages count mismatch!\n"
          f"  Expected: {expected_global_size}, Actual: {global_advantages.shape[0]}\n"
          f"  This will cause subsequent index extraction errors."
        )
        print(f"[Rank {self.trainer.global_rank}] {error_msg}")
        raise RuntimeError(error_msg)
      
      # [INFO] [FIX] Verify group relationship (on rank 0)
      if self.trainer.global_rank == 0:
        # Verify global rewards count is divisible by num_generations
        if global_rewards.shape[0] % config.num_generations != 0:
          error_msg = (
            f"[ERROR] ERROR: Global rewards count({global_rewards.shape[0]})cannot be divided bynum_generations({config.num_generations}) !\n"
            f"  This will break group relationships."
          )
          print(f"[Rank 0] {error_msg}")
          raise RuntimeError(error_msg)
        
        # Verify local_rewards count is consistent across ranks (done in all_gather check above)
        # Only final confirmation here
        num_groups = global_rewards.shape[0] // config.num_generations
        print(f"[Rank 0] [OK] Group relationship verified: total groups={num_groups}, each group with{config.num_generations}completions")
      
      # [INFO] [DIAG] Check if advantages are all zeros (frequency limited)
      if self.trainer.global_rank == 0:
        if self._advantage_log_counter % advantage_log_frequency == 0 or self._advantage_log_counter == 1:
          zero_advantages = (torch.abs(global_advantages) < 1e-6).sum().item()
          zero_ratio = zero_advantages / global_advantages.shape[0]
          print(f"[Rank 0] [ADVANTAGE_DIAG] Advantage statistics:")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Total advantages: {global_advantages.shape[0]}")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Near-zero advantages: {zero_advantages} ({100*zero_ratio:.1f}%)")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Advantage mean: {global_advantages.mean().item():.6f}, std: {global_advantages.std().item():.6f}")
          print(f"[Rank 0] [ADVANTAGE_DIAG]  - Advantage range: [{global_advantages.min().item():.6f}, {global_advantages.max().item():.6f}]")
          
          if zero_ratio > 0.9:
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [ERROR] Over 90% advantages near zero, model cannot learn!")
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [ERROR] Please check reward func and group rewards distribution")
          elif zero_ratio > 0.5:
            print(f"[Rank 0] [ADVANTAGE_DIAG]  [WARN] [WARN] exceed50%advantagesclose to0，learning signal is weak")
      
      print(f"[Rank {self.trainer.global_rank}] [OK] Advantage computation complete, global mean: {global_advantages.mean().item():.4f}, global size: {global_advantages.shape[0]}")
      
      # 5. Extract this rank's advantages
      if self.trainer.world_size > 1:
        local_batch_size = local_rewards.shape[0]
        start_idx = self.trainer.global_rank * local_batch_size
        end_idx = start_idx + local_batch_size
        
        # [INFO] [FIX] Check index bounds
        if end_idx > global_advantages.shape[0]:
          error_msg = (
            f"[ERROR] ERROR: Index out of bounds when extracting advantages!\n"
            f"  Rank {self.trainer.global_rank}: [{start_idx}:{end_idx}],\n"
            f"  but global_advantages only has{global_advantages.shape[0]}elements.\n"
            f"  This usually means all_gather phase had issues."
          )
          print(f"[Rank {self.trainer.global_rank}] {error_msg}")
          raise RuntimeError(error_msg)
        
        advantages = global_advantages[start_idx:end_idx]
        
        # [INFO] [FIX] Check if advantages count matches
        if advantages.shape[0] != local_batch_size:
          error_msg = (
            f"[ERROR] ERROR: Extracted advantages count mismatch!\n"
            f"  Expected: {local_batch_size}, Actual: {advantages.shape[0]}"
          )
          print(f"[Rank {self.trainer.global_rank}] {error_msg}")
          raise RuntimeError(error_msg)
        
        rewards = local_rewards # Use local rewards for logging
        rewards_per_func = local_rewards_per_func
        print(f"[Rank {self.trainer.global_rank}] [OK] Extract local advantages: [{start_idx}:{end_idx}], local mean: {advantages.mean().item():.4f}")
      else:
        advantages = global_advantages
        rewards = global_rewards
        rewards_per_func = global_rewards_per_func
      
      # 6. Process prompt_ids (for training_step policy log probability computation)
      # [INFO] Use existing input_ids from batch, no repeated tokenization
      # batch['input_ids'] is list of lists，needconvertfor tensor
      max_prompt_length = config.max_prompt_length
      prompt_ids_list = []
      
      for input_id in batch['input_ids']:
        # Truncate to max_prompt_length
        if len(input_id) > max_prompt_length:
          input_id = input_id[:max_prompt_length]
        prompt_ids_list.append(torch.tensor(input_id, dtype=torch.long))
      
      # Pad to same length
      max_prompt_len = max(ids.shape[0] for ids in prompt_ids_list)
      # Ensure doesn't exceed max_prompt_length
      max_prompt_len = min(max_prompt_len, max_prompt_length)
      padded_prompt_ids = []
      for ids in prompt_ids_list:
        pad_len = max_prompt_len - ids.shape[0]
        if pad_len > 0:
          padded_prompt_ids.append(torch.nn.functional.pad(ids, (0, pad_len), value=0))
        else:
          padded_prompt_ids.append(ids)
      
      prompt_ids = torch.stack(padded_prompt_ids).to(self.device) # [B, P]
      print(f"[Rank {self.trainer.global_rank}] [OK] Using input_ids from batch: {prompt_ids.shape}")

      # [RUN] [TRL Aligned] Expand prompt_ids in rollout phase so each completion has one row
      # So subsequent sample shuffling won't break prompt/completion alignment
      prompt_ids_expanded = prompt_ids.repeat_interleave(config.num_generations, dim=0) # [B*G, P]
      
      # Get completion data
      completion_ids = generation_results['completion_ids'].to(self.device)
      masks = generation_results['masks'].to(self.device)
      
      # [INFO] Compute sampling log probabilities
      # GRPOBatchGenerator doesn't return completion_logps (for performance), so always need to compute here
      # Required for GRPO importance sampling
      completion_logps = generation_results.get('completion_logps')
      if completion_logps is None or completion_logps.numel() == 0:
        print(f"[Rank {self.trainer.global_rank}] [STATS] Compute sampling log probabilities...")
        # Use current model to compute log probs (in eval mode)
        # [WARN] This is one of the most memory-intensive phases
        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
        with monitor_memory_and_time(self.trainer.global_rank, "[STATS] Sampling Log Probabilities Phase (memory intensive)", device):
          with torch.no_grad():
            completion_logps = self._compute_per_token_logps(prompt_ids_expanded, completion_ids)
        print(f"[Rank {self.trainer.global_rank}] [OK] Sampling log probs computed: {completion_logps.shape}")
      else:
        completion_logps = completion_logps.to(self.device)

      # Shape validation (TRL aligned: unified processing by sample dimension)
      if prompt_ids_expanded.shape[0] != completion_ids.shape[0]:
        raise RuntimeError(
          f"Prompt/completion count mismatch:"
          f"prompt_ids_expanded={prompt_ids_expanded.shape}, completion_ids={completion_ids.shape}. "
          f"Please check if generator output is B*G."
        )
      
      # Pack GRPO data (ensure all tensors on correct device)
      grpo_data = {
        'prompt_ids': prompt_ids_expanded, # [B*G, P]（align TRL：every completion one row）
        'completion_ids': completion_ids,
        'completion_logps': completion_logps,
        'masks': masks,
        'rewards': rewards.to(self.device),
        'advantages': advantages.to(self.device),
        # Save strings (for loss_mask_func)
        'prompts': expanded_prompts, # List[str], length B*G
        'completions': generation_results['completions'], # List[str], length B*G
      }

      # ========== [TRL Aligned] Shuffle before writing to buffer (like trl.trainer.utils.shuffle_sequence_dict)==========
      if getattr(config, 'shuffle_buffer', True):
        total_samples = grpo_data['completion_ids'].shape[0]
        perm = torch.randperm(total_samples, device=grpo_data['completion_ids'].device)
        perm_cpu = perm.cpu().tolist() # for shuffle charstring list
        grpo_data['prompt_ids'] = grpo_data['prompt_ids'][perm]
        grpo_data['completion_ids'] = grpo_data['completion_ids'][perm]
        grpo_data['completion_logps'] = grpo_data['completion_logps'][perm]
        grpo_data['masks'] = grpo_data['masks'][perm]
        grpo_data['rewards'] = grpo_data['rewards'][perm]
        grpo_data['advantages'] = grpo_data['advantages'][perm]
        # Apply same shuffle to string lists
        grpo_data['prompts'] = [grpo_data['prompts'][i] for i in perm_cpu]
        grpo_data['completions'] = [grpo_data['completions'][i] for i in perm_cpu]
        print(f"[Rank {self.trainer.global_rank}] [SHUFFLE] [OK] Shuffled rollout samples (total={total_samples}）")
      
      # ========== TRL-style split logic ==========
      # Split B*G samples into steps_per_generation parts
      # Each part contains B*G/steps_per_generation samples
      total_samples = grpo_data['completion_ids'].shape[0] # total samples of completion-related data (=B*G)
      
      samples_per_step = total_samples // steps_per_generation
      if total_samples % steps_per_generation != 0:
        print(f"[Rank {self.trainer.global_rank}] [WARN] [WARN] Total samples {total_samples} cannot be divided by {steps_per_generation} divide")
        samples_per_step = max(1, samples_per_step)
      
      print(f"[Rank {self.trainer.global_rank}] [SPLIT] [INFO] Splitting data: {total_samples} samples -> {steps_per_generation} parts (each {samples_per_step} ）")
      
      # Split and cache
      for i in range(steps_per_generation):
        start_idx = i * samples_per_step
        end_idx = min(start_idx + samples_per_step, total_samples)
        
        sub_grpo_data = {
          # [TRL Aligned] Slice by sample dimension (each completion one row)
          'prompt_ids': grpo_data['prompt_ids'][start_idx:end_idx], # [N, P]
          'completion_ids': grpo_data['completion_ids'][start_idx:end_idx], # [N, C]
          'completion_logps': grpo_data['completion_logps'][start_idx:end_idx],
          'masks': grpo_data['masks'][start_idx:end_idx],
          'rewards': grpo_data['rewards'][start_idx:end_idx],
          'advantages': grpo_data['advantages'][start_idx:end_idx],
          # String lists also sliced the same way
          'prompts': grpo_data['prompts'][start_idx:end_idx],
          'completions': grpo_data['completions'][start_idx:end_idx],
        }
        
        self._buffered_grpo_data.append(sub_grpo_data)
        print(f"[Rank {self.trainer.global_rank}]  Buffer[{i}]: prompt={sub_grpo_data['prompt_ids'].shape}, completion={sub_grpo_data['completion_ids'].shape}")
      
      print(f"[Rank {self.trainer.global_rank}] [OK] [OK] Data split complete, total {len(self._buffered_grpo_data)} parts")

      # ========== Collect metrics and save data (ref trl-main, rank 0 only)==========
      if self.trainer.is_global_zero:
        # Compute statistics (using global data)
        global_completion_lengths = generation_results['masks'].sum(dim=1) if self.trainer.world_size == 1 else None # [B*G]
        
        # Collect overall Rewards metrics（useglobal rewards）
        self._metrics['reward/mean'].append(global_rewards.mean().item())
        self._metrics['reward/std'].append(global_rewards.std().item())
        self._metrics['reward/min'].append(global_rewards.min().item())
        self._metrics['reward/max'].append(global_rewards.max().item())
        
        # collectevery Reward func individual metrics（ref trl-main）
        for i, reward_func_name in enumerate(self.reward_func_names):
          func_rewards = global_rewards_per_func[:, i] # [global B*G]
          self._metrics[f'rewards/{reward_func_name}/mean'].append(func_rewards.mean().item())
          self._metrics[f'rewards/{reward_func_name}/std'].append(func_rewards.std().item())
          self._metrics[f'rewards/{reward_func_name}/min'].append(func_rewards.min().item())
          self._metrics[f'rewards/{reward_func_name}/max'].append(func_rewards.max().item())
        
        # Note：no longerrecord advantage/mean with advantage/std。
        # reason：when scale_rewards='group'（default GRPO）time，advantages willwasgroupinsidestandardization，
        # its overall std tends to approach 1、mean tends to approach 0，infovery lowandcontaineasily misleadcausereadreaderjudgetrainingwhetherhaveeffect。
        
        # collect Completions metric（onlysingle GPU timecanuse）
        if global_completion_lengths is not None:
          self._metrics['completions/mean_length'].append(global_completion_lengths.float().mean().item())
          self._metrics['completions/min_length'].append(global_completion_lengths.float().min().item())
          self._metrics['completions/max_length'].append(global_completion_lengths.float().max().item())
        
        # save rollout data（e.g.resultneed，onlysavelocaldata）
        #
        # heavyneed：rollout data only generated in“happen rollout step”generated，thereforesavecan only happen atthissome step。
        # oldlogicneedrequest training_step_count mustexactlydivide save_rollout_steps，containprone to：
        #  save_rollout_steps=100，but rollout every 16 step once → onlyhave step 0 and step 400 onlywillsave
        # newlogic：onlyneeddistance from lasttimessave training_step_count >= save_rollout_steps，theninthistimes rollout timesaveonce。
        should_save_rollout = False
        if config.save_rollout_steps > 0:
          if self._last_saved_rollout_training_step_count is None:
            should_save_rollout = True
          else:
            should_save_rollout = (
              self.training_step_count - self._last_saved_rollout_training_step_count
            ) >= config.save_rollout_steps

        if should_save_rollout:
          rollout_data = {
            'prompts': expanded_prompts,
            'completions': generation_results['completions'],
            'completion_ids': completion_ids.cpu(), # [B*G, C] - forgenerate masked_completion
            'rewards': rewards.cpu().numpy(),
            'rewards_per_func': rewards_per_func.cpu().numpy(), # [B*G, num_funcs]
            'advantages': advantages.cpu().numpy(),
            'ground_truth_scores': extra_fields.get('ground_truth_scores', None),
            'ground_truth': extra_fields.get('ground_truth', None), # [OK] trueactual completion
            'ground_truth_answer': extra_fields.get('ground_truth_answer', None), # [INFO] add：GSM8K standardanswersolution
            'original_data': extra_fields.get('original_data', None),
          }
          self._save_rollout_data(rollout_data, self.training_step_count)
          self._last_saved_rollout_training_step_count = self.training_step_count
    
      # [INFO] heavyneed：placehave rank allcutback train mode（Lightning willautosamesteps）
      print(f"[Rank {self.trainer.global_rank}] [SWITCH] [INFO] Switching back to train mode...")
      self.train()
      
      # [INFO] samestepsplacehaveenterprogram（ensureplacehave ranks allcomplete rollout）
      if self.trainer.world_size > 1:
        print(f"[Rank {self.trainer.global_rank}] [SWITCH] [INFO] Waiting for all processes to sync...")
        self.trainer.strategy.barrier()
        print(f"[Rank {self.trainer.global_rank}] [OK] [OK] Sync complete, starting training steps...")
      
      print(f"[Rank {self.trainer.global_rank}] [OK] [OK] Rollout phase complete\n")
      
      # [INFO] Rollout completeafterimmediatelycleanvisiblestore/save，forsubsequent training_step free upemptyduring
      gc.collect()
      torch.cuda.empty_cache()
  
  def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
    """
    GRPO strategyupdatephase - useprecountedcompute GRPO data
    """
    # 1. Dummy Forward logic：skipalreadytrainingstepsnumber
    if self.global_step < self.skip_until_step:
      # returnback Dummy Loss (requires_grad=True fool Lightning)
      return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    # 2. delayloadizer：infaketrainingend、truetrainingstartthatonemomentload
    if self.global_step == self.skip_until_step and self.resume_checkpoint_path:
      self._reload_optimizer_state()

    device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else None
    
    # [INFO] add new：record training_step starttimevisiblestore/savestate
    start_mem = get_gpu_memory_info(device) if torch.cuda.is_available() else None
    if start_mem:
      print(f"[Rank {self.trainer.global_rank}] [INFO] enterenter training_step (batch_idx={batch_idx})...")
      print(f"[Rank {self.trainer.global_rank}]  [MEM] startvisiblestore/save: {start_mem['allocated']:.2f} GB (userate: {start_mem['usage_percent']:.1f}%)")
    else:
      print(f"[Rank {self.trainer.global_rank}] [INFO] enterenter training_step (batch_idx={batch_idx})...")
    
    # round training_step monitor
    with monitor_memory_and_time(self.trainer.global_rank, "[INFO] Training Step (Forward + Backward)", device):
      # statisticsone loss diagcountednumberizer：subsequentmanydiagmoduleswilldependrely on it
      # must be inanyuse self._loss_log_counter placeofbeforeinitialize
      if not hasattr(self, '_loss_log_counter'):
        self._loss_log_counter = 0
      self._loss_log_counter += 1
      loss_log_frequency = 50 # every50stepsprintonce

      # getgetprecountedcompute GRPO data（from buffer）
      grpo_data = self.current_grpo_data
      if grpo_data is None:
        raise RuntimeError("GRPO data not computed! Check on_train_batch_start")

      # align TRL：buffer insidepartalreadyalreadyisby sample expand prompt_ids（every completion one row）
      prompt_ids_expanded = grpo_data['prompt_ids'] # [N, P]

      # completion countcancanis notcomplete B×G（becausefor buffer split，everystepsonlygetonepartsamples）
      completion_ids = grpo_data['completion_ids'] # [N, C]
      actual_num_samples = completion_ids.shape[0]

      if prompt_ids_expanded.shape[0] != actual_num_samples:
        raise RuntimeError(
          f"GRPO buffer datainconsistent：prompt_ids={prompt_ids_expanded.shape}, completion_ids={completion_ids.shape}"
        )

      print(f"[Rank {self.trainer.global_rank}]  - datashape: N={actual_num_samples}")
      print(f"[Rank {self.trainer.global_rank}]  - prompt_ids={prompt_ids_expanded.shape}, completion_ids={completion_ids.shape}")
      
      # from GRPO dataget other variables from
      sampling_logps = grpo_data['completion_logps']  # [N, C]
      advantages = grpo_data['advantages']       # [N]
      mask = grpo_data['masks']             # [N, C]
      rewards = grpo_data['rewards']          # [N]
      
      # 1. countedcomputestrategymodel log probability（useexpandafter prompt_ids）
      # [INFO] key fix：forceuse logprob_batch_size enterrowschunkcountedcompute，avoid OOM
      logprob_batch_size = getattr(self.config, 'logprob_batch_size', None)
      print(f"[Rank {self.trainer.global_rank}]  - countedcomputestrategymodel log probability...")
      print(f"[Rank {self.trainer.global_rank}]  - usechunklargesmall: {logprob_batch_size} (None=no chunking)")
      print(f"[Rank {self.trainer.global_rank}]  - Total samples: {actual_num_samples}")

      # [INFO] LoRA adaptizerstateprotect：incountedcompute policy_logps beforevisiblemodeenableuse adapter
      # target：avoidononesteps ref_logps exceptioncausecause adapter statenotrestorecomplex/repeat，fromandlet policy==ref
      adapter_model = self.model
      if hasattr(adapter_model, 'module'):
        adapter_model = adapter_model.module
      elif hasattr(adapter_model, 'model'):
        adapter_model = adapter_model.model
      if self._use_peft and hasattr(adapter_model, 'enable_adapter'):
        try:
          adapter_model.enable_adapter()
        except Exception:
          pass
      
      with monitor_memory_and_time(self.trainer.global_rank, "[INFO] strategy Log Probabilities countedcalculation phase（[WARN] visiblestore/saveintensive）", device):
        # [INFO] forceenablegradient：preventonmisleaduse no_grad/inference_mode causecause policy_logps detachcountedcomputegraph
        with torch.enable_grad():
          # [INFO] visiblemodepass in batch_size paramnumber，ensurechunkgenerateeffect
          policy_logps = self._compute_per_token_logps_with_model(
            self.model, 
            prompt_ids_expanded, 
            completion_ids,
            batch_size=logprob_batch_size # forceuseconfigchunklargesmall
          )

      # [INFO] keydiag：policy_logps must be incountedcomputegraphin，otherwise LoRA/strategyparamnumbernotwillgetgradient
      if self._use_peft and not getattr(policy_logps, 'requires_grad', False):
        if self.trainer.is_global_zero:
          print(
            f"[Rank {self.trainer.global_rank}] [ERROR] [GRAD_DIAG] policy_logps.requires_grad=False, "
            f"torch.is_grad_enabled()={torch.is_grad_enabled()}"
          )
        raise RuntimeError("policy_logps does not require grad; policy update will not train any parameters")
      print(f"[Rank {self.trainer.global_rank}]  [OK] strategy log probabilitycountedcomputecomplete: shape={policy_logps.shape}")
      
      # 2. countedcomputerefmodel log probability（e.g.resultneed）
      ref_logps = None
      if self.config.beta > 0.0:
        print(f"[Rank {self.trainer.global_rank}]  - countedcomputerefmodel log probability (beta={self.config.beta})...")
        if self._use_peft:
          # use LoRA：disableuse adapter fromgetgetrefmodeloutput
          with torch.no_grad():
            with monitor_memory_and_time(self.trainer.global_rank, "[INFO] refmodel Log Probabilities (LoRA)", device):
              # [INFO] heavyneed：DeepSpeed/Lightning cancanwillwrapmodel，adapter statemustworkuseintruecorrect base module on
              adapter_model = self.model
              if hasattr(adapter_model, 'module'):
                adapter_model = adapter_model.module
              elif hasattr(adapter_model, 'model'):
                adapter_model = adapter_model.model

              if not hasattr(adapter_model, 'disable_adapter'):
                raise RuntimeError(
                  "use_peft=True butmodellackfew disable_adapter method，cannotcountedcompute reference logps"
                )

              adapter_model.disable_adapter()
              try:
                ref_logps = self._compute_per_token_logps_with_model(
                  self.model, 
                  prompt_ids_expanded, 
                  completion_ids,
                  batch_size=logprob_batch_size
                )
              finally:
                try:
                  adapter_model.enable_adapter()
                except Exception:
                  pass
          print(f"[Rank {self.trainer.global_rank}]  [OK] refmodel log probabilitycountedcomputecomplete (LoRA)")
        elif self.ref_model is not None:
          # useindependentrefmodel
          with torch.no_grad():
            with monitor_memory_and_time(self.trainer.global_rank, "[INFO] refmodel Log Probabilities", device):
              ref_logps = self._compute_per_token_logps_with_model(
                self.ref_model, 
                prompt_ids_expanded, 
                completion_ids,
                batch_size=logprob_batch_size # samelikeusechunk
              )
          print(f"[Rank {self.trainer.global_rank}]  [OK] refmodel log probabilitycountedcomputecomplete")

      if (
        self.trainer.is_global_zero
        and self._use_peft
        and (self._loss_log_counter % loss_log_frequency == 0 or self._loss_log_counter == 1)
      ):
        pass
      
      # 3. shoulduse custom loss mask（e.g.resultuseuserprovideprovide loss_mask_func）
      final_mask = mask # defaultuseoriginal completion mask
      if self.loss_mask_func is not None:
        print(f"[Rank {self.trainer.global_rank}]  - shoulduse custom loss mask...")
        try:
          custom_mask = self.loss_mask_func(
            prompts=grpo_data.get('prompts'),
            completions=grpo_data.get('completions'),
            completion_ids=completion_ids,
            tokenizer=self.tokenizer,
          )
          # verify custom_mask shape
          if custom_mask.shape != mask.shape:
            raise ValueError(
              f"loss_mask_func returnback mask shapemismatch：\n"
              f" Expected: {mask.shape}\n"
              f" get: {custom_mask.shape}"
            )
          
          # andoriginal mask do AND（onlyto"non padding anduseuserspecify" token countedcompute loss）
          final_mask = mask * custom_mask
          
          # protect：e.g.result custom_mask causecauseall 0（willcausecause loss=nan），backexit/backtokeepmostafter N token
          if final_mask.sum() == 0:
            print(f"[Rank {self.trainer.global_rank}]  [WARN] custom_mask causecauseallfor 0，backexit/backtokeepmostafter 10 token")
            # toeverysamples，keepmostafter min(10, actuallength) token
            for i in range(final_mask.shape[0]):
              valid_count = mask[i].sum().int().item()
              if valid_count > 0:
                keep_count = min(10, valid_count)
                # findtomostafter keep_count haveeffect token position
                valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
                if len(valid_indices) >= keep_count:
                  final_mask[i, valid_indices[-keep_count:]] = 1.0
          
          masked_tokens = final_mask.sum().item()
          total_tokens = mask.sum().item()
          print(f"[Rank {self.trainer.global_rank}]  [OK] Custom mask shouldusecomplete: {masked_tokens:.0f} / {total_tokens:.0f} tokens paramwith loss countedcompute ({masked_tokens/total_tokens*100:.1f}%)")
        except Exception as e:
          print(f"[Rank {self.trainer.global_rank}]  [WARN] loss_mask_func executerowsfailed: {e}")
          print(f"[Rank {self.trainer.global_rank}]  backexit/backtouseoriginal completion mask")
          final_mask = mask
      
      # 4. countedcomputeevery token loss（use final_mask）
      per_token_loss = self.loss_fn.compute_per_token_loss(
        logps_policy=policy_logps,
        logps_ref=ref_logps,
        logps_sampling=sampling_logps,
        advantages=advantages,
        mask=final_mask,
        beta=self.config.beta
      )
      
      # 5. aggregate loss
      loss = self.loss_fn.compute_loss(per_token_loss, final_mask)
      
      # 6. countedcomputemetricforlog
      with torch.no_grad():
        # heavyneedpropertysampleratiorate
        log_ratio = policy_logps - sampling_logps
        ratio = torch.exp(log_ratio)
        
        # Clip ratio（wasclip ratio）
        # Note：using final_mask countedcomputemetric（and loss keeponecause）
        clip_low = 1.0 - self.config.epsilon
        clip_high = 1.0 + self.config.epsilon_high
        clipped = ((ratio < clip_low) | (ratio > clip_high)).float()
        clip_ratio = (clipped * final_mask).sum() / final_mask.sum() if final_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # [INFO] diag：checklossandadvantagesrelationship（frequencylimit）
        if self._loss_log_counter % loss_log_frequency == 0 or self._loss_log_counter == 1:
          pass
        
        # KL divergence（e.g.resulthaverefmodel）
        # Note：using final_mask countedcomputemetric（and loss keeponecause）
        # [INFO] fix：usecorrect KL divergenceformula（with TRL align）
        # KL(π || π_ref) ≈ exp(log_ref - log_policy) - (log_ref - log_policy) - 1
        # thisis f-divergence a form of，always >= 0
        # ref：TRL grpo_trainer.py 1732-1734 rows
        if ref_logps is not None:
          log_ratio = ref_logps - policy_logps # [B*G, C]
          per_token_kl = torch.exp(log_ratio) - log_ratio - 1
          kl_div = (per_token_kl * final_mask).sum() / final_mask.sum() if final_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
        else:
          # [INFO] fix：ensure tensor incorrectdeviceon（avoid NCCL error）
          kl_div = torch.tensor(0.0, device=self.device)

        if (
          self.trainer.is_global_zero
          and (self._loss_log_counter % loss_log_frequency == 0 or self._loss_log_counter == 1)
        ):
          masked = final_mask.bool()
          if ref_logps is not None and masked.any():
            diff = policy_logps[masked] - ref_logps[masked]
            diff_mean_abs = float(diff.abs().mean().item())
            diff_max_abs = float(diff.abs().max().item())
          else:
            diff_mean_abs = 0.0
            diff_max_abs = 0.0

          ratio_masked = ratio[masked] if masked.any() else None
          ratio_min = float(ratio_masked.min().item()) if ratio_masked is not None and ratio_masked.numel() > 0 else 0.0
          ratio_max = float(ratio_masked.max().item()) if ratio_masked is not None and ratio_masked.numel() > 0 else 0.0

          mean_advantage = float(advantages.mean().item())
          std_advantage = float(advantages.std().item())
          zero_advantage_ratio = float((torch.abs(advantages) < 1e-6).float().mean().item())

          lora_a_max = None
          lora_b_max = None
          lora_disabled = None
          lora_fwd_calls = int(getattr(self, '_lora_forward_calls', 0))
          lora_hooked = int(getattr(self, '_lora_hooked_modules', 0))
          lora_b_gmax = None
          if getattr(self, '_use_peft', False):
            sample = self._get_lora_diag_sample()
            if sample is not None:
              try:
                lora_a_max = float(sample.lora_A.detach().abs().max().item())
                lora_b_max = float(sample.lora_B.detach().abs().max().item())
                lora_disabled = bool(getattr(sample, 'disabled', False))
              except Exception:
                pass
              try:
                g = torch.autograd.grad(
                  loss,
                  sample.lora_B,
                  retain_graph=True,
                  allow_unused=True,
                )[0]
                lora_b_gmax = float(g.detach().abs().max().item()) if g is not None else None
              except Exception:
                lora_b_gmax = None

          print(
            "\n".join(
              [
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] step#{self._loss_log_counter}",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] loss={float(loss.item()):.6f} clip_ratio={float(clip_ratio.item()):.6f} ratio_range=[{ratio_min:.6f},{ratio_max:.6f}]",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] adv_mean={mean_advantage:.6f} adv_std={std_advantage:.6f} adv_zero_ratio={100.0*zero_advantage_ratio:.2f}%",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] kl_div={float(kl_div.item()):.8e} mean|logp_diff|={diff_mean_abs:.8e} max|logp_diff|={diff_max_abs:.8e}",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] policy_logps.requires_grad={bool(getattr(policy_logps, 'requires_grad', False))}",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] lora_hooked={lora_hooked} lora_forward_calls={lora_fwd_calls}",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] lora_A_absmax={lora_a_max} lora_B_absmax={lora_b_max} lora_disabled={lora_disabled}",
                f"[Rank {self.trainer.global_rank}] [GRPO_DIAG] lora_B_autograd_absmax={lora_b_gmax}",
              ]
            )
          )
        
        # avgadvantage
        mean_advantage = advantages.mean()
        
        # avgreward
        mean_reward = rewards.mean()
        
        # rewardstandarddiff（ref TRL）
        reward_std = rewards.std()
        
        # Clip ratio detailedstatisticscounted（ref TRL）
        is_low_clipped = (ratio < clip_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (ratio > clip_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        
        def masked_batch_mean(x):
          """countedcomputemaskafteravgvalue（align TRL）"""
          if final_mask.sum() > 0:
            return (x * final_mask).sum() / final_mask.sum()
          else:
            return torch.tensor(0.0, device=self.device)
        
        low_clip_ratio = masked_batch_mean(is_low_clipped.float())
        high_clip_ratio = masked_batch_mean(is_high_clipped.float())
        region_clip_ratio = masked_batch_mean(is_region_clipped.float())
        
        # countedcomputeentropy（ref TRL）- needre-getget logits，butforsavevisiblestore/save，onlyinneedtimecountedcompute
        # Note：hereweusestrategymodel logits fromcountedcomputeentropy
        # forsavevisiblestore/save，weonlyinrecordlogtimecountedcompute（is notevery step allcountedcompute）
        mean_entropy = torch.tensor(0.0, device=self.device) # default value
        if getattr(self.config, 'log_entropy', True):
          try:
            # re-countedcompute logits（onlycountedcomputeonce，forentropy）
            # forsavevisiblestore/save，weusecomparesmall batch_size
            entropy_batch_size = min(16, actual_num_samples) if logprob_batch_size is None else min(logprob_batch_size, actual_num_samples)
            
            # onlycountedcomputebefore entropy_batch_size samplesentropy（savecountedcompute）
            if actual_num_samples <= entropy_batch_size:
              # small batch，directlycountedcompute
              input_ids_for_entropy = torch.cat([prompt_ids_expanded, completion_ids[:, :-1]], dim=1)
              seq_len = input_ids_for_entropy.shape[1]
              CHUNK_LEN = 16
              pad_len = 0
              if seq_len % CHUNK_LEN != 0:
                pad_len = CHUNK_LEN - (seq_len % CHUNK_LEN)
                padding = torch.zeros(
                  input_ids_for_entropy.shape[0],
                  pad_len,
                  dtype=input_ids_for_entropy.dtype,
                  device=input_ids_for_entropy.device,
                )
                input_ids_for_entropy = torch.cat([input_ids_for_entropy, padding], dim=1)
              logits_keep = completion_ids.shape[1] + pad_len
              logits_for_entropy = self.model(input_ids_for_entropy, logits_to_keep=logits_keep)
              if pad_len > 0:
                logits_for_entropy = logits_for_entropy[:, : completion_ids.shape[1]]
              from rwkvtune.training.grpo.utils import entropy_from_logits
              entropies = entropy_from_logits(logits_for_entropy) # [B*G, C]
              mean_entropy = masked_batch_mean(entropies)
              del logits_for_entropy, entropies
              torch.cuda.empty_cache()
            else:
              # large batch，onlycountedcomputepartsamplesentropy（estimate）
              sample_indices = torch.randperm(actual_num_samples, device=self.device)[:entropy_batch_size]
              prompt_ids_sample = prompt_ids_expanded[sample_indices]
              completion_ids_sample = completion_ids[sample_indices]
              mask_sample = final_mask[sample_indices]
              
              input_ids_for_entropy = torch.cat([prompt_ids_sample, completion_ids_sample[:, :-1]], dim=1)
              seq_len = input_ids_for_entropy.shape[1]
              CHUNK_LEN = 16
              pad_len = 0
              if seq_len % CHUNK_LEN != 0:
                pad_len = CHUNK_LEN - (seq_len % CHUNK_LEN)
                padding = torch.zeros(
                  input_ids_for_entropy.shape[0],
                  pad_len,
                  dtype=input_ids_for_entropy.dtype,
                  device=input_ids_for_entropy.device,
                )
                input_ids_for_entropy = torch.cat([input_ids_for_entropy, padding], dim=1)
              logits_keep = completion_ids_sample.shape[1] + pad_len
              logits_for_entropy = self.model(input_ids_for_entropy, logits_to_keep=logits_keep)
              if pad_len > 0:
                logits_for_entropy = logits_for_entropy[:, : completion_ids_sample.shape[1]]
              from rwkvtune.training.grpo.utils import entropy_from_logits
              entropies = entropy_from_logits(logits_for_entropy) # [batch_size, C]
              mean_entropy = (entropies * mask_sample).sum() / mask_sample.sum() if mask_sample.sum() > 0 else torch.tensor(0.0, device=self.device)
              del logits_for_entropy, entropies
              torch.cuda.empty_cache()
          except Exception as e:
            # e.g.resultcountedcomputeentropyfailed，usedefault value 0（not affecttraining）
            if self.trainer.global_rank == 0:
              print(f"[Rank 0] [WARN] countedcomputeentropyfailed: {e}，skipentropyrecord")
            mean_entropy = torch.tensor(0.0, device=self.device)
        
        # Completion lengthstatisticscounted（ref TRL）
        completion_lengths = final_mask.sum(dim=1) # [B*G] - every completion valid length
        mean_completion_length = completion_lengths.float().mean()
        min_completion_length = completion_lengths.float().min()
        max_completion_length = completion_lengths.float().max()
      
      # 6. Record metrics via Lightning logger
      # Align TRL: "sample" granularity is completion (B*G rows after expansion)
      self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=actual_num_samples)
      self.log("clip_ratio", clip_ratio, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("clip_ratio/region_mean", region_clip_ratio, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("clip_ratio/low_mean", low_clip_ratio, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("clip_ratio/high_mean", high_clip_ratio, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("kl_div", kl_div, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("entropy", mean_entropy, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("mean_advantage", mean_advantage, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("mean_reward", mean_reward, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("reward_std", reward_std, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("completions/mean_length", mean_completion_length, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("completions/min_length", min_completion_length, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("completions/max_length", max_completion_length, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True, batch_size=actual_num_samples)
      self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)
      
      # 8. Aggregate metrics (for potential external loggers); actual logging
      # is handled by Lightning's logger integration (e.g., SwanLabLogger).
      if self.global_rank == 0:
        metrics = {
          'train/loss': loss.item(),
          'train/clip_ratio': clip_ratio.item(),
          'train/clip_ratio/region_mean': region_clip_ratio.item(),
          'train/clip_ratio/low_mean': low_clip_ratio.item(),
          'train/clip_ratio/high_mean': high_clip_ratio.item(),
          'train/kl_div': kl_div.item(),
          'train/entropy': mean_entropy.item(),
          'train/mean_advantage': mean_advantage.item(),
          'train/mean_reward': mean_reward.item(),
          'train/reward_std': reward_std.item(),
          'train/completions/mean_length': mean_completion_length.item(),
          'train/completions/min_length': min_completion_length.item(),
          'train/completions/max_length': max_completion_length.item(),
          'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
        }

        for key, values in self._metrics.items():
          if len(values) > 0:
            metrics[f'train/{key}'] = sum(values) / len(values)

        self._log_metrics(metrics, self.training_step_count)
        self._metrics.clear()
      
      self.training_step_count += 1
      
      # [INFO] add new：record training_step endtimevisiblestore/savestate
      end_mem = get_gpu_memory_info(device) if torch.cuda.is_available() else None
      if start_mem and end_mem:
        mem_delta = end_mem['allocated'] - start_mem['allocated']
        print(f"[Rank {self.trainer.global_rank}] [OK] training_step complete: loss={loss.item():.4f}")
        print(f"[Rank {self.trainer.global_rank}]  [MEM] visiblestore/savechangeization: {mem_delta:+.2f} GB (end: {end_mem['allocated']:.2f} GB, userate: {end_mem['usage_percent']:.1f}%)")
      else:
        print(f"[Rank {self.trainer.global_rank}] [OK] training_step complete: loss={loss.item():.4f}\n")
      
      # [INFO] every training_step completeaftercleanvisiblestore/save（preventaccumulate）
      # [INFO] performanceization：visiblestore/savesufficienttimenot neededfrequentclean（fromevery8stepsmodifyforevery32steps）
      # current memory utilizationonlyhave 20-30%，frequentcleanwillaffect performance
      if batch_idx % 32 == 31: # every 32 stepscleanonce（reducefrequency）
        if self.trainer.global_rank == 0:
          print(f"[Rank {self.trainer.global_rank}] [CLEAN] periodiccleanvisiblestore/savebuffer...")
        gc.collect()
        torch.cuda.empty_cache()
      
      return loss
  
  def _compute_per_token_logps(
    self,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor
  ) -> torch.Tensor:
    """
    countedcomputeevery token tonumberprobability（usestrategymodel）
    
    Args:
      prompt_ids: [B*G, P] - provideshow IDs
      completion_ids: [B*G, C] - complete IDs
    
    Returns:
      logps: [B*G, C] - every completion token log probability
    """
    return self._compute_per_token_logps_with_model(
      self.model, prompt_ids, completion_ids
    )
  
  def _compute_per_token_logps_with_model(
    self,
    model: nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    batch_size: Optional[int] = None
  ) -> torch.Tensor:
    """
    usespecifymodelcountedcomputeevery token tonumberprobability（insidestore/saveizationversionthis）
    
    izationstrategy（ref trl-main）：
    1. chunkcountedcompute：by batch_size chunkforward，reducepeakvaluevisiblestore/save
    2. onlykeepnecessary logits：not keeping full sequence logits，onlykeep completion part
    3. use selective_log_softmax：avoidcomplete log_softmax operatework，directly gather need logp
    
    flow:
    1. concat [prompt; completion[:-1]]
    2. chunkforwardpropagation（e.g.result batch_size specify）
    3. onlyprovideget completion part logits
    4. use selective_log_softmax countedcompute logp（insidestore/savehigheffect）
    
    Args:
      model: RWKV model
      prompt_ids: [B*G, P] - provideshow IDs
      completion_ids: [B*G, C] - complete IDs
      batch_size: chunklargesmall，None tableshowno chunking（defaultuseconfiginvalue）
    
    Returns:
      logps: [B*G, C] - every completion token log probability
    
    Memory Complexity:
      - oldimplementation: O(B*G * (P+C) * vocab_size) for logits + log_probs
      - newimplementation: O(batch_size * C * vocab_size) for chunked logits
      - typical saving: 60GB+ for B*G=256, P+C~4k, vocab=65536
    
    References:
      - TRL GRPO: reference_project/trl-main/trl/trainer/grpo_trainer.py#L814
    """
    # getgetchunklargesmall（defaultfromconfiginreadget，e.g.resultnotspecifythenno chunking）
    if batch_size is None:
      batch_size = getattr(self.config, 'logprob_batch_size', None)
    
    # [INFO] key fix：getgetmodelplaceindevice（support DeepSpeed wrap）
    if hasattr(model, 'module'):
      model_device = next(model.module.parameters()).device
    else:
      model_device = next(model.parameters()).device
    
    # [INFO] ensure all inputs tensor allincorrectdeviceon
    prompt_ids = prompt_ids.to(model_device)
    completion_ids = completion_ids.to(model_device)
    
    # check batch size whethermatch
    if prompt_ids.shape[0] != completion_ids.shape[0]:
      raise ValueError(
        f"[ERROR] Batch size mismatch！\n"
        f" prompt_ids.shape: {prompt_ids.shape}\n"
        f" completion_ids.shape: {completion_ids.shape}\n"
        f" expected prompt_ids alreadywasExpand to [B*G, P]"
      )
    
    B_G, P = prompt_ids.shape
    _, C = completion_ids.shape
    
    # getgetmodel ctx_len（mostlargesequencelength）
    ctx_len = model.config.ctx_len
    
    # concat [prompt; completion[:-1]]
    completion_input = completion_ids[:, :-1] # [B*G, C-1]
    
    # checktotallengthwhetherexceed ctx_len，e.g.resultexceedthentruncate prompt
    max_prompt_len = ctx_len - completion_input.shape[1]
    if P > max_prompt_len:
      prompt_ids = prompt_ids[:, :max_prompt_len]
      P = max_prompt_len
    
    input_ids = torch.cat([prompt_ids, completion_input], dim=1) # [B*G, P+C-1]
    
    # RWKV7 trainingcomputesub-requirementsequencelengthis 16 multiple，needpad
    seq_len = input_ids.shape[1]
    CHUNK_LEN = 16
    pad_len = 0
    if seq_len % CHUNK_LEN != 0:
      pad_len = CHUNK_LEN - (seq_len % CHUNK_LEN)
      if seq_len + pad_len > ctx_len:
        max_seq_len = (ctx_len // CHUNK_LEN) * CHUNK_LEN
        input_ids = input_ids[:, :max_seq_len]
        actual_seq_len = input_ids.shape[1]
        actual_completion_input_len = actual_seq_len - P
        if actual_completion_input_len < 0:
          P = actual_seq_len
          actual_completion_input_len = 0
        pad_len = 0
      else:
        padding = torch.zeros(B_G, pad_len, dtype=input_ids.dtype, device=model_device)
        input_ids = torch.cat([input_ids, padding], dim=1)
    
    # ========== coreization：chunkcountedcompute + onlykeepnecessary logits ==========
    
    # logits_to_keep: weonlyneed completion part logits
    logits_to_keep = C + pad_len # onlyneedpretest completion tokens + padding tokens
    
    # [DEBUG] debug：recordizationstrategy
    if self.trainer.global_rank == 0 and not hasattr(self, '_logprob_optimization_logged'):
      print(f"[Rank 0] [INFO] LogProb izationstrategy:")
      print(f" - batch_size: {batch_size} (None=no chunking)")
      print(f" - B*G: {B_G}")
      print(f" - logits_to_keep: {logits_to_keep} (onlykeep completion part)")
      print(f" - use selective_log_softmax: True")
      self._logprob_optimization_logged = True
    
    # [INFO] key fix：onlyhavewhen batch_size for None timeonlyno chunking
    # original condition `batch_size >= B_G` is wrong，willcausecausechunkizationinvalid
    if batch_size is None:
      # no chunking，directlyforward
      # [INFO] pass in logits_to_keep paramnumber，letmodelonlycountedcalculate last N token logits
      logits = model(input_ids, logits_to_keep=logits_to_keep) # [B*G, logits_to_keep, vocab_size]
      if pad_len > 0:
        logits = logits[:, :C]
      
      # use selective_log_softmax avoidcomplete log_softmax
      # thisismostlargeMemory savingfromsource：notmaterializingcomplete [B*G, C, vocab_size] log_probs
      per_token_logps = selective_log_softmax(logits, completion_ids) # [B*G, C]
    else:
      # chunkcountedcompute，enteronestepsreducepeakvaluevisiblestore/save
      # [WARN] presentin batch_size=16 and B_G=128，placeinwillenterenterthisbranch
      all_logps = []
      for start in range(0, B_G, batch_size):
        end = min(start + batch_size, B_G)
        input_ids_batch = input_ids[start:end]
        completion_ids_batch = completion_ids[start:end]
        
        # forwardpropagation，pass in logits_to_keep
        logits_batch = model(input_ids_batch, logits_to_keep=logits_to_keep) # [batch_size, logits_to_keep, vocab_size]
        if pad_len > 0:
          logits_batch = logits_batch[:, :C]
        
        # use selective_log_softmax
        batch_logps = selective_log_softmax(logits_batch, completion_ids_batch)
        
        all_logps.append(batch_logps)
        
        # release memory promptly
        del logits_batch
        del batch_logps
        # everyprocessfew batch aftercleanoncebuffer
        if (end - start) % (batch_size * 4) == 0:
          torch.cuda.empty_cache()
      
      per_token_logps = torch.cat(all_logps, dim=0) # [B*G, C]
      # clean up at the endonce
      del all_logps
      torch.cuda.empty_cache()
    
    return per_token_logps
  
  def configure_optimizers(self):
    """configizer - complex/repeatuse RWKV7LightningModule logic"""
    config = self.config
    
    # paramnumbergrouplogic（withstandardtrainingkeeponecause）
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    
    for n, p in self.model.named_parameters():
      if not p.requires_grad:
        continue
      
      # att.w0 use2xlearning rate
      if "att.w0" in n:
        lr_2x.add(n)
      # >= 2dimensionandhave.weightparamnumberneedweight decay
      elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0) and (".weight" in n):
        lr_decay.add(n)
      # other parameters
      else:
        lr_1x.add(n)
    
    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    
    # [INFO] Bugfix：checkwhetherhavecantrainingparamnumber
    total_trainable = len(lr_1x) + len(lr_2x) + len(lr_decay)
    if total_trainable == 0:
      error_msg = (
        "[ERROR] ERROR: critical error：nohavefindtoanycantrainingparamnumber！\n"
        "  thisusuallymeans：\n"
        "  1. LoRAnotcorrectshoulduse（placehaveparamnumberwasfrozen）\n"
        "  2. requires_gradsethaveproblem\n"
        "  3. modelparamnumberconfigerror\n"
        "  please checkLoRAconfigandmodelinitialize。"
      )
      print(f"[OPTIMIZER_DIAG] {error_msg}")
      raise RuntimeError(error_msg)
    
    # [INFO] debuginfo：printparamnumberstatisticscounted（onlyinrank 0）
    if self.trainer is None or self.trainer.is_global_zero:
      print(f"[OPTIMIZER_DIAG] izerparamnumberstatisticscounted:")
      print(f"[OPTIMIZER_DIAG]  - lr_1x (1xlearning rate): {len(lr_1x)} paramnumber")
      print(f"[OPTIMIZER_DIAG]  - lr_2x (2xlearning rate): {len(lr_2x)} paramnumber")
      print(f"[OPTIMIZER_DIAG]  - lr_decay (weight_decay): {len(lr_decay)} paramnumber")
      print(f"[OPTIMIZER_DIAG]  - totalcounted: {total_trainable} cantrainingparamnumber")
      
      # checkLoRAparamnumber
      lora_params = [n for n in (lr_1x + lr_2x + lr_decay) if 'lora' in n.lower()]
      if self._use_peft:
        if len(lora_params) == 0:
          print(f"[OPTIMIZER_DIAG]  [WARN] [WARN] useLoRAbutnohavefindtoLoRAparamnumber！")
        else:
          print(f"[OPTIMIZER_DIAG]  [OK] LoRAparamnumber: {len(lora_params)} ")
          if len(lora_params) <= 10:
            print(f"[OPTIMIZER_DIAG]   LoRAparamnumberlist: {lora_params}")
    
    # buildparamnumberdict
    param_dict = {n: p for n, p in self.model.named_parameters()}
    
    # buildizerparamnumbergroup
    optim_groups = []
    
    # onlyaddnonemptyparamnumbergroup
    if len(lr_1x) > 0:
      optim_groups.append({"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0})
    if len(lr_2x) > 0:
      optim_groups.append({"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0})
    
    # weight_decayparamnumbersinglealoneonegroup
    if config.weight_decay > 0 and len(lr_decay) > 0:
      optim_groups.append({"params": [param_dict[n] for n in lr_decay], "weight_decay": config.weight_decay, "my_lr_scale": 1.0})
    
    # [INFO] Bugfix：againtimescheckizerparamnumbergroupwhetherforempty
    if len(optim_groups) == 0:
      error_msg = (
        "[ERROR] ERROR: critical error：izerparamnumbergroupforempty！\n"
        "  thiswillcausecausetrainingtimeparamnumbernotwillwasupdate。"
      )
      print(f"[OPTIMIZER_DIAG] {error_msg}")
      raise RuntimeError(error_msg)
    
    # selectizer
    # checkwhetheruse CPU training
    is_cpu_training = config.accelerator == "cpu" or not torch.cuda.is_available()
    
    if DEEPSPEED_AVAILABLE and self.deepspeed_offload:
      # DeepSpeed CPU offload
      optimizer = DeepSpeedCPUAdam(
        optim_groups,
        lr=config.lr_init,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        bias_correction=True,
        amsgrad=False
      )
    elif DEEPSPEED_AVAILABLE and not is_cpu_training:
      # FusedAdam onlycanin GPU onuse
      optimizer = FusedAdam(
        optim_groups,
        lr=config.lr_init,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        bias_correction=True,
        amsgrad=False
      )
    else:
      # CPU trainingornohave DeepSpeed timeusestandard AdamW
      optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.lr_init,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        weight_decay=0.0, # alreadyinparamnumbergroupset in
      )
    
    # learning ratecalldegree（cosine annealing）
    # e.g.result epoch_steps for None，usea larger default value（GRPO usuallyneedspecify epoch_steps）
    epoch_steps = config.epoch_steps if config.epoch_steps is not None else 1000
    # [INFO] key fix：presentin warmup_steps withprogress unit（training_step）keeponecause
    # 
    # Lightning scheduler rowsfor（ref pytorch-lightning/src/lightning/pytorch/loops/training_epoch_loop.py:449-453）：
    #  - e.g.result interval="step"，scheduler.step() will inevery training_step aftercalluse
    #  - butis，e.g.resultalsoingradientaccumulatephase（_should_accumulate() returnback True），scheduler.step() will not call
    #  - placeinactualon，scheduler.step() onlyin optimizer.step() timecalluse
    #  - therefore，scheduler.step() callusetimesnumber = optimizer.step() timesnumber = training_step / accumulate_grad_batches
    #
    # to make warmup_steps withprogress unitkeeponecause，wein lr_lambda converted in：
    #  - total_steps: optimizer.step() totaltimesnumber (for scheduler）
    #  - warmup_steps: training_step stepsnumber（useuserconfig，withenterdegreeonecause）
    total_steps = (config.epoch_count * epoch_steps) // config.accumulate_grad_batches
    
    def lr_lambda(current_step):
      # [INFO] key fix：warmup_steps presentinisbyenterdegreestepsnumber（training_step）countedcompute
      # Lightning scheduler.step() onlyin optimizer.step() timecalluse
      # placeinneedput current_step（scheduler callusetimesnumber）convertback training_step number
      actual_training_steps = current_step * config.accumulate_grad_batches
      
      # Warmup（by training_step countedcompute）
      if actual_training_steps < config.warmup_steps:
        return float(actual_training_steps) / float(max(1, config.warmup_steps))
      
      # Cosine decay
      # [INFO] protect：e.g.result warmup_steps >= total_steps，roundtrainingallin warmup phase
      # at this pointdirectlyreturnback lr_init（warmup completeaftervalue）
      total_training_steps = total_steps * config.accumulate_grad_batches
      if config.warmup_steps >= total_training_steps:
        return 1.0 # lr = lr_init * 1.0 = lr_init
      
      progress = float(actual_training_steps - config.warmup_steps) / float(max(1, total_training_steps - config.warmup_steps))
      progress = min(progress, 1.0) # prevent progress > 1.0
      cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
      lr_scale = config.lr_final / config.lr_init + (1.0 - config.lr_final / config.lr_init) * cosine_decay
      return lr_scale
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
      },
    }
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    """
    customizerstepsstep - supportlayered learning rate
    """
    # shoulduselayered learning rate
    for param_group in optimizer.param_groups:
      if 'my_lr_scale' in param_group:
        lr_scale = param_group['my_lr_scale']
        param_group['lr'] = optimizer.param_groups[0]['lr'] * lr_scale
    
    # executerowsizationstepsstep
    super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    # [INFO] LoRA parameter update diagnostic：confirm optimizer.step after lora_B whetherfrom 0 start changing
    if (
      getattr(self, '_use_peft', False)
      and self.trainer is not None
      and getattr(self.trainer, 'is_global_zero', False)
    ):
      counter = int(getattr(self, '_loss_log_counter', 0))
      freq = 50
      if counter == 1 or (counter % freq) == 0:
        sample = self._get_lora_diag_sample()
        if sample is not None:
          try:
            a_max = float(sample.lora_A.detach().abs().max().item())
            b_max = float(sample.lora_B.detach().abs().max().item())
            print(
              f"[Rank {self.trainer.global_rank}] [LORA_PARAM_DIAG_POST_OPT] after_opt(step#{counter}) "
              f"lora_A_absmax={a_max:.6e}, lora_B_absmax={b_max:.6e}"
            )
          except Exception:
            pass
  
  def _log_metrics(self, metrics: Dict[str, float], step: int):
    """
    Log aggregated GRPO metrics (per-reward-func stats, etc.) via
    Lightning's self.log() so they are picked up by the configured logger
    (SwanLab / WandB).  Core training metrics (loss, clip_ratio, etc.) are
    already logged directly in training_step; this method handles the
    supplementary metrics accumulated in the ``_metrics`` buffer during
    the rollout phase.
    """
    if not metrics:
      return
    for key, value in metrics.items():
      if isinstance(value, (int, float)):
        self.log(key, value, prog_bar=False, on_step=True, on_epoch=False,
                 rank_zero_only=True)
  
  def _save_rollout_data(self, rollout_data: Dict[str, Any], step: int):
    """
    save rollout datatofile
    
    Args:
      rollout_data: containsinfollowing fields：
        - prompts: provideshowlist
        - completions: generatebackanswerlist
        - completion_ids: completion token IDs（canselect，forgenerate masked_completion）
        - rewards: rewardvalue
        - rewards_per_func: every reward func individual values
        - advantages: advantagevalue
        - ground_truth: trueactual completion
        - original_data: originaldata
      step: whenbeforetrainingstepsnumber
    """
    if self.config.save_rollout_steps <= 0 or self.global_rank != 0:
      return
    
    try:
      # savefor jsonl format
      save_path = Path(self.config.save_rollout_path) / f"rollout_step_{step}.jsonl"
      
      # [INFO] e.g.resultuse loss_mask_func，provideget label_completion（use mask + completion_ids decode，forverifytrainingtimeactualuse label）
      label_completions = None
      if self.loss_mask_func is not None:
        print(f"[Rank {self.global_rank}] [TARGET] provideget label completions（use mask + completion_ids decode，verifytrainingtimeactualuse label）...")
        try:
          # getgetnecessarydata
          completions = rollout_data['completions']
          prompts = rollout_data['prompts']
          completion_ids = rollout_data['completion_ids'] # [B*G, C]
          
          # ensure completion_ids is tensor（e.g.resultis numpy array，convertfor tensor）
          if not isinstance(completion_ids, torch.Tensor):
            completion_ids = torch.tensor(completion_ids, dtype=torch.long)
          
          # calluse loss_mask_func getget mask（andtrainingtimeusemutualsamelogic）
          custom_mask = self.loss_mask_func(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids,
            tokenizer=self.tokenizer,
          ) # [B*G, C]
          
          # ensure mask is tensor
          if not isinstance(custom_mask, torch.Tensor):
            custom_mask = torch.tensor(custom_mask, dtype=torch.float32)
          
          # use mask and completion_ids decodeget label_completion
          label_completions = []
          for i in range(len(completions)):
            comp_ids = completion_ids[i].tolist() # [C]
            mask_i = custom_mask[i] # [C]
            
            # findto mask=1 token position
            mask_1_positions = torch.nonzero(mask_i > 0.5, as_tuple=True)[0].tolist()
            
            if len(mask_1_positions) > 0:
              # provideget mask=1 token IDs（exclude padding token=0）
              tokens_masked = [comp_ids[j] for j in mask_1_positions if comp_ids[j] != 0]
              
              if len(tokens_masked) > 0:
                try:
                  # Decode mask=1 part
                  label_text = self.tokenizer.decode(tokens_masked)
                  label_text = label_text.strip()
                  if label_text:
                    label_completions.append(label_text)
                  else:
                    label_completions.append("[EMPTY_AFTER_DECODE]")
                except Exception as e:
                  label_completions.append(f"[DECODE_ERROR: {str(e)}]")
              else:
                label_completions.append("[NO_VALID_TOKENS]")
            else:
              # mask allfor 0（should nothappen，butprocessoneunder）
              label_completions.append("[MASK_ALL_ZERO]")
          
          # statisticscounted
          valid_count = sum(1 for lc in label_completions if not lc.startswith('['))
          invalid_count = len(label_completions) - valid_count
          print(f"[Rank {self.global_rank}]  [OK] Label completions providegetcomplete")
          print(f"[Rank {self.global_rank}]  - haveeffectsamples: {valid_count}/{len(label_completions)} ({valid_count/len(label_completions)*100:.1f}%)")
          if invalid_count > 0:
            print(f"[Rank {self.global_rank}]  [WARN] noeffectsamples: {invalid_count}")
        except Exception as e:
          print(f"[Rank {self.global_rank}]  [WARN] provideget label completions failed: {e}")
          import traceback
          traceback.print_exc()
          label_completions = None
      
      # [INFO] addsavebeforestatisticscountedinfo
      if len(rollout_data['completions']) > 0:
        completion_lengths = [len(c) for c in rollout_data['completions']]
        completion_token_lengths = [len(self.tokenizer.encode(c)) for c in rollout_data['completions']]
        print(f"[Rank {self.global_rank}] [LOG] saveRolloutdatabeforestatisticscounted:")
        print(f"[Rank {self.global_rank}]  - Completioncount: {len(rollout_data['completions'])}")
        print(f"[Rank {self.global_rank}]  - charlength: min={min(completion_lengths)}, max={max(completion_lengths)}, avg={sum(completion_lengths)/len(completion_lengths):.1f}")
        print(f"[Rank {self.global_rank}]  - Tokenlength: min={min(completion_token_lengths)}, max={max(completion_token_lengths)}, avg={sum(completion_token_lengths)/len(completion_token_lengths):.1f}")
      
      saved_count = 0
      with open(save_path, 'w', encoding='utf-8') as f:
        # Get completion_ids for debugging decode issues
        completion_ids = rollout_data.get('completion_ids', None)
        
        for i in range(len(rollout_data['prompts'])):
          completion = rollout_data['completions'][i]
          completion_len = len(completion)
          completion_tokens = len(self.tokenizer.encode(completion))
          
          item = {
            'step': step,
            'prompt': rollout_data['prompts'][i],
            'completion': completion,  # ensure full text is saved
            'reward': float(rollout_data['rewards'][i]),
            'advantage': float(rollout_data['advantages'][i]),
          }
          
          # Add completion_ids for debugging decode issues (e.g., invalid tokens causing "�")
          if completion_ids is not None:
            try:
              if isinstance(completion_ids, torch.Tensor):
                comp_ids = completion_ids[i].cpu().tolist()  # [C]
              elif isinstance(completion_ids, np.ndarray):
                comp_ids = completion_ids[i].tolist()  # [C]
              elif isinstance(completion_ids, (list, tuple)):
                comp_ids = list(completion_ids[i])  # [C]
              else:
                comp_ids = None
              
              if comp_ids is not None:
                # Remove padding (0) tokens for cleaner output
                comp_ids_clean = [tid for tid in comp_ids if tid != 0]
                item['completion_ids'] = comp_ids_clean
                item['completion_token_count'] = len(comp_ids_clean)
                
                # Check for decode errors (contains replacement char)
                if '\ufffd' in completion or (completion == '\ufffd'):
                  item['decode_error'] = True
                  item['decode_error_tokens'] = comp_ids_clean
                  if saved_count < 5:  # Warn first few
                    print(f"[Rank {self.global_rank}] [WARN] Decode error detected for sample {i}: "
                          f"completion='{completion[:50]}...', tokens={comp_ids_clean[:20]}...")
            except Exception as e:
              item['completion_ids_error'] = str(e)
          
          # add label_completion（e.g.resulthave）- thisis </think> part to be learned after
          if label_completions is not None:
            item['label_completion'] = label_completions[i]
          
          # addevery reward func individual values（ref trl-main）
          if 'rewards_per_func' in rollout_data:
            rewards_dict = {}
            for j, func_name in enumerate(self.reward_func_names):
              rewards_dict[func_name] = float(rollout_data['rewards_per_func'][i, j])
            item['rewards_per_func'] = rewards_dict
          
          # add ground_truth_scores（trueactual score）
          if 'ground_truth_scores' in rollout_data and rollout_data['ground_truth_scores'] is not None:
            if i < len(rollout_data['ground_truth_scores']):
              item['ground_truth_scores'] = float(rollout_data['ground_truth_scores'][i])
          
          # add ground_truth（trueactual completion）
          if 'ground_truth' in rollout_data and rollout_data['ground_truth'] is not None:
            if i < len(rollout_data['ground_truth']):
              item['ground_truth'] = rollout_data['ground_truth'][i]
          
          # [INFO] add ground_truth_answer（GSM8K etctaskstandardanswersolution）
          if 'ground_truth_answer' in rollout_data and rollout_data['ground_truth_answer'] is not None:
            if i < len(rollout_data['ground_truth_answer']):
              item['ground_truth_answer'] = rollout_data['ground_truth_answer'][i]
          
          # addextrafield（e.g.resulthave）
          if 'original_data' in rollout_data and rollout_data['original_data'] is not None:
            if i < len(rollout_data['original_data']):
              item['original_data'] = rollout_data['original_data'][i]
          
          # [INFO] verifysavebeforecompletionlength
          json_str = json.dumps(item, ensure_ascii=False)
          f.write(json_str + '\n')
          saved_count += 1
          
          # [INFO] verifysaveafterdata（onlybefore3）
          if saved_count <= 3:
            loaded_item = json.loads(json_str)
            loaded_completion = loaded_item['completion']
            if len(loaded_completion) != completion_len:
              print(f"[Rank {self.global_rank}] [WARN] warn: saveaftercompletionlengthmismatch! original={completion_len}, loadafter={len(loaded_completion)}")
      
      # [INFO] addsaveafterverify
      print(f"[Rank {self.global_rank}] [OK] [Step {step}] Rollout dataalreadysaveto: {save_path}")
      print(f"[Rank {self.global_rank}]  - save {saved_count} record")
      
      # [INFO] readgetandverifysavefile（onlyverify before3）
      try:
        with open(save_path, 'r', encoding='utf-8') as f:
          for i, line in enumerate(f):
            if i >= 3:
              break
            loaded_item = json.loads(line)
            loaded_comp_len = len(loaded_item['completion'])
            loaded_comp_tokens = len(self.tokenizer.encode(loaded_item['completion']))
            print(f"[Rank {self.global_rank}]  - verifyrecord{i+1}: completionlength={loaded_comp_len} chars, {loaded_comp_tokens} tokens")
      except Exception as e:
        print(f"[Rank {self.global_rank}] [WARN] verifysavefiletimeerror: {e}")
      
    except Exception as e:
      print(f"[WARN] save rollout datafailed: {e}")

