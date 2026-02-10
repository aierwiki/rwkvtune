"""
GRPO Trainer - Integrates all GRPO training components.

This module implements Group Relative Policy Optimization (GRPO) training for RWKV models,
following the trl-main design pattern for model, tokenizer and reward functions.
"""
import os
import time
import json
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from pathlib import Path
from typing import Optional, Union, List, Callable, Any
from torch.utils.data import Dataset

from rwkvtune.training.grpo_config import GRPOConfig
from rwkvtune.training.grpo.lightning_module import GRPOLightningModule
from rwkvtune.training.grpo.dataset import create_grpo_dataloader, GRPODataBuffer
from rwkvtune.training.grpo.advantage import AdvantageCalculator
from rwkvtune.data.tokenizers import get_tokenizer

# HuggingFace Dataset support
try:
    from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset
    _has_datasets = True
except ImportError:
    _has_datasets = False
    HFDataset = None
    HFIterableDataset = None


class GRPOTrainingCallback(Callback):
    """GRPO training callback - handles generation, reward calculation and logging."""

    def __init__(self, config: GRPOConfig, grpo_trainer: 'GRPOTrainer'):
        super().__init__()
        self.config = config
        self.grpo_trainer = grpo_trainer
        self.loss_file = os.path.join(config.proj_dir, "grpo_loss_data.jsonl")

        # Clear old loss file
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)

        # Checkpoint rotation queue
        self.saved_checkpoints_queue = []

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        if trainer.is_global_zero:
            print(f"\n{'='*80}")
            print(f"GRPO training started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Config: {vars(self.config)}")
            print(f"{'='*80}\n")

        # Initialize accumulation loss
        trainer.grpo_accumulation_loss_sum = 0.0

        # Create log file
        trainer.my_log = open(
            os.path.join(self.config.proj_dir, "grpo_train_log.txt"), "a"
        )
        trainer.my_log.write(f"\n{'='*80}\n")
        trainer.my_log.write(f"GRPO training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        trainer.my_log.write(f"Config: {vars(self.config)}\n")
        trainer.my_log.write(f"{'='*80}\n")
        trainer.my_log.flush()

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Compatible with custom batch_sampler epoch shuffle.
        When using custom batch_sampler (repeat batch) and disabling Lightning's
        use_distributed_sampler, we need to manually call set_epoch on DistributedSampler.
        """
        train_dl = getattr(trainer, "train_dataloader", None)
        if train_dl is None:
            return

        batch_sampler = getattr(train_dl, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(trainer.current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when a training batch ends."""
        # Get loss and sync across GPUs
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = outputs

        # Multi-GPU sync (all_reduce is a collective operation)
        if trainer.world_size > 1:
            import torch.distributed as dist
            loss_for_sync = loss.clone().detach()
            dist.all_reduce(loss_for_sync, op=dist.ReduceOp.SUM)
            loss = loss_for_sync / trainer.world_size

        loss = loss.item()

        # Only record log on rank 0
        if trainer.is_global_zero:
            config = self.config
            if trainer.accumulate_grad_batches is not None and trainer.accumulate_grad_batches > 1:
                if not hasattr(trainer, 'grpo_accumulation_loss_sum'):
                    trainer.grpo_accumulation_loss_sum = 0.0

                trainer.grpo_accumulation_loss_sum += loss

                if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
                    avg_loss = trainer.grpo_accumulation_loss_sum
                    with open(self.loss_file, 'a') as f:
                        json.dump({
                            "step": trainer.global_step,
                            "loss": float(avg_loss),
                            "timestamp": time.time()
                        }, f)
                        f.write('\n')
                    trainer.grpo_accumulation_loss_sum = 0.0
            else:
                with open(self.loss_file, 'a') as f:
                    json.dump({
                        "step": trainer.global_step,
                        "loss": float(loss),
                        "timestamp": time.time()
                    }, f)
                    f.write('\n')

            # Save model by batch number
            epoch_steps = config.epoch_steps if config.epoch_steps is not None else trainer.num_training_batches
            real_batch = batch_idx + 1 + trainer.current_epoch * epoch_steps
            if config.save_every_n_batches > 0:
                if real_batch > 0 and real_batch % config.save_every_n_batches == 0:
                    self._save_checkpoint(trainer, pl_module, batch=real_batch)

    def _save_checkpoint(self, trainer, pl_module, epoch=None, batch=None):
        """Save checkpoint (supports LoRA mode)."""
        # Skip saving during dummy forward phase
        if hasattr(pl_module, 'skip_until_step') and trainer.global_step < pl_module.skip_until_step:
            return

        config = self.config

        # Generate checkpoint name
        if batch is not None:
            checkpoint_name = f"rwkv7-batch{batch}"
        elif epoch is not None:
            checkpoint_name = f"rwkv7-epoch{epoch + 1}"
        else:
            checkpoint_name = "rwkv7-final"

        print(f"\nSaving checkpoint: {checkpoint_name}")

        use_lora = getattr(config, 'use_lora', False)
        lora_save_mode = getattr(config, 'lora_save_mode', 'lora_only')
        lora_config = getattr(self.grpo_trainer, 'lora_config', None)

        if use_lora:
            print(f"  LoRA mode: {lora_save_mode}")

        from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support
        try:
            save_checkpoint_with_lora_support(
                lightning_module=pl_module,
                save_directory=config.proj_dir,
                checkpoint_name=checkpoint_name,
                tokenizer=getattr(self.grpo_trainer, 'tokenizer', None),
                model_path=getattr(self.grpo_trainer, 'model_path', None),
                use_lora=use_lora,
                lora_save_mode=lora_save_mode,
                lora_config=lora_config,
            )
        except Exception as e:
            print(f"Warning: Save failed: {e}")
            print(f"  Falling back to weights-only save...")
            save_path = os.path.join(config.proj_dir, f"{checkpoint_name}.pth")
            torch.save(pl_module.model.state_dict(), save_path)
            print(f"Weights saved to: {save_path}")
        else:
            # Save PyTorch Lightning complete checkpoint (for resume)
            try:
                ckpt_path = os.path.join(config.proj_dir, f"{checkpoint_name}.ckpt")
                tmp_path = f"{ckpt_path}.tmp"
                save_optimizer_state = getattr(config, 'save_optimizer_state', True)
                print(f"Saving training state checkpoint: {ckpt_path}")
                print(f"  Save optimizer state: {save_optimizer_state}")

                trainer.save_checkpoint(tmp_path, weights_only=not save_optimizer_state)

                # Atomic rename
                if os.path.exists(ckpt_path):
                    if os.path.isdir(ckpt_path):
                        import shutil
                        shutil.rmtree(ckpt_path)
                    else:
                        os.remove(ckpt_path)
                os.rename(tmp_path, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

                self._rotate_checkpoints(ckpt_path)
            except Exception as e:
                print(f"Warning: Failed to save training state: {e}")

    def _rotate_checkpoints(self, new_ckpt_path):
        """Manage checkpoint count, delete old checkpoints."""
        limit = getattr(self.config, 'save_total_limit', 0)
        if limit <= 0:
            return

        self.saved_checkpoints_queue.append(new_ckpt_path)

        while len(self.saved_checkpoints_queue) > limit:
            old_ckpt = self.saved_checkpoints_queue.pop(0)
            if os.path.exists(old_ckpt):
                try:
                    print(f"Deleting old checkpoint (limit={limit}): {old_ckpt}")
                    if os.path.isdir(old_ckpt):
                        import shutil
                        shutil.rmtree(old_ckpt)
                    else:
                        os.remove(old_ckpt)
                except Exception as e:
                    print(f"Warning: Failed to delete old checkpoint: {e}")
            else:
                print(f"Warning: Old checkpoint not found, skipping delete: {old_ckpt}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when a training epoch ends."""
        # Skip saving during dummy forward phase
        if hasattr(pl_module, 'skip_until_step') and trainer.global_step < pl_module.skip_until_step:
            return

        if trainer.is_global_zero:
            print(f"\nEpoch {trainer.current_epoch} complete")

            checkpoint_name = f"checkpoint_epoch_{trainer.current_epoch}"
            use_lora = getattr(self.config, 'use_lora', False)
            lora_save_mode = getattr(self.config, 'lora_save_mode', 'lora_only')
            lora_config = getattr(self.grpo_trainer, 'lora_config', None)

            if use_lora:
                from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support
                save_checkpoint_with_lora_support(
                    lightning_module=pl_module,
                    save_directory=self.config.proj_dir,
                    checkpoint_name=checkpoint_name,
                    tokenizer=getattr(self.grpo_trainer, 'tokenizer', None),
                    model_path=getattr(self.grpo_trainer, 'model_path', None),
                    use_lora=use_lora,
                    lora_save_mode=lora_save_mode,
                    lora_config=lora_config,
                )
            else:
                save_path = os.path.join(
                    self.config.proj_dir, f"{checkpoint_name}.pth"
                )
                torch.save(pl_module.model.state_dict(), save_path)
                print(f"Model saved to: {save_path}")


class GRPOTrainer:
    """
    GRPO Trainer - follows trl-main design style.

    Responsibilities:
    1. Initialize all components
    2. Manage training loop
    3. Coordinate generation, reward calculation, advantage computation and policy updates

    Example:
        ```python
        from rwkvtune.training import GRPOConfig, GRPOTrainer
        from rwkvtune import AutoModel, AutoTokenizer
        from datasets import Dataset

        # Load model and tokenizer
        model = AutoModel.from_pretrained("path/to/model")
        tokenizer = AutoTokenizer.from_pretrained()

        # Define reward function
        def my_reward_func(prompts, completions, **kwargs):
            return [float(len(c)) for c in completions]

        # Create dataset
        dataset = Dataset.from_list([
            {'prompt': 'Hello'},
            {'prompt': 'World'},
        ])

        # Create config
        config = GRPOConfig(
            output_dir="./output",
            num_train_epochs=3,
        )

        # Create Trainer
        trainer = GRPOTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=dataset,
            reward_funcs=my_reward_func,
        )

        # Start training
        trainer.train()
        ```
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        reward_funcs: Union[Callable, List[Callable]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional["LoraConfig"] = None,
        loss_mask_func: Optional[Callable] = None,
    ):
        """
        Args:
            model: Model path (string) or instantiated model object.
            reward_funcs: Reward function(s). Can be a single function or list of functions.
                Each function should accept (prompts, completions, **kwargs) and return List[float].
            args: GRPO config (optional, defaults to GRPOConfig()).
            train_dataset: HuggingFace Dataset (required, must contain 'prompt' and 'input_ids' fields).
            eval_dataset: Validation dataset (optional, not implemented yet).
            processing_class: Tokenizer (optional, auto-loads if None).
            callbacks: Callback list (optional, not implemented yet).
            optimizers: Optimizer and scheduler tuple (optional, not implemented yet).
            peft_config: LoRA config (optional). If provided, uses LoRA fine-tuning.
            loss_mask_func: Custom loss mask function (optional).
        """
        # Config
        if args is None:
            args = GRPOConfig()
        self.config = args

        # Create output directory
        os.makedirs(self.config.proj_dir, exist_ok=True)

        # Set random seed
        pl.seed_everything(self.config.seed)

        # Save PEFT config
        self.peft_config = peft_config
        self._use_peft = peft_config is not None

        # Setup model
        print("Initializing model...")
        self.model = self._setup_model(model)
        self.model_path = model if isinstance(model, str) else None

        # Apply PEFT if configured
        if self._use_peft:
            from rwkvtune.peft import get_peft_model
            print("Applying LoRA...")
            self.model = get_peft_model(self.model, peft_config)

            self.lora_config = {
                'r': peft_config.r,
                'lora_alpha': peft_config.lora_alpha,
                'lora_dropout': peft_config.lora_dropout,
                'target_modules': getattr(peft_config, 'target_modules', None),
                'modules_to_save': getattr(peft_config, 'modules_to_save', None),
                'use_rslora': getattr(peft_config, 'use_rslora', False),
                'base_model_path': self.model_path,
            }
        else:
            self.lora_config = None

        # Load model config for Lightning module
        self._load_model_config_for_lightning()

        # Setup tokenizer
        print("Initializing Tokenizer...")
        self.tokenizer = self._setup_tokenizer(processing_class)

        # Validate dataset
        if train_dataset is None:
            raise ValueError(
                "train_dataset is required!\n"
                "Please use HuggingFace Dataset, e.g.:\n"
                "  from datasets import Dataset\n"
                "  dataset = Dataset.from_list([{'prompt': '...', ...}, ...])\n"
                "  trainer = GRPOTrainer(model=model, train_dataset=dataset, reward_funcs=...)"
            )

        if not _has_datasets:
            raise ImportError("datasets library required! Run: pip install datasets")

        if not isinstance(train_dataset, (HFDataset, HFIterableDataset)):
            raise TypeError(
                f"train_dataset must be HuggingFace Dataset, got {type(train_dataset)}"
            )

        if 'prompt' not in train_dataset.column_names:
            raise ValueError(
                f"train_dataset must contain 'prompt' field!\n"
                f"Current fields: {train_dataset.column_names}"
            )

        if 'input_ids' not in train_dataset.column_names:
            raise ValueError(
                f"train_dataset must contain 'input_ids' field!\n"
                f"Current fields: {train_dataset.column_names}\n"
                "GRPO training requires pre-tokenized data."
            )

        print(f"Using HuggingFace Dataset: {len(train_dataset)} samples")
        print(f"  Fields: {train_dataset.column_names}")
        self.dataset = train_dataset

        # Create Lightning module
        print("Initializing GRPO module...")
        self.lightning_module = GRPOLightningModule(
            self.config, use_peft=self._use_peft, loss_mask_func=loss_mask_func
        )

        # Ensure LightningModule uses the same tokenizer
        self.lightning_module.tokenizer = self.tokenizer

        # Load model weights to Lightning module
        if self._use_peft:
            self.lightning_module.model = self.model
            print("Using LoRA model (contains adapter layers)")
        else:
            self.lightning_module.model.load_state_dict(self.model.state_dict(), strict=False)

            # Load reference model weights (only needed when not using PEFT)
            if self.config.beta > 0.0 and self.lightning_module.ref_model is not None:
                if self.config.ref_model_path:
                    print(f"Loading reference model: {self.config.ref_model_path}")
                    from rwkvtune import AutoModel
                    ref_model = AutoModel.from_pretrained(self.config.ref_model_path)
                    self.lightning_module.ref_model.load_state_dict(ref_model.state_dict(), strict=False)
                else:
                    self.lightning_module.ref_model.load_state_dict(self.model.state_dict(), strict=False)

        # Set RWKV_DEVICE environment variable based on config
        if self.config.accelerator == "cpu":
            os.environ["RWKV_DEVICE"] = "cpu"
        elif self.config.accelerator == "gpu" and torch.cuda.is_available():
            os.environ["RWKV_DEVICE"] = "cuda"
        else:
            os.environ["RWKV_DEVICE"] = "cpu"

        # Check if using multi-GPU strategy
        is_multi_gpu = (
            self.config.accelerator == "gpu"
            and self.config.devices > 1
            and ("deepspeed" in str(self.config.strategy).lower()
                 or "ddp" in str(self.config.strategy).lower())
        )

        if is_multi_gpu:
            print(f"Model kept on CPU (multi-GPU mode, Lightning/DeepSpeed will distribute at setup)")
        else:
            device = "cuda" if self.config.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
            self.lightning_module.to(device)
            print(f"Model moved to device: {device}")

        # Set model precision
        if self.config.precision in ["bf16", "bf16-mixed"]:
            self.lightning_module.to(torch.bfloat16)
        elif self.config.precision in ["fp16", "16", "16-mixed"]:
            self.lightning_module.to(torch.float16)
        else:
            self.lightning_module.to(torch.float32)

        print(f"Model precision: {self.config.precision}")
        print(f"RWKV operator device: {os.environ.get('RWKV_DEVICE', 'auto')}")

        # Setup reward functions
        self._setup_reward_funcs(reward_funcs)

        # Pass reward functions to LightningModule
        self.lightning_module.reward_functions = self.reward_functions
        self.lightning_module.reward_weights = self.reward_weights
        self.lightning_module.reward_func_names = self.reward_func_names

        print("GRPO Trainer initialization complete\n")

    def _setup_model(self, model: Union[str, torch.nn.Module]) -> torch.nn.Module:
        """Setup model."""
        from rwkvtune import AutoModel

        if isinstance(model, str):
            print(f"Loading model from path: {model}")
            model_obj = AutoModel.from_pretrained(model, device="cpu")
            print(f"Model loaded (CPU)")
            return model_obj
        elif isinstance(model, torch.nn.Module):
            print(f"Using instantiated model: {type(model).__name__}")
            return model
        else:
            raise TypeError(f"model parameter type error: {type(model)}")

    def _setup_tokenizer(self, processing_class: Optional[Any]):
        """Setup Tokenizer."""
        if processing_class is not None:
            print(f"Using provided Tokenizer: {type(processing_class).__name__}")
            return processing_class
        else:
            print(f"Using default Tokenizer...")
            tokenizer = get_tokenizer()
            print(f"Tokenizer loaded")
            return tokenizer

    def _setup_reward_funcs(
        self, reward_funcs: Optional[Union[Callable, List[Callable]]]
    ):
        """Setup reward functions."""
        if reward_funcs is None:
            raise ValueError(
                "reward_funcs is required!\n"
                "Please define your reward function, e.g.:\n"
                "  def my_reward_func(prompts, completions, **kwargs):\n"
                "      return [float(len(c)) for c in completions]"
            )
        elif not isinstance(reward_funcs, list):
            self.reward_functions = [reward_funcs]
        else:
            self.reward_functions = reward_funcs

        # Validate all reward functions are callable
        for i, func in enumerate(self.reward_functions):
            if not callable(func):
                raise ValueError(f"reward_funcs[{i}] is not callable: {type(func)}")

        # Generate reward function names
        self.reward_func_names = []
        for i, func in enumerate(self.reward_functions):
            if hasattr(func, '__name__'):
                name = func.__name__
            else:
                name = f"reward_func_{i}"
            self.reward_func_names.append(name)

        print(f"Reward functions: {', '.join(self.reward_func_names)}")

        # Setup weights
        if hasattr(self.config, 'reward_weights') and self.config.reward_weights is not None:
            self.reward_weights = self.config.reward_weights
            if len(self.reward_weights) != len(self.reward_functions):
                raise ValueError(
                    f"Reward function count ({len(self.reward_functions)}) does not match "
                    f"weights count ({len(self.reward_weights)})"
                )
        else:
            self.reward_weights = [1.0] * len(self.reward_functions)

    def _load_model_config_for_lightning(self):
        """
        Load model config for Lightning module.
        Extracts config from the loaded model and syncs training parameters.
        """
        try:
            if hasattr(self.model, 'config'):
                model_config = self.model.config

                # 1. Read architecture parameters from model config
                self.config.n_layer = model_config.n_layer
                self.config.n_embd = model_config.n_embd
                self.config.vocab_size = model_config.vocab_size
                self.config.head_size_a = model_config.head_size_a

                # LORA dimensions
                self.config.dim_att_lora = getattr(model_config, 'dim_att_lora', 32)
                self.config.dim_gate_lora = getattr(model_config, 'dim_gate_lora', 128)
                self.config.dim_mv_lora = getattr(model_config, 'dim_mv_lora', 32)

                # Derived parameters
                self.config.dim_att = self.config.n_embd
                self.config.dim_ffn = int((self.config.n_embd * 3.5) // 32 * 32)
                self.config.head_size_divisor = 8
                self.config.my_testing = 'x070'

                # 2. Pass training parameters to model config
                model_config.grad_cp = self.config.grad_cp
                model_config.ctx_len = self.config.ctx_len

                print(f"Model config synced to GRPOConfig")
                print(f"  Architecture: {self.config.n_layer} layers, {self.config.n_embd} dim, {self.config.vocab_size} vocab")
                print(f"  Training params: grad_cp={self.config.grad_cp}, ctx_len={self.config.ctx_len}")
            else:
                raise ValueError("Model object has no config attribute")
        except Exception as e:
            print(f"Error: Cannot extract config from model")
            print(f"  {str(e)}")
            raise

    def train(self):
        """Start training using Lightning Trainer."""
        config = self.config
        steps_per_generation = int(getattr(config, "steps_per_generation", config.accumulate_grad_batches))
        generate_every = steps_per_generation * int(getattr(config, "num_iterations", 1))

        print(f"Starting GRPO training...")
        print(f"  - Dataset size: {len(self.dataset)}")
        print(f"  - Generations per prompt: {config.num_generations}")
        print(f"  - Batch size: {config.micro_bsz}")
        print(f"  - steps_per_generation: {steps_per_generation}")
        print(f"  - num_iterations: {getattr(config, 'num_iterations', 1)}")
        print(f"  - generate_every: {generate_every}")
        print(f"  - DataLoader repeat_batch_count: {generate_every}")
        print(f"  - Epochs: {config.epoch_count}")
        print(f"  - Devices: {config.devices} x {config.accelerator}")
        print()

        # Create dataloader with repeat batch semantics
        world_size_for_dataloader = getattr(config, 'devices', 1)
        rank_for_dataloader = 0

        dataloader = create_grpo_dataloader(
            dataset=self.dataset,
            batch_size=config.micro_bsz,
            num_workers=config.num_workers,
            shuffle=True,
            repeat_batch_count=generate_every,
            seed=int(getattr(config, "seed", 42)),
            world_size=world_size_for_dataloader,
            rank=rank_for_dataloader,
        )

        if config.epoch_steps is None:
            config.epoch_steps = len(dataloader)

        # Create Lightning Trainer
        trainer = self._create_trainer()

        # Handle checkpoint resume (Dummy Forward strategy)
        if config.resume_from_checkpoint:
            ckpt_path = config.resume_from_checkpoint

            # Auto-find latest checkpoint if directory
            if os.path.isdir(ckpt_path):
                ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith('.ckpt')]
                subdirs = [d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))]
                valid_subdirs = []
                for d in subdirs:
                    if os.path.exists(os.path.join(ckpt_path, d, 'trainer_state.ckpt')):
                        valid_subdirs.append(d)

                if valid_subdirs:
                    valid_subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)), reverse=True)
                    latest_subdir = valid_subdirs[0]
                    ckpt_path = os.path.join(ckpt_path, latest_subdir, 'trainer_state.ckpt')
                    print(f"Auto-selected latest checkpoint: {ckpt_path}")
                elif ckpt_files:
                    if 'last.ckpt' in ckpt_files:
                        ckpt_path = os.path.join(ckpt_path, 'last.ckpt')
                    else:
                        ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)), reverse=True)
                        ckpt_path = os.path.join(ckpt_path, ckpt_files[0])
                    print(f"Auto-selected checkpoint: {ckpt_path}")
                elif os.path.exists(os.path.join(ckpt_path, 'adapter_model.bin')):
                    ckpt_path = os.path.join(ckpt_path, 'adapter_model.bin')
                    print(f"Auto-selected LoRA weights: {ckpt_path}")
                else:
                    print(f"Assuming DeepSpeed checkpoint directory: {ckpt_path}")

            print(f"\nPreparing to resume from checkpoint: {ckpt_path}")

            try:
                # Load checkpoint metadata
                metadata_path = ckpt_path
                if os.path.isdir(ckpt_path):
                    candidates = [
                        os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt'),
                    ]
                    for candidate in candidates:
                        if os.path.isfile(candidate):
                            metadata_path = candidate
                            break
                    if metadata_path == ckpt_path:
                        for root, _, files in os.walk(ckpt_path):
                            for filename in files:
                                if filename.endswith('_model_states.pt'):
                                    metadata_path = os.path.join(root, filename)
                                    break
                            if metadata_path != ckpt_path:
                                break

                try:
                    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
                except Exception:
                    rank = 0
                if rank == 0:
                    print(f"  [DEBUG] Metadata load path: {metadata_path}")

                checkpoint = torch.load(metadata_path, map_location='cpu', weights_only=False)

                # Get processed sample count
                if 'samples_seen' in checkpoint:
                    samples_seen = checkpoint['samples_seen']
                    print(f"  Processed samples (samples_seen): {samples_seen}")
                else:
                    print("Warning: 'samples_seen' not found, estimating from global_step...")
                    global_step = checkpoint.get('global_step', 0)
                    samples_seen = global_step * config.micro_bsz * config.devices * config.accumulate_grad_batches
                    print(f"  Estimated samples: {samples_seen}")

                # Calculate steps to skip
                numerator = samples_seen * generate_every
                denominator = config.accumulate_grad_batches * config.micro_bsz * config.devices
                target_global_step = numerator // denominator
                print(f"  GRPO params: generate_every={generate_every}")
                print(f"  Target skip steps: {target_global_step}")

                # Set model state
                self.lightning_module.set_skip_steps(target_global_step)
                self.lightning_module.set_resume_checkpoint_path(ckpt_path)

                # Load model weights
                print("  Loading model weights...")
                if os.path.isdir(ckpt_path):
                    pass
                else:
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        print("Warning: 'state_dict' not found, assuming weights-only file")
                        state_dict = checkpoint
                    self.lightning_module.load_state_dict(state_dict, strict=False)
                    print("Model weights loaded")

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("  Starting training from scratch...")

        # Start training
        print(f"\n{'='*80}")
        print(f"Starting Lightning training...")
        print(f"{'='*80}\n")

        trainer.fit(self.lightning_module, dataloader)

        print(f"\nGRPO training complete!")
        print(f"  Model saved to: {config.proj_dir}")

    def save_model(self, output_dir: Optional[str] = None):
        """Save model (supports LoRA mode)."""
        from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support

        if output_dir is None:
            output_dir = self.config.proj_dir

        use_lora = getattr(self.config, 'use_lora', False) or self._use_peft
        lora_save_mode = getattr(self.config, 'lora_save_mode', 'lora_only')

        print(f"\nSaving final model to: {output_dir}")
        if use_lora:
            print(f"  LoRA mode: {lora_save_mode}")

        try:
            save_checkpoint_with_lora_support(
                lightning_module=self.lightning_module,
                save_directory=output_dir,
                checkpoint_name="final",
                tokenizer=getattr(self, 'tokenizer', None),
                model_path=getattr(self, 'model_path', None),
                use_lora=use_lora,
                lora_save_mode=lora_save_mode,
                lora_config=getattr(self, 'lora_config', None),
            )
        except Exception as e:
            print(f"Warning: Failed to save complete model: {e}")
            print(f"  Falling back to weights-only save...")
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "final_model.pth")
            torch.save(self.lightning_module.model.state_dict(), save_path)
            print(f"Weights saved to: {save_path}")

    def _create_trainer(self) -> Trainer:
        """Create Lightning Trainer."""
        config = self.config

        # Precision setup
        precision = config.precision
        if config.accelerator == 'cpu':
            if precision in ["bf16", "fp16"]:
                print(f"Warning: CPU training does not recommend {precision}, switching to fp32")
                precision = 32
            elif precision == "fp32":
                precision = 32
        else:
            if precision == "fp32":
                precision = 32
            elif precision == "fp16":
                precision = 16

        # Strategy setup
        strategy = config.strategy
        if config.accelerator == 'cpu':
            strategy = 'auto'
            devices = 1
        else:
            devices = config.devices

        # Create callback
        callbacks = [GRPOTrainingCallback(config, self)]

        # Create trainer parameters
        trainer_kwargs = dict(
            accelerator=config.accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=1,
            precision=precision,
            max_epochs=config.epoch_count,
            accumulate_grad_batches=config.accumulate_grad_batches,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
            log_every_n_steps=config.log_steps,
            # Important: Use custom sampler/batch_sampler to implement "repeat batch" semantics
            use_distributed_sampler=False,
        )

        # Add gradient clip support
        trainer_kwargs['gradient_clip_val'] = config.grad_clip

        if config.epoch_steps is not None:
            trainer_kwargs['max_steps'] = config.epoch_count * config.epoch_steps

        trainer = Trainer(**trainer_kwargs)
        return trainer
