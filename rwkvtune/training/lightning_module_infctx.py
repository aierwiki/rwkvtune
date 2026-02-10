"""
Lightning Training Module - Infinite Context Support

Supports two training modes (via truncated_bptt parameter):

1. Truncated mode (truncated_bptt=True, default):
      - Each chunk has separate backward + detach state
      - Memory = O(chunk_size), supports infinite sequence length
      - Gradients truncated at chunk boundaries (Truncated BPTT)

2. Full mode (truncated_bptt=False):
      - Accumulate all chunk losses, single backward at end
      - Memory = O(seq_len), no support for very long sequences
      - Gradients propagate to all chunks (Full BPTT)

Both modes force torch_checkpoint to save memory on layer activations.
"""
import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import lightning as pl
from lightning.pytorch.strategies import DeepSpeedStrategy

# DeepSpeed is optional (required for GPU training)
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    DeepSpeedCPUAdam = None
    FusedAdam = None
    print("DeepSpeed not available, using standard PyTorch optimizer")

from rwkvtune.models.rwkv7.model_infctx import RWKV7Infctx
from rwkvtune.models.rwkv7.infctx_module import BlockStateList

class L2Wrap(torch.autograd.Function):
    """
    L2 regularization wrapper (for infctx training)
    """

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]

        # Calculate L2 regularization gradient
        # Encourage logits near 0 to prevent overflow
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)

        return grad_output, gy

class RWKV7InfctxLightningModule(pl.LightningModule):
    """
    RWKV7 Infinite Context Lightning Module

    Supports two training modes (via config.truncated_bptt):

    1. Truncated mode (truncated_bptt=True, default):
          - Each chunk: separate backward + detach
          - Memory = O(chunk_size), handles infinite length

    2. Full mode (truncated_bptt=False):
          - Accumulate loss, single backward, full gradient propagation
          - Memory = O(seq_len), no very long sequence support

    Both modes force torch_checkpoint to save layer activations.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = RWKV7Infctx(config)
        # Loss function (ignore_index=-100 ignores padding/prompt)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Disable Lightning auto-optimization, manual backward control
        self.automatic_optimization = False

        # Training mode: Truncated BPTT (default) or Full BPTT
        self.truncated_bptt = getattr(config, 'truncated_bptt', True)

        # Gradient checkpoint: use torch_checkpoint to save layer activations
        self.grad_cp = getattr(config, 'grad_cp', 1)

        self.save_hyperparameters({
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'vocab_size': config.vocab_size,
            'ctx_len': config.ctx_len,
            'chunk_ctx': getattr(config, 'chunk_ctx', 512),
            'truncated_bptt': self.truncated_bptt,
            'grad_cp': self.grad_cp,
        })

        # Resume training state
        self.skip_until_step = 0
        self.resume_checkpoint_path = None

        # If LoRA enabled, disable strict loading
        if getattr(config, 'use_lora', False):
            self.strict_loading = False

    def set_skip_steps(self, steps: int):
        """Set steps to skip (for resume training)"""
        self.skip_until_step = steps

    def set_resume_checkpoint_path(self, path: str):
        """Set checkpoint path for resume (lazy optimizer loading)"""
        self.resume_checkpoint_path = path

    def _reload_optimizer_state(self):
        """Lazy load optimizer state"""
        if not self.resume_checkpoint_path:
            return

        print(f"\n🔄 [Resume] Re-loading optimizer state from: {self.resume_checkpoint_path}")

        # Check if weights-only file
        try:
            if not os.path.isdir(self.resume_checkpoint_path):
                checkpoint = torch.load(self.resume_checkpoint_path, map_location='cpu')
                if 'optimizer_states' not in checkpoint:
                    print("[Resume] Checkpoint lacks optimizer_states. Skipping.")
                    self.resume_checkpoint_path = None
                    return
        except Exception as e:
            print(f"[Resume] Error checking checkpoint: {e}. Proceeding...")

        try:
            # Use strategy to load checkpoint (DeepSpeed compatible)
            self.trainer.strategy.load_checkpoint(self.resume_checkpoint_path)
            print("[Resume] Optimizer state loaded successfully")
        except Exception as e:
            print(f"[Resume] Failed to load optimizer state: {e}")
            # Fallback
            try:
                checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
                if "optimizer_states" in checkpoint:
                    self.trainer.strategy.load_optimizer_state_dict(checkpoint)
                    print("[Resume] Optimizer state loaded manually")
            except Exception as e2:
                print(f"[Resume] Manual load also failed: {e2}")

        self.resume_checkpoint_path = None  # Prevent repeated loading

    def forward(self, idx, last_shift_states, last_wkv_states):
        """
        Forward pass

        Args:
            idx: [B, T] - Input token IDs
            last_shift_states: [N_layer, 2, B, C] - Shift states
            last_wkv_states: [N_layer, B, H, N, N] - WKV states

        Returns:
            logits: [B, T, vocab_size]
            new_shift_states: [N_layer, 2, B, C]
            new_wkv_states: [N_layer, B, H, N, N]
        """
        return self.model.forward_infctx(idx, last_shift_states, last_wkv_states)

    def training_step(self, batch, batch_idx):
        """
        Training step - supports two modes
        """
        # 1. Dummy forward: skip already trained steps
        if self.global_step < self.skip_until_step:
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # 2. Lazy load optimizer
        if self.global_step == self.skip_until_step and self.resume_checkpoint_path:
            self._reload_optimizer_state()

        if self.truncated_bptt:
            return self._training_step_truncated(batch, batch_idx)
        else:
            return self._training_step_full(batch, batch_idx)

    def _forward_chunk(self, chunk_ids, shift_states, wkv_states):
        """
        chunk Forward pass

        Uses torch_checkpoint based on grad_cp parameter:
        - grad_cp=1: Use checkpoint, saves memory, requires recomputation
        - grad_cp=0: No checkpoint, faster but uses more memory
        """
        if self.grad_cp == 1:
            # Use torch_checkpoint to save memory
            def forward_fn(chunk_ids, shift_states, wkv_states):
                return self.model.forward_infctx(chunk_ids, shift_states, wkv_states)

            return torch_checkpoint(
                forward_fn,
                chunk_ids,
                shift_states,
                wkv_states,
                use_reentrant=False
            )
        else:
            # No checkpoint, direct forward
            return self.model.forward_infctx(chunk_ids, shift_states, wkv_states)

    def _training_step_truncated(self, batch, batch_idx):
        """
        Truncated BPTT mode

        ┌─────────────────────────────────────────────────────────────────┐
        |  Chunk 0:  forward -> loss -> backward                           |
        |                v                                                |
        |            state.detach() <- break gradient!                     |
        |                v                                                |
        |  Chunk 1:  forward -> loss -> backward                           |
        |                v                                                |
        |            state.detach()                                        |
        |                ...                                               |
        │                                                                 │
        |  Memory = O(chunk_size) - handles infinite length                |
        |  Gradients = truncated (not passed to previous chunks)           |
        └─────────────────────────────────────────────────────────────────┘
        """
        config = self.config
        chunk_size = getattr(config, 'chunk_ctx', 512)

        optimizer = self.optimizers()

        idx = batch["input_ids"]
        targets = batch["labels"]
        B, T = idx.shape
        C = config.n_embd
        H = config.dim_att // config.head_size_a

        # Initialize state
        states = BlockStateList.create(
            config.n_layer, B, C, H,
            idx.device, self.model.emb.weight.dtype
        )

        # Process in chunks
        num_chunks = math.ceil(T / chunk_size)
        total_loss = 0.0
        total_valid_tokens = 0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, T)

            chunk_ids = idx[:, start_idx:end_idx]
            chunk_targets = targets[:, start_idx:end_idx]

            # ========== Forward (using checkpoint) ==========
            logits, new_shift_states, new_wkv_states = self._forward_chunk(
                chunk_ids, states.shift_states, states.wkv_states
            )

            # ========== Calculate Loss ==========
            num_valid = (chunk_targets != -100).sum().item()

            if num_valid > 0:
                chunk_loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    chunk_targets.reshape(-1)
                )
                # L2 regularization
                chunk_loss = L2Wrap.apply(chunk_loss, logits)
            else:
                chunk_loss = logits.sum() * 0.0

            # ========== Backward (immediate per chunk) ==========
            # Gradients accumulate to model params
            # Divide by num_chunks for average
            self.manual_backward(chunk_loss / num_chunks)

            # Accumulate statistics
            total_loss += chunk_loss.item() * num_valid
            total_valid_tokens += num_valid

            # ========== Detach state (critical!) ==========
            # After backward, current chunk computation graph can be released
            # Detach state to break gradient connection with previous chunk
            states = BlockStateList(
                new_shift_states.detach(),
                new_wkv_states.detach()
            )

        # ========== Optimizer step ==========
        self._apply_lr_schedule(optimizer)

        if hasattr(config, 'grad_clip') and config.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=config.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        # Calculate average loss
        avg_loss = total_loss / max(total_valid_tokens, 1)

        # Log metrics
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True)
        self.log('mode', 0.0)  # 0 = truncated

        return torch.tensor(avg_loss)

    def _training_step_full(self, batch, batch_idx):
        """
        Full BPTT mode

        ┌─────────────────────────────────────────────────────────────────┐
        |  Chunk 0:  forward ----------------------------------------|      |
        │                ↓                                   │            │
        |            state (keep connected)                          |      |
        |                v                                   | accumulate  |
        |  Chunk 1:  forward ----------------------------------------|      |
        │                ↓                                   │            │
        |            state (keep connected)                          |      |
        |                v                                   v            |
        |  Chunk N:  forward -----------------------------> total_loss      |
        |                                                    |            |
        |                                                    v            |
        |                                               backward          |
        |                                                    |            |
        |                                    gradients to all chunks!      |
        │                                                                 │
        |  Memory = O(seq_len) - no support for very long sequences        |
        |  Gradients = full (passed to all chunks)                         |
        └─────────────────────────────────────────────────────────────────┘
        """
        config = self.config
        chunk_size = getattr(config, 'chunk_ctx', 512)

        optimizer = self.optimizers()

        idx = batch["input_ids"]
        targets = batch["labels"]
        B, T = idx.shape
        C = config.n_embd
        H = config.dim_att // config.head_size_a

        # Initialize state(requires grad for cross-chunk propagation)
        states = BlockStateList.create(
            config.n_layer, B, C, H,
            idx.device, self.model.emb.weight.dtype
        )

        # Process in chunks
        num_chunks = math.ceil(T / chunk_size)
        total_loss = torch.tensor(0.0, device=idx.device, dtype=torch.float32)
        total_valid_tokens = 0
        loss_for_logging = 0.0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, T)

            chunk_ids = idx[:, start_idx:end_idx]
            chunk_targets = targets[:, start_idx:end_idx]

            # ========== Forward (using checkpoint) ==========
            logits, new_shift_states, new_wkv_states = self._forward_chunk(
                chunk_ids, states.shift_states, states.wkv_states
            )

            # ========== Calculate Loss ==========
            num_valid = (chunk_targets != -100).sum().item()

            if num_valid > 0:
                chunk_loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    chunk_targets.reshape(-1)
                )
                # L2 regularization
                chunk_loss = L2Wrap.apply(chunk_loss, logits)
                total_loss = total_loss + chunk_loss
                loss_for_logging += chunk_loss.item() * num_valid

            total_valid_tokens += num_valid

            # ========== No Detach! Keep computation graph connected ==========
            # State keeps gradient connection for backward propagation
            states = BlockStateList(
                new_shift_states,  # No detach!
                new_wkv_states     # No detach!
            )

        # ========== Final unified Backward ==========
        # Backward only after all chunks processed
        # Gradients propagate to all chunks
        self.manual_backward(total_loss / num_chunks)

        # ========== Optimizer step ==========
        self._apply_lr_schedule(optimizer)

        if hasattr(config, 'grad_clip') and config.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=config.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        # Calculate average loss
        avg_loss = loss_for_logging / max(total_valid_tokens, 1)

        # Log metrics
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('lr', optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True)
        self.log('mode', 1.0)  # 1 = full bptt

        return torch.tensor(avg_loss)

    def _apply_lr_schedule(self, optimizer):
        """Apply learning rate schedule"""
        config = self.config

        # Calculate steps per epoch

        epoch_steps = config.epoch_steps if config.epoch_steps is not None else self.trainer.num_training_batches
        if isinstance(epoch_steps, float):
            if not math.isfinite(epoch_steps):
                raise ValueError(
                    "Unable to infer epoch_steps because trainer.num_training_batches is infinite/unknown. "
                    "Please set --epoch_steps to a finite integer (map-style dataset required)."
                )
            epoch_steps = int(epoch_steps)

        # Calculate current step

        real_step = self.trainer.global_step + config.epoch_begin * epoch_steps

        total_training_steps = config.epoch_count * epoch_steps

        # LR schedule (cosine annealing + warmup)

        if config.warmup_steps > 0 and real_step < config.warmup_steps:
            # Warmup phase
            lr = config.lr_init * real_step / config.warmup_steps
        else:
            # Cosine annealing
            # Protection: if warmup_steps >= total_steps, use initial lr
            if config.warmup_steps >= total_training_steps:
                lr = config.lr_init
            else:
                progress = (real_step - config.warmup_steps) / max(total_training_steps - config.warmup_steps, 1)
                progress = min(max(progress, 0.0), 1.0)
                lr = config.lr_final + 0.5 * (config.lr_init - config.lr_final) * (
                    1 + math.cos(math.pi * progress)
                )

        # Apply learning rate (with layerwise lr)
        for param_group in optimizer.param_groups:
            if "my_lr_scale" in param_group:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr

    def configure_optimizers(self):
        """
        Configure optimizer - following standard SFT implementation
        """
        config = self.config

        # Parameter grouping following official implementation
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Official: att.w0 uses 2x learning rate
            if "att.w0" in n:
                lr_2x.add(n)
            # >= 2D params with .weight need weight decay
            elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            # Other params
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        param_dict = {n: p for n, p in self.model.named_parameters()}

        # Build optimizer param groups
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        # Separate group for weight_decay params
        if config.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": config.weight_decay, "my_lr_scale": 1.0}]

        # Select optimizer
        if DEEPSPEED_AVAILABLE and self.deepspeed_offload:
            # DeepSpeed CPU offload optimizer
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                lr=config.lr_init,
                betas=(config.beta1, config.beta2),
                eps=config.adam_eps,
                bias_correction=True,
                adamw_mode=(config.weight_decay > 0),
                amsgrad=False
            )
        elif DEEPSPEED_AVAILABLE:
            # FusedAdam
            optimizer = FusedAdam(
                optim_groups,
                lr=config.lr_init,
                betas=(config.beta1, config.beta2),
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=(config.weight_decay > 0),
                amsgrad=False
            )
        else:
            # Fallback to AdamW (CPU)
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=config.lr_init,
                betas=(config.beta1, config.beta2),
                eps=config.adam_eps,
            )

        return optimizer

    @property
    def deepspeed_offload(self) -> bool:
        """Check if DeepSpeed offload is enabled"""
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
