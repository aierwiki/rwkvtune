"""
RWKV7 Lightning Module - Multi-GPU training support
"""
import os
import math
import torch
import torch.nn as nn
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

from rwkvtune.models.rwkv7 import RWKV7Model

class L2Wrap(torch.autograd.Function):
    """L2 regularization wrapper - encourages logits to stay near 0"""

    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # Encourage logits near 0 to prevent overflow
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class RWKV7LightningModule(pl.LightningModule):
    """
    RWKV7 Lightning Module

    Features:
    - Multi-GPU training (DDP)
    - DeepSpeed optimization
    - Automatic mixed precision
    - Gradient accumulation
    """

    def __init__(self, config, model: nn.Module = None):
        """
        Args:
            config: Training configuration
            model: Optional, external model (supports LoRA).
                   If None, creates RWKV7Model internally.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['model'])

        # Use external model or create new one
        if model is not None:
            self.model = model
        else:
            self.model = RWKV7Model(config)

        # Loss function (ignore_index=-100 ignores padding/prompt)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Resume training state
        self.skip_until_step = 0
        self.resume_checkpoint_path = None
        self.disable_optimizer_resume = False
        self.resume_model_weights_path = None

        # If LoRA enabled, allow loading checkpoints without base model params
        if getattr(config, 'use_lora', False):
            self.strict_loading = False

    def set_skip_steps(self, steps: int):
        """Set steps to skip (for resume training)"""
        self.skip_until_step = steps

    def set_resume_checkpoint_path(self, path: str):
        """Set checkpoint path for resume (lazy optimizer loading)"""
        self.resume_checkpoint_path = path

    def set_resume_model_weights_path(self, path: str):
        self.resume_model_weights_path = path

    def set_disable_optimizer_resume(self, disable: bool = True):
        """Disable optimizer state resume (for elastic resume when world size changes)"""
        self.disable_optimizer_resume = disable

    def on_save_checkpoint(self, checkpoint):
        """Save extra state to checkpoint"""
        # Save total samples seen for elastic resume
        # samples_seen = global_step * micro_bsz * devices * accumulate_grad_batches
        if hasattr(self.config, 'micro_bsz') and hasattr(self.config, 'devices'):
            samples_seen = self.global_step * self.config.micro_bsz * self.config.devices * self.config.accumulate_grad_batches
            checkpoint['samples_seen'] = samples_seen

            # Save metadata for elastic resume
            checkpoint['devices'] = int(getattr(self.config, 'devices', 1))
            checkpoint['micro_bsz'] = int(getattr(self.config, 'micro_bsz', 1))
            checkpoint['accumulate_grad_batches'] = int(getattr(self.config, 'accumulate_grad_batches', 1))

        # LoRA slimming: if saving only LoRA weights, remove base model params
        # This significantly reduces .ckpt file size while preserving optimizer state
        if getattr(self.config, 'use_lora', False) and getattr(self.config, 'lora_save_mode', 'lora_only') == 'lora_only':
            state_dict = checkpoint['state_dict']
            keys_to_remove = []

            # Iterate all model parameters
            for name, param in self.named_parameters():
                # Remove frozen base model params from checkpoint
                if not param.requires_grad:
                    # Lightning state_dict keys typically match named_parameters

                    if name in state_dict:
                        keys_to_remove.append(name)

            if keys_to_remove:
                print(f"[Checkpoint] LoRA mode: removing {len(keys_to_remove)} frozen base model params")
                for k in keys_to_remove:
                    del state_dict[k]

    def _reload_optimizer_state(self):
        """Lazy load optimizer state"""
        if not self.resume_checkpoint_path:
            return

        if self.disable_optimizer_resume:
            self.resume_checkpoint_path = None
            return

        print(f"\n🔄 [Resume] Re-loading optimizer state from: {self.resume_checkpoint_path}")

        # Check if weights-only file (e.g. adapter_model.bin)
        try:
            if not os.path.isdir(self.resume_checkpoint_path):

                checkpoint = torch.load(self.resume_checkpoint_path, map_location='cpu', weights_only=False)
                if 'optimizer_states' not in checkpoint:
                    print("[Resume] Checkpoint lacks optimizer_states. Skipping optimizer load.")
                    self.resume_checkpoint_path = None
                    return
        except Exception as e:
            print(f"[Resume] Error checking checkpoint type: {e}. Proceeding...")

        try:
            # Use strategy to load checkpoint (compatible with DeepSpeed)

            self.trainer.strategy.load_checkpoint(self.resume_checkpoint_path)
            print("[Resume] Optimizer state loaded successfully")
        except Exception as e:
            print(f"[Resume] Failed to load optimizer state: {e}")
            # Fallback: try manual loading for standard checkpoint
            try:
                checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device, weights_only=False)
                if "optimizer_states" in checkpoint:
                    self.trainer.strategy.load_optimizer_state_dict(checkpoint)
                    print("[Resume] Optimizer state loaded manually")
            except Exception as e2:
                print(f"[Resume] Manual load also failed: {e2}")

        self.resume_checkpoint_path = None  # Prevent repeated loading

    def forward(self, idx):
        """Forward pass"""
        return self.model(idx)

    def training_step(self, batch, batch_idx):
        """Training step - following official RWKV-LM implementation"""
        # 1. Dummy forward: skip already trained steps
        if self.global_step < self.skip_until_step:
            dummy_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            p0 = next((p for p in self.parameters() if p.requires_grad), None)
            if p0 is not None:
                dummy_loss = dummy_loss + p0.view(-1)[0].float() * 0.0
            return dummy_loss

        # 2. Lazy load optimizer: load when real training starts
        if self.global_step == self.skip_until_step and self.resume_checkpoint_path:
            if self.resume_model_weights_path:
                state_dict = torch.load(self.resume_model_weights_path, map_location='cpu', weights_only=False)
                self.load_state_dict(state_dict, strict=False)
                self.resume_model_weights_path = None
            self._reload_optimizer_state()

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = self(input_ids)

        # Calculate loss directly (dataset already handles shifting)

        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # L2 regularization (following official implementation)

        # Apply L2Wrap following RWKV-PEFT logic

        return L2Wrap.apply(loss, logits)

    def configure_optimizers(self):
        """
        Configure optimizer - following RWKV-PEFT parameter grouping

        Parameter groups:
        - lr_1x: Standard learning rate (most params)
        - lr_2x: 2x learning rate (time_decay)
        - lr_3x: 3x learning rate (time_first)
        """
        config = self.config

        # Parameter grouping following RWKV-PEFT
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # RWKV-PEFT logic:
            # time_first uses 3x learning rate
            # time_decay uses 2x learning rate
            # time_mix uses 1x learning rate

            if ("time_first" in n):
                lr_3x.add(n)
            elif ("time_decay" in n) or ("time_daaaa" in n):
                lr_2x.add(n)
            elif ("time_mix" in n) or ("time_maa" in n):

                lr_1x.add(n)
            # >= 2D params with .weight need weight decay
            elif (len(p.squeeze().shape) >= 2) and (config.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            # Other params use 1x learning rate
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))

        param_dict = {n: p for n, p in self.model.named_parameters()}

        # Build optimizer param groups
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
            {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
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
                adamw_mode=True, # Force adamw_mode
                amsgrad=False
            )
        elif DEEPSPEED_AVAILABLE and self.device.type != 'cpu':
            # FusedAdam (GPU only)
            optimizer = FusedAdam(
                optim_groups,
                lr=config.lr_init,
                betas=(config.beta1, config.beta2),
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=True, # Force adam_w_mode for RWKV-PEFT compatibility
                amsgrad=False
            )
        else:
            # Fallback to standard Adam/AdamW (CPU)
            print("Using standard PyTorch optimizer (CPU mode)")
            if config.weight_decay > 0:
                optimizer = torch.optim.AdamW(
                    optim_groups,
                    lr=config.lr_init,
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps,
                    weight_decay=config.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    optim_groups,
                    lr=config.lr_init,
                    betas=(config.beta1, config.beta2),
                    eps=config.adam_eps
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Optimizer step

        LR scheduling moved to on_train_batch_start (following RWKV-PEFT)
        This only executes the optimization step
        """

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

