"""
GRPO Training Configuration.

Defines all hyperparameters and settings for Group Relative Policy Optimization training.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    # ========== Model Configuration ==========
    ref_model_path: Optional[str] = None  # Reference model path (optional)

    # Model architecture parameters (loaded from model_config, not manually set)
    n_layer: Optional[int] = None
    n_embd: Optional[int] = None
    vocab_size: Optional[int] = None
    head_size_a: Optional[int] = None
    dim_att_lora: Optional[int] = None
    dim_gate_lora: Optional[int] = None
    dim_mv_lora: Optional[int] = None

    # ========== Infinite Context Configuration ==========
    train_type: str = "normal"  # Training type: normal, infctx
    chunk_ctx: int = 512  # Chunk size for infctx mode
    truncated_bptt: bool = True  # Truncated BPTT mode
    grad_cp: int = 0  # Gradient checkpoint

    # ========== GRPO Core Parameters ==========
    num_generations: int = 8  # Number of completions per prompt (G)
    num_iterations: int = 1  # Training iterations per batch (mu)

    # ========== Generation Parameters ==========
    max_prompt_length: int = 512  # Maximum prompt length
    max_completion_length: int = 256  # Maximum completion length
    temperature: float = 1.0  # Sampling temperature
    top_p: float = 1.0  # Nucleus sampling
    top_k: int = 0  # Top-k sampling (0 = disabled)

    # ========== Reward Configuration ==========
    reward_weights: Optional[List[float]] = None  # Multi-reward function weights

    # ========== Advantage Calculation ==========
    scale_rewards: str = "group"  # 'group', 'batch', 'none'

    # ========== Loss Function ==========
    loss_type: str = "dapo"  # 'dapo', 'dr_grpo', 'bnpo', 'grpo'
    epsilon: float = 0.2  # PPO clip lower bound
    epsilon_high: Optional[float] = None  # PPO clip upper bound (None = same as epsilon)
    delta: Optional[float] = None  # Additional upper bound clip (optional)

    # ========== KL Penalty ==========
    beta: float = 0.0  # KL coefficient (0 = disabled)
    kl_approximator: str = "schulman"  # KL approximation method

    # ========== Importance Sampling ==========
    importance_sampling_level: str = "token"  # 'token', 'sequence'

    # ========== Entropy Mask ==========
    top_entropy_quantile: float = 1.0  # Entropy quantile (1.0 = disabled)

    # ========== RWKV Generation Optimization ==========
    prefill_chunk_size: int = 2048  # Chunked Prefill chunk size
    max_prefill_batch_size: int = -1  # Max prefill batch size (-1 = unlimited)
    max_decode_batch_size: int = -1  # Max decode batch size (-1 = unlimited)
    logit_bias: Optional[dict] = None  # Token generation bias
    use_torch_compile: bool = False  # Use torch.compile (experimental)
    generation_batch_mode: str = "sequential"  # 'sequential', 'parallel' (deprecated)

    # ========== Log Probabilities Calculation (Memory Optimization) ==========
    logprob_batch_size: Optional[int] = None  # Chunk size for logprobs calculation

    # ========== Reference Model Sync ==========
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6

    # ========== Training Parameters ==========
    lr_init: float = 1e-6  # Initial learning rate
    lr_final: float = 1e-7  # Final learning rate
    weight_decay: float = 0.01
    warmup_steps: int = 50
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.99  # Adam beta2
    adam_eps: float = 1e-8  # Adam epsilon
    grad_clip: float = 1.0  # Gradient clipping
    accumulate_grad_batches: int = 1  # Gradient accumulation steps
    layerwise_lr: int = 1  # Layer-wise learning rate

    # Data parameters
    micro_bsz: int = 1  # Prompts per GPU batch
    num_workers: int = 0  # DataLoader workers
    generation_batch_size: int = 8  # Generation batch size
    steps_per_generation: int = 1  # Training steps per generation
    shuffle_buffer: bool = True  # Shuffle buffer before training

    # Hardware parameters
    accelerator: str = "gpu"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "bf16"

    # Other parameters
    proj_dir: str = "output_grpo"
    seed: int = 42
    epoch_count: int = 1
    epoch_steps: Optional[int] = None  # Steps per epoch (None = auto-calculate)
    save_every_n_batches: int = 0  # Save interval by batch (0 = disabled)
    save_total_limit: int = 2  # Max checkpoints to keep (0 = unlimited)
    save_optimizer_state: bool = True  # Save optimizer state
    resume_from_checkpoint: Optional[str] = None  # Checkpoint resume path
    log_steps: int = 10

    # ========== LoRA Save Parameters ==========
    use_lora: bool = False
    lora_save_mode: str = "lora_only"  # 'lora_only', 'full', 'both'

    # ========== Logging and Monitoring ==========
    report_to: Optional[str] = None  # None, "swanlab", "wandb"
    run_name: Optional[str] = None

    # ========== Rollout Data Saving ==========
    save_rollout_steps: int = 0  # Save rollout data interval (0 = disabled)
    save_rollout_path: Optional[str] = None

    # ========== Advanced Options ==========
    mask_truncated_completions: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self):
        """Post-initialization processing."""
        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        # Validate config
        assert self.num_generations > 0, "num_generations must be > 0"
        assert self.scale_rewards in ['group', 'batch', 'none'], \
            f"scale_rewards must be one of ['group', 'batch', 'none'], got {self.scale_rewards}"
        assert self.loss_type in ['dapo', 'dr_grpo', 'bnpo', 'grpo'], \
            f"loss_type must be one of ['dapo', 'dr_grpo', 'bnpo', 'grpo'], got {self.loss_type}"
        assert self.importance_sampling_level in ['token', 'sequence'], \
            f"importance_sampling_level must be one of ['token', 'sequence'], got {self.importance_sampling_level}"

        # Key parameter constraint validation
        samples_per_rollout = self.micro_bsz * self.num_generations

        if self.steps_per_generation <= 0:
            raise ValueError(f"steps_per_generation must > 0, got {self.steps_per_generation}")

        if samples_per_rollout < self.steps_per_generation:
            raise ValueError(
                f"micro_bsz x num_generations ({self.micro_bsz} x {self.num_generations} = {samples_per_rollout}) "
                f"must >= steps_per_generation ({self.steps_per_generation})"
            )

        samples_per_step = samples_per_rollout // self.steps_per_generation
        if samples_per_step == 0:
            raise ValueError(
                f"Samples per training_step is 0\n"
                f"micro_bsz x num_generations = {samples_per_rollout}\n"
                f"steps_per_generation = {self.steps_per_generation}"
            )

        generate_every = self.steps_per_generation * self.num_iterations
        if generate_every % self.accumulate_grad_batches != 0:
            raise ValueError(
                f"generate_every must be divisible by accumulate_grad_batches.\n"
                f"generate_every = {generate_every}, accumulate_grad_batches = {self.accumulate_grad_batches}"
            )

        if self.accumulate_grad_batches > 16:
            raise ValueError(
                f"accumulate_grad_batches={self.accumulate_grad_batches} is too large (>16)."
            )

        if self.train_type == "infctx":
            assert self.chunk_ctx > 0, "chunk_ctx must be > 0 for infctx training"

        self.ctx_len = self.max_prompt_length + self.max_completion_length

        if self.save_rollout_steps > 0 and self.save_rollout_path is None:
            import os
            self.save_rollout_path = os.path.join(self.proj_dir, "rollouts")

        if self.report_to is not None:
            assert self.report_to in ['swanlab', 'wandb'], \
                f"report_to must be one of ['swanlab', 'wandb', None], got {self.report_to}"
