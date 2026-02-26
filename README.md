# RWKVTune

A comprehensive training toolkit for RWKV language models, supporting Pre-training, SFT, and GRPO.

[![PyPI version](https://badge.fury.io/py/rwkvtune.svg)](https://badge.fury.io/py/rwkvtune)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **[中文文档 (Chinese Documentation)](README_zh.md)**

## Features

- **Three Training Paradigms**
  - `PretrainTrainer`: Continue pre-training from existing models
  - `SFTTrainer`: Supervised Fine-Tuning for instruction following
  - `GRPOTrainer`: Group Relative Policy Optimization for RLHF

- **Efficient Training**
  - Multi-GPU training with DeepSpeed ZeRO Stage 2/3
  - Gradient checkpointing for memory efficiency
  - Mixed precision training (bf16 / fp16 / fp32)
  - Gradient accumulation

- **Parameter-Efficient Fine-Tuning (PEFT)**
  - LoRA with customizable target modules
  - One-command LoRA weight merging

- **Advanced Capabilities**
  - Infinite context training via chunked BPTT
  - HuggingFace Datasets integration
  - Checkpoint resume and elastic training
  - Multiple GRPO loss functions: DAPO / DR-GRPO / BNPO / GRPO
  - Completion post-processing hook for custom rollout filtering
  - Rollout data saving for analysis and debugging
  - Logging integration: SwanLab / WandB / TensorBoard

- **CLI Tools**
  - `rwkvtune-merge-lora`: Merge LoRA adapters into base model
  - `rwkvtune-create-hub`: Create standard model directory

## Installation

### From PyPI (Recommended)

```bash
pip install rwkvtune
```

### From Source

```bash
git clone https://github.com/rwkv-community/rwkvtune.git
cd rwkvtune
pip install -e .
```

### With DeepSpeed Support

```bash
pip install rwkvtune[deepspeed]
```

### Development Installation

```bash
pip install rwkvtune[dev]
```

## Model Preparation

Before training, convert RWKV model weights into the RWKVTune standard hub-style directory.

### Option 1: Convert with rwkvtune-create-hub

```bash
rwkvtune-create-hub \
    --output-dir models/rwkv7-0.1b \
    --model-file /path/to/rwkv7-0.1b.pth \
    --config-name rwkv7-0.1b
```

The resulting directory layout:

```
models/rwkv7-0.1b/
  config.json
  model.pth
  tokenizer_config.json
  vocab.txt
  generation_config.json
```

### Option 2: Download from ModelScope

```python
from modelscope import snapshot_download
model_dir = snapshot_download('aierwiki/rwkv7-g1d-0.1b')
```

## Quick Start

### Supervised Fine-Tuning (SFT)

```python
from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.training import SFTConfig, SFTTrainer
from datasets import Dataset

# Load model and tokenizer
model = AutoModel.from_pretrained("/path/to/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Prepare dataset (must contain 'input_ids' and 'labels' fields)
def prepare_data(example):
    prompt = f"User: {example['instruction']}\n\nAssistant: "
    completion = example['output']
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels}

dataset = Dataset.from_list([...]).map(prepare_data)

# Configure training
config = SFTConfig(
    ctx_len=2048,
    micro_bsz=4,
    epoch_count=3,
    lr_init=2e-5,
    lr_final=1e-6,
    precision="bf16",
)

# Create trainer and start training
trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### SFT with LoRA

```python
from rwkvtune import AutoModel
from rwkvtune.peft import LoraConfig, get_peft_model
from rwkvtune.training import SFTConfig, SFTTrainer

model = AutoModel.from_pretrained("/path/to/model")

# Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
)
model = get_peft_model(model, lora_config)

config = SFTConfig(
    ctx_len=2048,
    micro_bsz=4,
    epoch_count=3,
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

### GRPO Training

```python
from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.training import GRPOConfig, GRPOTrainer
from rwkvtune.peft import LoraConfig
from datasets import Dataset

# Define reward function
def reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        rewards.append(1.0 if "correct" in completion else 0.0)
    return rewards

# Prepare dataset (must contain 'prompt' and 'input_ids' fields)
dataset = Dataset.from_list([
    {"prompt": "What is 2+2?", "input_ids": [...]},
])

# Configure GRPO
config = GRPOConfig(
    micro_bsz=1,
    num_generations=8,
    max_prompt_length=2048,
    max_completion_length=512,
    steps_per_generation=8,
    temperature=1.0,
    top_p=0.95,
    lr_init=1e-6,
    lr_final=1e-7,
    accumulate_grad_batches=8,
    loss_type="dapo",
    beta=0.04,
    save_rollout_steps=16,
)

# LoRA config
lora_config = LoraConfig(r=64, lora_alpha=128)

# Create trainer
trainer = GRPOTrainer(
    model="/path/to/model",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
    peft_config=lora_config,
)
trainer.train()
```

### Continue Pre-training

```python
from rwkvtune import AutoModel
from rwkvtune.training import PretrainConfig, PretrainTrainer
from datasets import Dataset

# Prepare dataset (must contain 'input_ids' and 'labels' fields)
dataset = Dataset.from_list([
    {"input_ids": [...], "labels": [...]},
])

config = PretrainConfig(
    ctx_len=4096,
    micro_bsz=8,
    epoch_count=1,
    lr_init=3e-4,
)

trainer = PretrainTrainer(
    model="/path/to/model",
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

## CLI Tools

### rwkvtune-create-hub -- Create Standard Model Directory

Convert raw RWKV `.pth` weight files into a HuggingFace-style model directory that can be loaded with `AutoModel.from_pretrained()`.

```bash
rwkvtune-create-hub \
    --output-dir models/rwkv7-0.1b \
    --model-file /path/to/rwkv7-0.1b.pth \
    --config-name rwkv7-0.1b
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory (required) |
| `--model-file` | Path to model weight file (required) |
| `--config-name` | Predefined model config name (required), see table below |
| `--ctx-len` | Override default context length |
| `--chat-template` | Chat template file path (.jinja format) |
| `--link-weights` | Create symlink instead of copying weights (saves disk space) |
| `--save-format` | Weight save format: pth (default) / safetensors |
| `--overwrite` | Overwrite existing directory |
| `--verbose` | Show verbose output |

Available model configs:

| config-name | Layers | Dim | Params |
|-------------|--------|-----|--------|
| rwkv7-0.1b | 12 | 768 | 0.1B |
| rwkv7-0.4b | 24 | 1024 | 0.4B |
| rwkv7-1.5b | 24 | 2048 | 1.5B |
| rwkv7-2.9b | 32 | 2560 | 2.9B |
| rwkv7-7.2b | 32 | 4096 | 7.2B |
| rwkv7-13.3b | - | - | 13.3B |

### rwkvtune-merge-lora -- Merge LoRA Weights

After training, merge LoRA adapter weights into the base model to produce a standalone model.

```bash
rwkvtune-merge-lora \
    --base-model models/rwkv7-g1d-0.1b \
    --lora-model output_sft/rwkv7-epoch3 \
    --output models/rwkv7-g1d-0.1b-merged
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--base-model`, `-b` | Base model directory (required) |
| `--lora-model`, `-l` | LoRA model directory containing adapter_model.bin (required) |
| `--output`, `-o` | Output directory for merged model (required) |
| `--device` | Device for merging: cuda / cpu (default: cpu) |
| `--precision` | Weight precision: fp32 / fp16 / bf16 (default: bf16) |
| `--save-format` | Output format: pth (default) / safetensors |

After merging, the tool automatically:
- Merges LoRA adapters into base weights
- Updates config.json (syncs eos_token_id, etc.)
- Copies tokenizer files (prefers LoRA model's version)
- Creates merge_info.json with merge metadata

## Example Scripts

The `examples/` directory provides ready-to-use training scripts:

| Script | Description |
|--------|-------------|
| `run_sft_single_gpu.sh` | Single-GPU SFT training with LoRA support |
| `run_grpo_single_gpu.sh` | Single-GPU GRPO training |
| `run_sft_grpo_single_gpu.sh` | SFT + GRPO combined pipeline |
| `run_test_generation.sh` | Model generation testing |
| `train_sft_single_gpu.py` | SFT training Python script |
| `train_grpo_single_gpu.py` | GRPO training Python script |
| `test_generation.py` | Generation test Python script |

### Usage

```bash
cd examples/

# SFT training
MODEL_PATH=/path/to/model DATA_FILE=data/train.jsonl ./run_sft_single_gpu.sh

# GRPO training
./run_grpo_single_gpu.sh

# Test generation
./run_test_generation.sh
```

Shell scripts are configured via environment variables at the top of each file, which can also be overridden from the command line:

```bash
MODEL_PATH=models/rwkv7-0.4b CTX_LEN=4096 MICRO_BSZ=4 ./run_sft_single_gpu.sh
```

## Multi-GPU Training

RWKVTune supports multi-GPU training via PyTorch Lightning:

```python
config = SFTConfig(
    devices=4,                          # Number of GPUs
    strategy="deepspeed_stage_2",       # DeepSpeed ZeRO Stage 2
    precision="bf16",
)
```

## Configuration Reference

### SFTConfig / PretrainConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ctx_len` | int | 2048 | Context length |
| `micro_bsz` | int | 4 | Batch size per GPU |
| `epoch_count` | int | 10 | Number of training epochs |
| `epoch_steps` | int | None | Steps per epoch limit (None = full dataset) |
| `epoch_save` | int | 1 | Save interval by epoch |
| `save_every_n_batches` | int | 0 | Save interval by batch (0 = disabled) |
| `save_total_limit` | int | 2 | Max checkpoints to keep |
| `lr_init` | float | 3e-4 | Initial learning rate |
| `lr_final` | float | 1e-5 | Final learning rate |
| `warmup_steps` | int | 50 | Warmup steps |
| `weight_decay` | float | 0.0 (SFT) / 0.01 (Pretrain) | Weight decay |
| `accumulate_grad_batches` | int | 1 | Gradient accumulation steps |
| `grad_cp` | int | 1 | Gradient checkpointing (0 = off, 1 = on) |
| `grad_clip` | float | 1.0 | Gradient clipping threshold |
| `devices` | int | 1 | Number of GPUs |
| `precision` | str | bf16 | Training precision |
| `strategy` | str | auto | Training strategy (auto / ddp / deepspeed_stage_2 / deepspeed_stage_3) |
| `train_type` | str | normal | Training mode (normal / infctx) |
| `resume_from_checkpoint` | str | None | Checkpoint path for resuming |
| `report_to` | str | "" | Logging backend (swanlab / tensorboard / empty string) |

### GRPOConfig

**Core GRPO parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | 8 | Completions per prompt (G) |
| `num_iterations` | int | 1 | Training iterations per batch |
| `steps_per_generation` | int | 1 | Training steps per rollout |
| `micro_bsz` | int | 1 | Prompts per GPU per batch |

**Generation parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_prompt_length` | int | 512 | Max prompt length (tokens) |
| `max_completion_length` | int | 256 | Max completion length (tokens) |
| `temperature` | float | 1.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `top_k` | int | 0 | Top-K sampling (0 = disabled) |
| `repetition_penalty` | float | 1.0 | Repetition penalty (>1 to suppress) |

**Loss and optimization:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_type` | str | dapo | Loss type: dapo / dr_grpo / bnpo / grpo |
| `epsilon` | float | 0.2 | PPO clip lower bound |
| `epsilon_high` | float | None | PPO clip upper bound (None = same as epsilon) |
| `scale_rewards` | str | group | Advantage normalization: group / batch / none |
| `advantage_clip` | float | None | Clamp advantages to [-clip, +clip] |
| `low_reward_threshold` | float | None | Low-quality group suppression threshold |
| `low_reward_scale` | float | 0.01 | Advantage scale factor for low-quality groups |

**KL divergence penalty:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 0.0 | KL penalty coefficient (0 = disabled) |
| `kl_approximator` | str | schulman | KL approximation method |

**Reward functions:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_weights` | List[float] | None | Weights for multiple reward functions |

**Checkpointing and logging:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_every_n_batches` | int | 0 | Save interval by batch (0 = disabled) |
| `save_total_limit` | int | 2 | Max checkpoints to keep |
| `save_rollout_steps` | int | 0 | Rollout data save interval (0 = disabled) |
| `save_rollout_path` | str | None | Rollout data save directory |
| `report_to` | str | None | Logging backend: swanlab / wandb |

**RWKV generation optimization:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefill_chunk_size` | int | 2048 | Chunked prefill size |
| `max_prefill_batch_size` | int | -1 | Max prefill batch size (-1 = unlimited) |
| `max_decode_batch_size` | int | -1 | Max decode batch size (-1 = unlimited) |
| `logprob_batch_size` | int | None | Log-prob computation chunk size (memory optimization) |

### LoraConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 64 | LoRA rank |
| `lora_alpha` | int | 128 | LoRA alpha |
| `lora_dropout` | float | 0.0 | LoRA dropout |
| `target_modules` | list | auto | Modules to apply LoRA to |

## Advanced Features

### Infinite Context Training

Train on ultra-long sequences via chunked processing and truncated BPTT:

```python
config = SFTConfig(
    train_type="infctx",
    ctx_len=32768,
    chunk_ctx=512,
    truncated_bptt=True,
    grad_cp=1,
)
```

### GRPO Completion Post-Processing Hook

Insert custom post-processing logic after rollout generation and before reward computation (e.g., truncating at stop words, appending EOS):

```python
import torch

def my_postprocessor(prompts, completions, completion_ids, masks, tokenizer, **extra_fields):
    # 1. Text-level processing (truncation, cleanup, etc.)
    new_completions = [process(text) for text in completions]

    # 2. Re-encode into token IDs and rebuild masks
    B_G, C = completion_ids.shape
    new_ids = torch.zeros(B_G, C, dtype=completion_ids.dtype, device=completion_ids.device)
    new_masks = torch.zeros(B_G, C, dtype=masks.dtype, device=completion_ids.device)
    for i, text in enumerate(new_completions):
        token_ids = tokenizer.encode(text)
        n = min(len(token_ids), C)
        new_ids[i, :n] = torch.tensor(token_ids[:n], dtype=completion_ids.dtype)
        new_masks[i, :n] = True

    return {"completions": new_completions, "completion_ids": new_ids, "masks": new_masks}

trainer = GRPOTrainer(
    model="/path/to/model",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
    completion_postprocess_fn=my_postprocessor,
)
```

Post-processing function contract:
- Input: prompts (list of prompt texts), completions (list of generated texts), completion_ids (token ID tensor [B*G, C]), masks (bool mask tensor [B*G, C]), tokenizer, and any extra dataset fields
- Output: a dict with keys `completions`, `completion_ids`, and `masks`
- After text modification, re-encode via `tokenizer.encode()` to produce new completion_ids -- the old token IDs may no longer match the modified text
- Returned tensors must keep the same [B*G, C] shape; pad with 0 if shorter, truncate if longer

### Multiple Reward Functions

GRPOTrainer supports combining multiple reward functions with weights:

```python
def format_reward(prompts, completions, **kwargs):
    return [1.0 if is_well_formatted(c) else 0.0 for c in completions]

def quality_reward(prompts, completions, **kwargs):
    return [score_quality(c) for c in completions]

trainer = GRPOTrainer(
    model="/path/to/model",
    reward_funcs=[format_reward, quality_reward],
    args=GRPOConfig(reward_weights=[0.5, 1.0]),
    train_dataset=dataset,
)
```

### Rollout Data Saving

Save prompt, completion, and reward data from each rollout for analysis and debugging:

```python
config = GRPOConfig(
    save_rollout_steps=16,
    save_rollout_path="output_grpo/rollouts",
)
```

## Typical Workflow

### Full Pipeline: SFT -> LoRA Merge -> GRPO

```bash
# 1. SFT training
cd examples/
./run_sft_single_gpu.sh

# 2. Merge LoRA weights
rwkvtune-merge-lora \
    --base-model models/rwkv7-g1d-0.1b \
    --lora-model output_sft/rwkv7-epoch3 \
    --output models/rwkv7-sft-merged

# 3. GRPO training (on top of the SFT model)
MODEL_PATH=models/rwkv7-sft-merged ./run_grpo_single_gpu.sh

# 4. Merge GRPO LoRA weights
rwkvtune-merge-lora \
    --base-model models/rwkv7-sft-merged \
    --lora-model output_grpo/rwkv7-batch512 \
    --output models/rwkv7-final

# 5. Test generation
./run_test_generation.sh
```

## Model Support

Currently supported architectures:
- RWKV-7 (all sizes: 0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Lightning >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- CUDA (recommended for training)

## License

Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{rwkvtune,
  title = {RWKVTune: RWKV Model Training Toolkit},
  year = {2024},
  url = {https://github.com/rwkv-community/rwkvtune}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [RWKV](https://github.com/BlinkDL/RWKV-LM) - The original RWKV implementation
- [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) - Reference for PEFT implementation
- [trl](https://github.com/huggingface/trl) - API design inspiration
