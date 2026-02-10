# RWKVTune

RWKV Model Training Toolkit - A comprehensive library for training RWKV language models.

[![PyPI version](https://badge.fury.io/py/rwkvtune.svg)](https://badge.fury.io/py/rwkvtune)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Three Training Paradigms**
  - `PretrainTrainer`: Continue pre-training from existing models
  - `SFTTrainer`: Supervised Fine-Tuning for instruction following
  - `GRPOTrainer`: GRPO (Group Relative Policy Optimization) for RLHF

- **Efficient Training**
  - Multi-GPU training with DeepSpeed ZeRO optimization
  - Gradient checkpointing for memory efficiency
  - Mixed precision training (bf16/fp16/fp32)

- **Parameter-Efficient Fine-Tuning**
  - LoRA support with customizable target modules
  - Easy adapter merging and saving

- **Advanced Capabilities**
  - Infinite context training support
  - HuggingFace Datasets integration
  - Checkpoint resume and elastic training

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

## Quick Start

### Supervised Fine-Tuning (SFT)

```python
from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.training import SFTConfig, SFTTrainer
from datasets import Dataset

# Load model and tokenizer
model = AutoModel.from_pretrained("/path/to/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Prepare dataset (must have 'input_ids' and 'labels')
def prepare_data(examples):
    # Your data preprocessing logic
    return {"input_ids": [...], "labels": [...]}

dataset = Dataset.from_list([...])
dataset = dataset.map(prepare_data)

# Configure training
config = SFTConfig(
    ctx_len=2048,
    micro_bsz=4,
    epoch_count=3,
    lr_init=1e-4,
    devices=1,
    precision="bf16",
)

# Create trainer and train
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

# Load model
model = AutoModel.from_pretrained("/path/to/model")

# Apply LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
)
model = get_peft_model(model, lora_config)

# Configure training
config = SFTConfig(
    ctx_len=2048,
    micro_bsz=4,
    epoch_count=3,
)

# Train
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
from datasets import Dataset

# Define reward function
def reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Your reward logic
        rewards.append(1.0 if "correct" in completion else 0.0)
    return rewards

# Prepare dataset (must have 'prompt' and 'input_ids')
dataset = Dataset.from_list([
    {"prompt": "What is 2+2?", "input_ids": [...]},
    ...
])

# Configure GRPO
config = GRPOConfig(
    ctx_len=2048,
    micro_bsz=2,
    num_generations=4,
    epoch_count=1,
)

# Create trainer
trainer = GRPOTrainer(
    model="/path/to/model",
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

### Continue Pre-training

```python
from rwkvtune import AutoModel
from rwkvtune.training import PretrainConfig, PretrainTrainer
from datasets import Dataset

# Prepare dataset (must have 'input_ids' and 'labels')
dataset = Dataset.from_list([
    {"input_ids": [...], "labels": [...]},
    ...
])

# Configure pre-training
config = PretrainConfig(
    ctx_len=4096,
    micro_bsz=8,
    epoch_count=1,
    lr_init=3e-4,
)

# Create trainer
trainer = PretrainTrainer(
    model="/path/to/model",
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

## Command Line Tools

### Merge LoRA Weights

```bash
rwkvtune-merge-lora \
    --base-model /path/to/base \
    --lora-model /path/to/lora \
    --output /path/to/merged
```

## Multi-GPU Training

RWKVTune supports multi-GPU training with DeepSpeed:

```python
config = SFTConfig(
    devices=4,                          # Number of GPUs
    strategy="deepspeed_stage_2",       # DeepSpeed ZeRO Stage 2
    precision="bf16",
)
```

Or with environment variables:

```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py

# Using DeepSpeed launcher
deepspeed --num_gpus=4 train.py
```

## Configuration Options

### SFTConfig / PretrainConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ctx_len` | int | 1024 | Context length |
| `micro_bsz` | int | 4 | Batch size per GPU |
| `epoch_count` | int | 10 | Number of epochs |
| `lr_init` | float | 3e-4 | Initial learning rate |
| `lr_final` | float | 1e-5 | Final learning rate |
| `warmup_steps` | int | 50 | Warmup steps |
| `grad_cp` | int | 0 | Gradient checkpointing (0=off, 1=on) |
| `devices` | int | 1 | Number of GPUs |
| `precision` | str | "bf16" | Training precision |
| `strategy` | str | "auto" | Training strategy |

### GRPOConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | int | 4 | Completions per prompt |
| `beta` | float | 0.04 | KL penalty coefficient |
| `temperature` | float | 1.0 | Sampling temperature |
| `max_new_tokens` | int | 256 | Max tokens to generate |

### LoraConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 64 | LoRA rank |
| `lora_alpha` | int | 128 | LoRA alpha |
| `lora_dropout` | float | 0.0 | LoRA dropout |
| `target_modules` | list | auto | Modules to apply LoRA |

## Model Support

Currently supported models:
- RWKV-7 (all sizes: 0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B)

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Lightning >= 2.0.0
- CUDA (recommended for training)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use RWKVTune in your research, please cite:

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
