# GRPO Training Module

Group Relative Policy Optimization (GRPO) reinforcement learning training module.

## Quick Start

### Command Line Usage

```bash
python -m rwkvtune.cli.train_grpo \
  --load_model /path/to/model.pth \
  --data_file prompts.jsonl \
  --num_generations 8 \
  --devices 1 \
  --proj_dir output_grpo
```

### Python API Usage

```python
from rwkvtune.training.grpo import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    load_model="/path/to/model.pth",
    data_file="prompts.jsonl",
    num_generations=8,
)

trainer = GRPOTrainer(config)
trainer.train()
```

## Module Components

- `config.py`: Configuration class
- `advantage.py`: Advantage calculator
- `loss.py`: Loss functions (DAPO, Dr.GRPO, BNPO)
- `generation.py`: RWKV generation engine
- `reward.py`: Reward functions
- `dataset.py`: Dataset and data buffer
- `lightning_module.py`: Lightning training module
- `trainer.py`: GRPO trainer

## Core Features

✅ In-group relative advantage normalization  
✅ Multiple loss functions (DAPO, Dr.GRPO, BNPO)  
✅ RWKV state cache optimization  
✅ Flexible reward function system  
✅ Support for standard and infctx modes  
✅ Lightning + DeepSpeed integration  

## Detailed Documentation

- [Usage Guide](../../../docs/GRPO_Usage_Guide.md)
- [Design Document](../../../docs/GRPO_Design_Document.md)
- [Implementation Summary](../../../docs/GRPO_Implementation_Summary.md)

## References

- [DeepSeekMath Paper](https://arxiv.org/abs/2402.03300)
- [TRL Library](https://github.com/huggingface/trl)

