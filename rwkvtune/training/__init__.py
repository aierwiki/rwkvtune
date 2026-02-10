"""
Training utilities and trainers

Provides three trainers (following trl-main design):
1. PretrainTrainer - Pre-training
2. SFTTrainer - Supervised Fine-Tuning
3. GRPOTrainer - GRPO Reinforcement Learning

All trainers follow a unified interface:
- Accept Config and HuggingFace Dataset
- User handles data preprocessing
- Automatically load model architecture from model_config

Example:
    ```python
    from rwkvtune.training import PretrainConfig, PretrainTrainer
    from datasets import Dataset
    
    dataset = Dataset.from_list([...])
    config = PretrainConfig(model_config="rwkv7-0.1b", ...)
    trainer = PretrainTrainer(config=config, train_dataset=dataset)
    trainer.train()
    ```
"""

from rwkvtune.training.pretrain_config import PretrainConfig
from rwkvtune.training.pretrain_trainer import PretrainTrainer
from rwkvtune.training.sft_config import SFTConfig
from rwkvtune.training.sft_trainer import SFTTrainer
from rwkvtune.training.grpo_config import GRPOConfig
from rwkvtune.training.grpo_trainer import GRPOTrainer

__all__ = [
    "PretrainConfig",
    "PretrainTrainer",
    "SFTConfig", 
    "SFTTrainer",
    "GRPOConfig",
    "GRPOTrainer",
]

