"""
RWKVTune: RWKV Model Training Toolkit

This package provides comprehensive training capabilities for RWKV models:
- PretrainTrainer: Continue pre-training from existing models
- SFTTrainer: Supervised Fine-Tuning for instruction following
- GRPOTrainer: GRPO (Group Relative Policy Optimization) for RLHF

Key Features:
- Multi-GPU training with DeepSpeed ZeRO optimization
- LoRA and other PEFT methods for efficient fine-tuning
- Infinite context training support
- HuggingFace Datasets integration
- PyTorch Lightning based training loop

Philosophy: Simple, Clean, and Powerful Training for RWKV Models
"""

__version__ = "0.1.0"
__author__ = "RWKV-Tune Contributors"
__license__ = "Apache-2.0"

# Core model imports
from rwkvtune.models import (
    RWKV7Model,
    RWKV7Config,
    RWKVConfig,
    GenerationConfig,
    AutoModel,
)

# Tokenizer imports
from rwkvtune.data.tokenizers import (
    get_tokenizer,
    TRIE_TOKENIZER,
    AutoTokenizer,
    RWKVTokenizer,
)

# Training imports
from rwkvtune.training import (
    PretrainConfig,
    PretrainTrainer,
    SFTConfig,
    SFTTrainer,
    GRPOConfig,
    GRPOTrainer,
)

# PEFT imports
from rwkvtune.peft import (
    LoraConfig,
    get_peft_model,
    is_peft_model,
    load_peft_model,
    RWKV7_DEFAULT_TARGET_MODULES,
)

# Utility imports
from rwkvtune.utils import create_model_hub

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Models
    "RWKV7Model",
    "RWKV7Config",
    "RWKVConfig",
    "GenerationConfig",
    "AutoModel",
    # Tokenizers
    "get_tokenizer",
    "TRIE_TOKENIZER",
    "AutoTokenizer",
    "RWKVTokenizer",
    # Training
    "PretrainConfig",
    "PretrainTrainer",
    "SFTConfig",
    "SFTTrainer",
    "GRPOConfig",
    "GRPOTrainer",
    # PEFT
    "LoraConfig",
    "get_peft_model",
    "is_peft_model",
    "load_peft_model",
    "RWKV7_DEFAULT_TARGET_MODULES",
    # Utilities
    "create_model_hub",
]
