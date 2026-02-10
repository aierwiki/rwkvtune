"""
RWKVTune PEFT (Parameter-Efficient Fine-Tuning) Module

Provides LoRA and other parameter-efficient fine-tuning methods,
inspired by HuggingFace PEFT and RWKV-PEFT designs.

Example:
    ```python
    from rwkvtune import AutoModel
    from rwkvtune.peft import LoraConfig, get_peft_model
    
    model = AutoModel.from_pretrained("path/to/model")
    
    lora_config = LoraConfig(r=64, lora_alpha=128)
    
    model = get_peft_model(model, lora_config)
    # Now only LoRA parameters are trainable
    ```
"""

from .config import LoraConfig, RWKV7_DEFAULT_TARGET_MODULES
from .lora import LoraLinear
from .peft_model import get_peft_model, is_peft_model, load_peft_model

__all__ = [
    "LoraConfig",
    "LoraLinear", 
    "get_peft_model",
    "is_peft_model",
    "load_peft_model",
    "RWKV7_DEFAULT_TARGET_MODULES",
]
