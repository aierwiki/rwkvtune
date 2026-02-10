"""
LoRA Configuration - inspired by peft-main design
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


# RWKV7 default target modules (based on RWKV-PEFT)
RWKV7_DEFAULT_TARGET_MODULES = [
    # Attention modules
    "receptance",
    "key", 
    "value",
    "output",
    # FFN modules
    "key_ffn",
    "value_ffn",
]


@dataclass
class LoraConfig:
    """
    LoRA Configuration - inspired by peft-main design
    
    Example:
        ```python
        from rwkvtune.peft import LoraConfig
        
        # Basic configuration
        config = LoraConfig(
            r=64,
            lora_alpha=128,
        )
        
        # Custom target modules
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["receptance", "key", "value", "output"],
            lora_dropout=0.05,
        )
        ```
    
    Attributes:
        r: LoRA rank (rank of low-rank matrices)
        lora_alpha: LoRA alpha (scaling factor = alpha / r)
        lora_dropout: LoRA dropout probability
        target_modules: Names of modules to apply LoRA to
        modules_to_save: Additional modules to train besides LoRA
        use_rslora: Whether to use Rank-Stabilized LoRA
        init_lora_weights: LoRA weight initialization method
    """
    
    r: int = 64
    """LoRA rank (rank of low-rank matrices)"""
    
    lora_alpha: int = 128
    """LoRA alpha (scaling factor = alpha / r, or alpha / sqrt(r) if using RSLoRA)"""
    
    lora_dropout: float = 0.0
    """LoRA dropout probability"""
    
    target_modules: Optional[Union[List[str], str]] = None
    """
    Names of modules to apply LoRA to.
    - None: Use RWKV7 default configuration (all Linear layers in att + ffn)
    - "att": Apply only to attention modules
    - "ffn": Apply only to FFN modules
    - ["receptance", "key", ...]: Specify exact module names
    """
    
    modules_to_save: Optional[List[str]] = None
    """
    Additional modules to train besides LoRA (e.g., ln, emb, head)
    Default: ["ln", "time"] (consistent with RWKV-PEFT)
    """
    
    use_rslora: bool = False
    """Use Rank-Stabilized LoRA (scaling factor = alpha / sqrt(r))"""
    
    init_lora_weights: Union[bool, str] = True
    """
    LoRA weight initialization method:
    - True: Kaiming uniform for A, zeros for B (default)
    - "gaussian": Gaussian for A, zeros for B
    - False: No initialization (for loading pretrained weights)
    """
    
    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"`r` must be a positive integer, got {self.r}")
        
        if self.target_modules is None:
            self.target_modules = RWKV7_DEFAULT_TARGET_MODULES.copy()
        elif isinstance(self.target_modules, str):
            if self.target_modules == "att":
                self.target_modules = ["receptance", "key", "value", "output"]
            elif self.target_modules == "ffn":
                self.target_modules = ["key_ffn", "value_ffn"]
            else:
                self.target_modules = [self.target_modules]
        
        if self.modules_to_save is None:
            self.modules_to_save = ["ln", "time"]
    
    @property
    def scaling(self) -> float:
        """Calculate LoRA scaling factor"""
        if self.use_rslora:
            import math
            return self.lora_alpha / math.sqrt(self.r)
        else:
            return self.lora_alpha / self.r
    
    def __repr__(self) -> str:
        return (
            f"LoraConfig(\n"
            f"  r={self.r},\n"
            f"  lora_alpha={self.lora_alpha},\n"
            f"  lora_dropout={self.lora_dropout},\n"
            f"  target_modules={self.target_modules},\n"
            f"  modules_to_save={self.modules_to_save},\n"
            f"  use_rslora={self.use_rslora},\n"
            f"  scaling={self.scaling:.4f}\n"
            f")"
        )
