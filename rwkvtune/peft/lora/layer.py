"""
LoRA Linear Layer - inspired by RWKV-PEFT and peft-main design

Formula: y = Wx + (BA)x * scaling
where A in R^{r x in}, B in R^{out x r}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoraLinear(nn.Module):
    """
    LoRA Linear Layer
    
    Applies LoRA adapter to an existing nn.Linear layer.
    
    Args:
        base_layer: Base Linear layer
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: LoRA dropout probability
        use_rslora: Whether to use Rank-Stabilized LoRA
        init_lora_weights: Whether to initialize LoRA weights
    
    Example:
        ```python
        base = nn.Linear(256, 512, bias=False)
        lora = LoraLinear(base, r=64, lora_alpha=128)
        
        x = torch.randn(2, 10, 256)
        y = lora(x)  # [2, 10, 512]
        ```
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        self.r = r
        self.lora_alpha = lora_alpha
        
        if use_rslora:
            self.scaling = lora_alpha / math.sqrt(r)
        else:
            self.scaling = lora_alpha / r
        
        # LoRA parameters: A in R^{r x in}, B in R^{out x r}
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        if init_lora_weights:
            self.reset_lora_parameters()
        
        self.merged = False
        self.disabled = False
    
    def reset_lora_parameters(self):
        """
        Reset LoRA parameters
        
        Initialization strategy (consistent with peft-main):
        - A: Kaiming uniform initialization
        - B: Zero initialization
        
        This ensures LoRA output is 0 initially, not affecting original model behavior.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        result = self.base_layer(x)
        
        if self.disabled or self.merged:
            return result
        
        if x.device != self.lora_A.device or x.dtype != self.lora_A.dtype:
            x = x.to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        # LoRA delta: (x @ A^T) @ B^T * scaling
        try:
            lora_out = F.linear(
                F.linear(self.lora_dropout(x), self.lora_A),
                self.lora_B
            )
        except RuntimeError as e:
            print(f"LoRA Forward Error:")
            print(f"  x: shape={x.shape}, dtype={x.dtype}, device={x.device}, contiguous={x.is_contiguous()}")
            print(f"  lora_A: shape={self.lora_A.shape}, dtype={self.lora_A.dtype}, device={self.lora_A.device}")
            print(f"  lora_B: shape={self.lora_B.shape}, dtype={self.lora_B.dtype}, device={self.lora_B.device}")
            print(f"  result: shape={result.shape}, dtype={result.dtype}, device={result.device}")
            raise e
        
        if lora_out.device != result.device or lora_out.dtype != result.dtype:
            lora_out = lora_out.to(device=result.device, dtype=result.dtype)
        
        return result + lora_out * self.scaling
    
    def merge(self):
        """
        Merge LoRA weights into base layer
        
        After merging, forward pass only requires one matrix multiplication,
        but LoRA weights can no longer be adjusted separately.
        """
        if self.merged:
            return
        
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
        
        self.merged = True
    
    def unmerge(self):
        """
        Remove LoRA weights from base layer
        
        Restores to pre-merge state.
        """
        if not self.merged:
            return
        
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
        
        self.merged = False
    
    def disable_adapter(self):
        """
        Disable adapter
        
        When disabled, forward pass only uses base layer without LoRA delta.
        Useful for getting reference model outputs in GRPO.
        """
        self.disabled = True
    
    def enable_adapter(self):
        """Enable adapter"""
        self.disabled = False
    
    def get_lora_state_dict(self) -> dict:
        """Get LoRA parameters state_dict"""
        return {
            'lora_A': self.lora_A.data.clone(),
            'lora_B': self.lora_B.data.clone(),
        }
    
    def load_lora_state_dict(self, state_dict: dict):
        """Load LoRA parameters"""
        if 'lora_A' in state_dict:
            self.lora_A.data.copy_(state_dict['lora_A'])
        if 'lora_B' in state_dict:
            self.lora_B.data.copy_(state_dict['lora_B'])
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.r}, "
            f"lora_alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"merged={self.merged}, "
            f"disabled={self.disabled}"
        )
