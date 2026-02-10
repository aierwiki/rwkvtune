"""
Sampling Parameters Definition - vLLM-style API

Provides vLLM SamplingParams-like interface for controlling text generation behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class SamplingParams:
    """
    Sampling parameters class - Controls text generation behavior
    
    Similar to vLLM's SamplingParams, but simplified for RWKV models.
    
    Args:
        max_tokens: Maximum number of tokens to generate, default 256
        temperature: Sampling temperature, higher is more random, lower is more deterministic. 0 means greedy sampling
        top_p: Nucleus sampling parameter, keeps tokens with cumulative probability >= top_p
        top_k: Top-k sampling parameter, 0 means no limit
        stop: Stop word list (string form)
        stop_token_ids: Stop token ID list
        skip_special_tokens: Whether to skip special tokens during decoding
        repetition_penalty: Repetition penalty coefficient (not yet implemented)
        logit_bias: token ID -> bias value mapping, -100 means prohibit generating that token
    
    Example:
        >>> params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
        >>> # Prohibit generating token 65530
        >>> params = SamplingParams(logit_bias={"65530": -100})
        >>> llm.generate(prompts, params)
    """
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    skip_special_tokens: bool = True
    repetition_penalty: float = 1.0
    logit_bias: Optional[dict] = None  # token ID -> bias value mapping, -100 means prohibit
    
    def __post_init__(self):
        """Parameter validation"""
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got: {self.temperature}")
        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be in [0, 1] range, got: {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got: {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got: {self.max_tokens}")
        
        # Initialize defaults
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []
    
    @classmethod
    def from_dict(cls, d: dict) -> "SamplingParams":
        """Create SamplingParams from dictionary"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            "stop_token_ids": self.stop_token_ids,
            "skip_special_tokens": self.skip_special_tokens,
            "repetition_penalty": self.repetition_penalty,
            "logit_bias": self.logit_bias,
        }
