"""
Output Type Definitions - vLLM-style API

Provides vLLM RequestOutput-like interface for returning generation results.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CompletionOutput:
    """
    Single completion output result
    
    Attributes:
        index: Output index (same prompt may generate multiple completions)
        text: Generated text
        token_ids: Generated token ID list
        cumulative_logprob: Cumulative log probability (optional)
        finish_reason: Finish reason ("stop", "length", or None)
    """
    index: int
    text: str
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    
    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text[:50]!r}{'...' if len(self.text) > 50 else ''}, "
            f"finish_reason={self.finish_reason!r})"
        )


@dataclass
class RequestOutput:
    """
    Complete output result for a single request
    
    Attributes:
        request_id: Request ID
        prompt: Original prompt text
        prompt_token_ids: Token ID list for prompt
        outputs: List of generated completions
        finished: Whether generation is complete
    
    Example:
        >>> output = llm.generate("Hello")[0]
        >>> print(output.prompt)
        "Hello"
        >>> print(output.outputs[0].text)
        " world! How are you?"
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int] = field(default_factory=list)
    outputs: List[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    
    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt[:30]!r}{'...' if len(self.prompt) > 30 else ''}, "
            f"outputs={len(self.outputs)} completions)"
        )
