"""
Structured Output Module - JSON Schema Constrained Generation

Uses lm-format-enforcer library to implement JSON Schema constraints,
masking tokens that don't conform to the schema during sampling.

Supports:
- JSON Schema constraints (OpenAI compatible response_format)
- JSON Object mode (force valid JSON output)

Usage:
    from rwkvtune.inference.structured_output import JsonSchemaEnforcer
    
    # Create enforcer
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    enforcer = JsonSchemaEnforcer(tokenizer, schema)
    
    # Use in sampling loop
    logits = enforcer.mask_logits(logits)
    token = sample(logits)
    is_done = enforcer.update(token)
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Union
import json

# Try to import lm-format-enforcer
try:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer, CharacterLevelParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
    )
    LM_FORMAT_ENFORCER_AVAILABLE = True
except ImportError:
    LM_FORMAT_ENFORCER_AVAILABLE = False


class TokenizerAdapter:
    """
    Tokenizer adapter - adds lm-format-enforcer required attributes to RWKV Tokenizer
    
    lm-format-enforcer requires tokenizer to have:
    - all_special_ids: List of special token IDs
    - vocab_size: Vocabulary size
    - convert_ids_to_tokens: Convert IDs to token strings
    
    RWKV tokenizer special handling:
    - Tokens 0-255 are usually byte-level tokens, many decode to invalid characters
    - Need to add these invalid tokens to all_special_ids to exclude them
    """
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        
        # Get vocab size
        if hasattr(tokenizer, 'vocab_size'):
            vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, 'get_vocab'):
            vocab_size = len(tokenizer.get_vocab())
        else:
            vocab_size = 65536
        self._vocab_size = vocab_size
        
        # Build all_special_ids (includes real special tokens and invalid byte tokens)
        special_ids = set()
        
        # Add standard special tokens
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            special_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            special_ids.add(tokenizer.bos_token_id)
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            special_ids.add(tokenizer.pad_token_id)
        if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
            special_ids.add(tokenizer.unk_token_id)
        
        # Add tokens that decode to invalid characters
        for tid in range(min(512, vocab_size)):
            try:
                decoded = tokenizer.decode([tid])
                if not decoded:
                    special_ids.add(tid)
                elif decoded == '\ufffd' or (len(decoded) == 1 and ord(decoded) < 32 and decoded not in '\n\r\t'):
                    special_ids.add(tid)
            except:
                special_ids.add(tid)
        
        self.all_special_ids = list(special_ids)
    
    def __getattr__(self, name):
        """Proxy other attributes to original tokenizer"""
        return getattr(self._tokenizer, name)
    
    def convert_ids_to_tokens(self, ids):
        """Convert ID list to token string list"""
        if hasattr(self._tokenizer, 'convert_ids_to_tokens'):
            return self._tokenizer.convert_ids_to_tokens(ids)
        
        if isinstance(ids, int):
            return self._tokenizer.decode([ids])
        return [self._tokenizer.decode([i]) for i in ids]
    
    @property
    def vocab_size(self):
        """Get vocab size"""
        return self._vocab_size
    
    def __len__(self):
        """Return vocab size (required by lm-format-enforcer)"""
        return self.vocab_size
    
    def get_vocab(self):
        """Get vocabulary (required by lm-format-enforcer)"""
        if hasattr(self._tokenizer, 'get_vocab'):
            return self._tokenizer.get_vocab()
        return {self._tokenizer.decode([i]): i for i in range(min(self.vocab_size, 1000))}


class JsonSchemaEnforcer:
    """
    JSON Schema Constraint Sampler
    
    Uses lm-format-enforcer library to force model output to conform to JSON Schema.
    
    How it works:
    1. Parse JSON Schema, build state machine
    2. Before each sample, compute allowed tokens based on current state
    3. Set disallowed token logits to -inf
    4. Update state machine after sampling
    
    Args:
        tokenizer: HuggingFace style tokenizer
        schema: JSON Schema dictionary
        
    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "tags": {
        ...             "type": "array",
        ...             "items": {"type": "string"},
        ...             "minItems": 5,
        ...             "maxItems": 5
        ...         }
        ...     },
        ...     "required": ["tags"]
        ... }
        >>> enforcer = JsonSchemaEnforcer(tokenizer, schema)
        >>> # In generation loop
        >>> logits = enforcer.mask_logits(logits)
        >>> token = sample(logits)
        >>> is_done = enforcer.update(token)
    """
    
    def __init__(self, tokenizer, schema: Dict[str, Any]):
        """
        Initialize JSON Schema Enforcer
        
        Args:
            tokenizer: Tokenizer object (must support encode/decode)
            schema: JSON Schema dictionary
        """
        if not LM_FORMAT_ENFORCER_AVAILABLE:
            raise ImportError(
                "lm-format-enforcer is required for JSON Schema enforcement. "
                "Install it with: pip install lm-format-enforcer"
            )
        
        self.tokenizer = tokenizer
        self.schema = schema
        
        # Wrap tokenizer with adapter
        self.adapted_tokenizer = TokenizerAdapter(tokenizer)
        
        # Build tokenizer data (required by lm-format-enforcer)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.adapted_tokenizer)
        
        # Create JSON Schema parser
        self.parser = JsonSchemaParser(schema)
        
        # Create Token Enforcer
        self.enforcer = TokenEnforcer(self.tokenizer_data, self.parser)
        
        # Generated token sequence (required by new API)
        self._generated_tokens: List[int] = []
        self._is_finished = False
        
        # Cache invalid tokens set
        self._invalid_tokens: Optional[set] = None
    
    def _get_invalid_tokens(self) -> set:
        """Get invalid token set (lazy load)"""
        if self._invalid_tokens is None:
            self._invalid_tokens = set()
            vocab_size = self.adapted_tokenizer.vocab_size
            for tid in range(vocab_size):
                try:
                    decoded = self.tokenizer.decode([tid])
                    if decoded == '\ufffd':
                        self._invalid_tokens.add(tid)
                except:
                    self._invalid_tokens.add(tid)
        return self._invalid_tokens
    
    def get_allowed_tokens(self) -> List[int]:
        """
        Get allowed token IDs for current state
        
        Returns:
            Allowed token ID list (invalid tokens filtered)
        """
        token_list = self.enforcer.get_allowed_tokens(self._generated_tokens)
        allowed = token_list.allowed_tokens if token_list.allowed_tokens else []
        
        invalid = self._get_invalid_tokens()
        return [tid for tid in allowed if tid not in invalid]
    
    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask disallowed token logits
        
        Sets logits of tokens not conforming to current JSON Schema state to -inf,
        making their probability 0 after softmax.
        
        Args:
            logits: [vocab_size] or [batch_size, vocab_size] logits tensor
            
        Returns:
            Masked logits tensor (same shape as input)
        """
        if self._is_finished:
            return logits
        
        allowed_tokens = self.get_allowed_tokens()
        
        if not allowed_tokens:
            self._is_finished = True
            return logits
        
        if logits.dim() == 1:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_tokens] = 0
            return logits + mask
        elif logits.dim() == 2:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, allowed_tokens] = 0
            return logits + mask
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
    
    def update(self, token_id: int) -> bool:
        """
        Update state machine (call after sampling a token)
        
        Args:
            token_id: Sampled token ID
            
        Returns:
            Whether finished (True = JSON is complete, can stop generation)
        """
        self._generated_tokens.append(token_id)
        
        try:
            allowed = self.get_allowed_tokens()
            if not allowed:
                self._is_finished = True
            if hasattr(self.enforcer, 'eos_token_id') and token_id == self.enforcer.eos_token_id:
                self._is_finished = True
        except Exception as e:
            print(f"Warning: JsonSchemaEnforcer update error: {e}")
            self._is_finished = True
        
        return self._is_finished
    
    def is_finished(self) -> bool:
        """Check if finished"""
        return self._is_finished
    
    def reset(self):
        """Reset state machine (for new generation)"""
        self._generated_tokens = []
        self._is_finished = False


class JsonObjectEnforcer:
    """
    JSON Object Constraint Sampler (simplified version)
    
    Forces model to output valid JSON object, but doesn't restrict specific structure.
    More lenient than JsonSchemaEnforcer, only ensures output is valid JSON.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize JSON Object Enforcer
        
        Args:
            tokenizer: Tokenizer object
        """
        if not LM_FORMAT_ENFORCER_AVAILABLE:
            raise ImportError(
                "lm-format-enforcer is required for JSON enforcement. "
                "Install it with: pip install lm-format-enforcer"
            )
        
        self.tokenizer = tokenizer
        
        self.adapted_tokenizer = TokenizerAdapter(tokenizer)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.adapted_tokenizer)
        
        # Use generic JSON Schema (allows any JSON object)
        self.schema = {"type": "object"}
        self.parser = JsonSchemaParser(self.schema)
        self.enforcer = TokenEnforcer(self.tokenizer_data, self.parser)
        
        self._generated_tokens: List[int] = []
        self._is_finished = False
        self._invalid_tokens: Optional[set] = None
    
    def _get_invalid_tokens(self) -> set:
        """Get invalid token set (lazy load)"""
        if self._invalid_tokens is None:
            self._invalid_tokens = set()
            vocab_size = self.adapted_tokenizer.vocab_size
            for tid in range(vocab_size):
                try:
                    decoded = self.tokenizer.decode([tid])
                    if decoded == '\ufffd':
                        self._invalid_tokens.add(tid)
                except:
                    self._invalid_tokens.add(tid)
        return self._invalid_tokens
    
    def get_allowed_tokens(self) -> List[int]:
        """Get allowed token IDs for current state (invalid tokens filtered)"""
        token_list = self.enforcer.get_allowed_tokens(self._generated_tokens)
        allowed = token_list.allowed_tokens if token_list.allowed_tokens else []
        invalid = self._get_invalid_tokens()
        return [tid for tid in allowed if tid not in invalid]
    
    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask disallowed token logits"""
        if self._is_finished:
            return logits
        
        allowed_tokens = self.get_allowed_tokens()
        
        if not allowed_tokens:
            self._is_finished = True
            return logits
        
        if logits.dim() == 1:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_tokens] = 0
            return logits + mask
        elif logits.dim() == 2:
            mask = torch.full_like(logits, float('-inf'))
            mask[:, allowed_tokens] = 0
            return logits + mask
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
    
    def update(self, token_id: int) -> bool:
        """Update state machine"""
        self._generated_tokens.append(token_id)
        
        try:
            allowed = self.get_allowed_tokens()
            if not allowed:
                self._is_finished = True
            if hasattr(self.enforcer, 'eos_token_id') and token_id == self.enforcer.eos_token_id:
                self._is_finished = True
        except Exception as e:
            print(f"Warning: JsonObjectEnforcer update error: {e}")
            self._is_finished = True
        
        return self._is_finished
    
    def is_finished(self) -> bool:
        """Check if finished"""
        return self._is_finished
    
    def reset(self):
        """Reset state machine"""
        self._generated_tokens = []
        self._is_finished = False


def create_enforcer(
    tokenizer,
    response_format: Optional[Dict[str, Any]] = None,
) -> Optional[Union[JsonSchemaEnforcer, JsonObjectEnforcer]]:
    """
    Create appropriate enforcer based on response_format
    
    Args:
        tokenizer: Tokenizer object
        response_format: OpenAI style response_format config
            - {"type": "text"}: No constraint (returns None)
            - {"type": "json_object"}: Use JsonObjectEnforcer
            - {"type": "json_schema", "json_schema": {...}}: Use JsonSchemaEnforcer
    
    Returns:
        Appropriate enforcer object, or None (if no constraint needed)
    
    Example:
        >>> response_format = {
        ...     "type": "json_schema",
        ...     "json_schema": {
        ...         "name": "tags",
        ...         "schema": {"type": "object", "properties": {"tags": {"type": "array"}}}
        ...     }
        ... }
        >>> enforcer = create_enforcer(tokenizer, response_format)
    """
    if response_format is None:
        return None
    
    format_type = response_format.get("type", "text")
    
    if format_type == "text":
        return None
    
    elif format_type == "json_object":
        return JsonObjectEnforcer(tokenizer)
    
    elif format_type == "json_schema":
        json_schema_config = response_format.get("json_schema", {})
        schema = json_schema_config.get("schema") or json_schema_config.get("schema_", {})
        if not schema:
            raise ValueError("json_schema.schema is required when type is 'json_schema'")
        return JsonSchemaEnforcer(tokenizer, schema)
    
    else:
        raise ValueError(f"Unsupported response_format type: {format_type}")


def is_structured_output_available() -> bool:
    """Check if structured output feature is available"""
    return LM_FORMAT_ENFORCER_AVAILABLE
