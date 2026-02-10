"""
Tokenizers for RWKV models
"""

import os
from typing import Optional
from rwkvtune.data.tokenizers.rwkv_tokenizer import TRIE_TOKENIZER
from rwkvtune.data.tokenizers.auto_tokenizer import (
    AutoTokenizer,
    RWKVTokenizer,
    TokenizerConfig,
)

__all__ = [
    "TRIE_TOKENIZER",
    "AutoTokenizer",
    "RWKVTokenizer",
    "TokenizerConfig",
    "get_tokenizer",
    "get_vocab_path",
]


def get_vocab_path(vocab_name: str = "rwkv_vocab_v20230424.txt") -> str:
    """
    Get the path to RWKV vocabulary file.
    
    Args:
        vocab_name: Vocabulary file name, default is "rwkv_vocab_v20230424.txt"
    
    Returns:
        Absolute path to vocabulary file
    
    Raises:
        FileNotFoundError: If vocabulary file does not exist
    """
    tokenizers_dir = os.path.dirname(__file__)
    vocab_path = os.path.join(tokenizers_dir, "vocab", vocab_name)
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            f"Please ensure vocabulary file is at: {os.path.join(tokenizers_dir, 'vocab')}"
        )
    
    return vocab_path


def get_tokenizer(
    vocab_path: Optional[str] = None, 
    vocab_name: str = "rwkv_vocab_v20230424.txt",
    eos_token: Optional[str] = None,
) -> TRIE_TOKENIZER:
    """
    Get RWKV tokenizer instance.
    
    Convenience function for loading RWKV's TRIE_TOKENIZER.
    If vocab_path is not specified, uses the built-in default vocabulary.
    
    Args:
        vocab_path: Full path to vocabulary file. If None, uses default vocabulary.
        vocab_name: Vocabulary file name when vocab_path is None.
        eos_token: EOS (End-of-Sequence) token, default is double newline.
                  Can be a string or token ID (int).
    
    Returns:
        TRIE_TOKENIZER instance
    
    Raises:
        FileNotFoundError: If vocabulary file does not exist
    """
    if vocab_path is None:
        vocab_path = get_vocab_path(vocab_name)
    
    return TRIE_TOKENIZER(vocab_path, eos_token=eos_token)

