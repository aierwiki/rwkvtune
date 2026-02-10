"""
Special Token Handler - Following Transformers implementation

Implements Transformers-like special token handling:
1. Uses Trie for priority matching of special tokens (ensures no splitting)
2. During encoding: prioritizes special token mapping
3. During decoding: prioritizes special token mapping
"""

from typing import Dict, List, Optional, Tuple, Union


class SpecialTokenTrie:
    """
    Trie for matching special tokens (text-level, not byte-level)
    
    Following Transformers Trie implementation, used to split text before tokenization,
    ensuring special tokens are not split.
    """
    
    def __init__(self):
        self.data = {}
        self._tokens = set()
        self._termination_char = ""
    
    def add(self, word: str):
        """Add a special token to Trie"""
        if not word:
            return
        
        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = ref.setdefault(char, {})
            ref = ref[char]
        ref[self._termination_char] = 1
    
    def split(self, text: str) -> List[str]:
        """
        Split text, prioritizing special token matches
        
        Following Transformers Trie.split() implementation.
        Returns split text list, with special tokens separated.
        
        Example:
            >>> trie = SpecialTokenTrie()
            >>> trie.add("<|im_start|>")
            >>> trie.add("<|im_end|>")
            >>> trie.split("Hello<|im_start|>user<|im_end|>")
            ["Hello", "<|im_start|>", "user", "<|im_end|>"]
        """
        if not text:
            return []
        
        states = []
        states.append({0: self.data})
        
        matches = []
        
        idx = 0
        while idx < len(text):
            char = text[idx]
            new_states = {}
            
            for start_idx, trie_state in states[-1].items():
                if char in trie_state:
                    new_trie_state = trie_state[char]
                    if self._termination_char in new_trie_state:
                        matches.append((start_idx, idx + 1, text[start_idx:idx + 1]))
                    if start_idx not in new_states:
                        new_states[start_idx] = new_trie_state
                if 0 not in new_states:
                    new_states[0] = self.data
                    if char in self.data:
                        new_states[0] = self.data[char]
                        if self._termination_char in new_states[0]:
                            matches.append((idx, idx + 1, text[idx:idx + 1]))
            
            states.append(new_states)
            idx += 1
        
        if not matches:
            return [text]
        
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
        filtered_matches = []
        for match in matches:
            start, end, token = match
            overlap = False
            for existing_start, existing_end, _ in filtered_matches:
                if not (end <= existing_start or start >= existing_end):
                    if (end - start) > (existing_end - existing_start):
                        filtered_matches.remove((existing_start, existing_end, _))
                        filtered_matches.append((start, end, token))
                    overlap = True
                    break
            if not overlap:
                filtered_matches.append((start, end, token))
        
        filtered_matches.sort(key=lambda x: x[0])
        
        result = []
        last_end = 0
        
        for start, end, token in filtered_matches:
            if start > last_end:
                result.append(text[last_end:start])
            result.append(token)
            last_end = end
        
        if last_end < len(text):
            result.append(text[last_end:])
        
        return result if result else [text]


def encode_with_special_tokens(
    text: str,
    tokenizer,
    additional_special_tokens: Dict[str, Union[int, List[int]]]
) -> List[int]:
    """
    Encode text, prioritizing special tokens
    
    Following Transformers implementation:
    1. Use Trie to split text, prioritizing special token matches
    2. For matched special tokens, use mapped ID (or sequence)
    3. For normal text, use normal encoding
    
    Args:
        text: Text to encode
        tokenizer: RWKV tokenizer instance
        additional_special_tokens: Special token mapping {token_string: token_id or token_sequence}
    
    Returns:
        Encoded token ID list
    """
    if not additional_special_tokens:
        return tokenizer.encode(text)
    
    trie = SpecialTokenTrie()
    for token_str in additional_special_tokens.keys():
        trie.add(token_str)
    
    segments = trie.split(text)
    
    result = []
    for segment in segments:
        if segment in additional_special_tokens:
            token_id_or_seq = additional_special_tokens[segment]
            if isinstance(token_id_or_seq, list):
                result.extend(token_id_or_seq)
            else:
                result.append(token_id_or_seq)
        else:
            result.extend(tokenizer.encode(segment))
    
    return result


def decode_with_special_tokens(
    token_ids: List[int],
    tokenizer,
    additional_special_tokens: Dict[str, Union[int, List[int]]],
    reverse_mapping: Optional[Dict[Tuple[int, ...], str]] = None
) -> str:
    """
    Decode token IDs, prioritizing special tokens
    
    Following Transformers implementation:
    1. Check if it's a special token sequence
    2. If yes, use corresponding token string directly
    3. Otherwise, decode normally
    
    Args:
        token_ids: Token ID list
        tokenizer: RWKV tokenizer instance
        additional_special_tokens: Special token mapping {token_string: token_id or token_sequence}
        reverse_mapping: Reverse mapping {(token_id,): token_string} or {(id1, id2, ...): token_string}
                        If None, will be built automatically
    
    Returns:
        Decoded text
    """
    if not additional_special_tokens:
        return tokenizer.decode(token_ids)
    
    if reverse_mapping is None:
        reverse_mapping = {}
        for token_str, token_id_or_seq in additional_special_tokens.items():
            if isinstance(token_id_or_seq, list):
                reverse_mapping[tuple(token_id_or_seq)] = token_str
            else:
                reverse_mapping[(token_id_or_seq,)] = token_str
    
    result = []
    i = 0
    
    while i < len(token_ids):
        matched = False
        
        for seq_len in range(min(10, len(token_ids) - i), 0, -1):
            seq = tuple(token_ids[i:i + seq_len])
            if seq in reverse_mapping:
                result.append(reverse_mapping[seq])
                i += seq_len
                matched = True
                break
        
        if not matched:
            decoded_char = tokenizer.decode([token_ids[i]])
            result.append(decoded_char)
            i += 1
    
    return "".join(result)
