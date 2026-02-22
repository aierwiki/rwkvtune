########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name, eos_token=None):
        self.idx2token = {}
        sorted = []
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # Vocab size: use max(token_id) + 1, not len(idx2token)
        self.vocab_size = max(self.idx2token.keys()) + 1 if self.idx2token else 0

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))
        
        # Set EOS token (default is double newline)
        if eos_token is None:
            eos_token = '\n\n'
        
        if isinstance(eos_token, str):
            eos_tokens = self.encode(eos_token)
            self.eos_token_id = eos_tokens[0] if eos_tokens else None
        elif isinstance(eos_token, int):
            self.eos_token_id = eos_token if eos_token in self.idx2token else None
        else:
            self.eos_token_id = None

    def __len__(self):
        """Return vocab size, supports len(tokenizer)"""
        return self.vocab_size

    def __getstate__(self):
        """Serialize only necessary data, avoiding TRIE tree circular reference issues."""
        return {
            'idx2token': self.idx2token,
            'token2idx': self.token2idx,
            'vocab_size': self.vocab_size,
            'eos_token_id': self.eos_token_id,
        }
    
    def __setstate__(self, state):
        """Rebuild TRIE tree during deserialization."""
        self.idx2token = state['idx2token']
        self.token2idx = state['token2idx']
        self.vocab_size = state['vocab_size']
        self.eos_token_id = state.get('eos_token_id', None)
        
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        """
        Decode token IDs to text
        
        Args:
            tokens: token IDs (list, tuple, torch.Tensor, numpy.ndarray, etc.)
        
        Returns:
            Decoded string
        """
        import os
        debug_decode = os.environ.get("RWKV_DEBUG_DECODE", "0") == "1"
        
        # Support torch.Tensor and numpy.ndarray
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        elif not isinstance(tokens, (list, tuple)):
            tokens = list(tokens)
        
        if debug_decode and len(tokens) > 0:
            invalid_tokens = [t for t in tokens if t not in self.idx2token]
            if invalid_tokens:
                print(f"[DEBUG DECODE] Invalid token IDs: {invalid_tokens[:10]}")
            print(f"[DEBUG DECODE] Decoding {len(tokens)} tokens, last 5: {tokens[-5:] if len(tokens) >= 5 else tokens}")
        
        # Tolerate unknown token IDs and invalid UTF-8 so one bad token does not make whole completion "\ufffd"
        parts = []
        for tid in tokens:
            if tid in self.idx2token:
                parts.append(self.idx2token[tid])
            else:
                parts.append(b'?')
        raw_bytes = b''.join(parts)
        result = raw_bytes.decode('utf-8', errors='replace')
        if debug_decode:
            print(f"[DEBUG DECODE] Decode success, result length: {len(result)}")
        return result

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()
