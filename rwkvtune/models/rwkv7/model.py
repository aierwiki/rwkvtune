"""
RWKV7 Model Definition
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Any, cast, List

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except Exception:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

try:
    from ..generation_mixin import GenerationMixin
except ImportError:
    from rwkvtune.models.generation_mixin import GenerationMixin

try:
    from .operators import rwkv7_wkv, load_wkv7_operator
except ImportError:
    from rwkvtune.models.rwkv7.operators import rwkv7_wkv, load_wkv7_operator


class RWKV7TimeMix(nn.Module):
    """RWKV7 Time Mix layer (attention mechanism)."""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size_a
        self.n_head = config.dim_att // self.head_size
        
        C = config.n_embd
        H = self.n_head
        N = self.head_size
        
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            
            if hasattr(config, 'dim_att_lora') and config.dim_att_lora > 0:
                D_DECAY_LORA = config.dim_att_lora
            else:
                D_DECAY_LORA = max(32, int(round((2.0 * (C ** 0.5)) / 32) * 32))
            
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(self._ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)
            
            D_AAA_LORA = D_DECAY_LORA
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(self._ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)
            
            if layer_id > 0:
                if hasattr(config, 'dim_mv_lora') and config.dim_mv_lora > 0:
                    D_MV_LORA = config.dim_mv_lora
                else:
                    D_MV_LORA = max(32, int(round((1.4 * (C ** 0.5)) / 32) * 32))
                
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(self._ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)
            
            if hasattr(config, 'dim_gate_lora') and config.dim_gate_lora > 0:
                D_GATE_LORA = config.dim_gate_lora
            else:
                D_GATE_LORA = max(32, int(round((5.6 * (C ** 0.5)) / 32) * 32))
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(self._ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))
            
            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        
        with torch.no_grad():
            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.output.weight.data.zero_()
    
    def _ortho_init(self, x, scale):
        """Orthogonal initialization (supports 2D and 3D tensors)."""
        with torch.no_grad():
            shape = x.shape
            if len(shape) == 2:
                gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            else:
                raise ValueError(f"ortho_init only supports 2D or 3D tensors, got {len(shape)}D")
            return x
    
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        
        x_any = rwkv7_wkv(r, w, k, v, -kk, kk * a, self.head_size)
        x = cast(torch.Tensor, x_any)
        
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * 
                 v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        
        return x, v_first
    
    def forward_with_state(self, x, v_first, x_prev, att_kv):
        """
        Forward pass with state (for inference).
        
        Args:
            x: Input [B, T, C]
            v_first: v_first state
            x_prev: Previous x [B, 1, C]
            att_kv: WKV state [B, H, N, N]
        
        Returns:
            y: Output [B, T, C]
            v_first: Updated v_first
            new_x: Current x (as next x_prev)
            new_att_kv: Updated WKV state
        """
        B, T, C = x.size()
        H = self.n_head
        
        def _cast_prev(t: torch.Tensor) -> torch.Tensor:
            return t.to(x.dtype) if t.dtype != x.dtype else t
        
        if T == 1:
            xx = _cast_prev(x_prev).unsqueeze(1) - x
        else:
            x_prev_expanded = _cast_prev(x_prev).unsqueeze(1)
            shifted = torch.cat([x_prev_expanded, x[:, :-1, :]], dim=1)
            xx = shifted - x
        
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        
        r = self.receptance(xr)
        w_lin = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        w = -F.softplus(-(self.w0 + w_lin)) - 0.5

        is_cuda = r.is_cuda
        if is_cuda:
            wkv7s_op = load_wkv7_operator(head_size=self.head_size, device='cuda', for_inference=True)
            
            dtype = r.dtype
            
            r_batch = r.to(dtype).contiguous()
            w_batch = w.to(dtype).contiguous()
            k_batch = k.to(dtype).contiguous()
            v_batch = v.to(dtype).contiguous()
            kk_batch = kk.to(dtype).contiguous()
            a_batch = a.to(dtype).contiguous()

            if att_kv is not None:
                att_kv = att_kv.to(torch.float32).contiguous()
            
            op_any = cast(Any, wkv7s_op)
            y_state = op_any(r_batch, w_batch, k_batch, v_batch, -kk_batch, kk_batch * a_batch, 
                            state=att_kv, return_state=True)
            assert isinstance(y_state, tuple) and len(y_state) == 2
            y = cast(torch.Tensor, y_state[0])
            new_att_kv = y_state[1]
        else:
            result_any = rwkv7_wkv(r, w, k, v, -kk, kk * a, 
                                  self.head_size, state=att_kv, return_state=True)
            assert isinstance(result_any, tuple) and len(result_any) == 2
            y = cast(torch.Tensor, result_any[0])
            new_att_kv = result_any[1]

        y_norm_in = y.view(B * T, C)
        weight = self.ln_x.weight if self.ln_x.weight is not None else None
        bias = self.ln_x.bias if self.ln_x.bias is not None else None
        if weight is not None and weight.dtype != y_norm_in.dtype:
            weight = weight.to(y_norm_in.dtype)
        if bias is not None and bias.dtype != y_norm_in.dtype:
            bias = bias.to(y_norm_in.dtype)
        y = F.group_norm(y_norm_in, num_groups=H, weight=weight, bias=bias, eps=64e-5).view(B, T, C)

        y = y + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * 
                 v.view(B, T, H, -1)).view(B, T, C)
        y = self.output(y * g)
        
        new_x_prev = x[:, -1, :]

        if isinstance(new_x_prev, torch.Tensor) and new_x_prev.dtype != torch.float32:
            new_x_prev = new_x_prev.to(torch.float32)
        if isinstance(new_att_kv, torch.Tensor) and new_att_kv.dtype != torch.float32:
            new_att_kv = new_att_kv.to(torch.float32)
        
        return y, v_first, new_x_prev, new_att_kv


class RWKV7ChannelMix(nn.Module):
    """RWKV7 Channel Mix layer (feed-forward network)."""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        
        self.key = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
        self.value = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)
        
        with torch.no_grad():
            self.key.weight.data.uniform_(-0.5 / (config.n_embd ** 0.5), 0.5 / (config.n_embd ** 0.5))
            self.value.weight.data.zero_()
    
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
    
    def forward_with_state(self, x, x_prev):
        """
        Forward pass with state (for inference).
        
        Args:
            x: Input [B, T, C]
            x_prev: Previous x [B, 1, C]
        
        Returns:
            y: Output [B, T, C]
            new_x: Current x (as next x_prev)
        """
        B, T, C = x.size()
        
        def _cast_prev(t: torch.Tensor) -> torch.Tensor:
            return t.to(x.dtype) if t.dtype != x.dtype else t
        
        if T == 1:
            xx = _cast_prev(x_prev).unsqueeze(1) - x
        else:
            x_prev_expanded = _cast_prev(x_prev).unsqueeze(1)
            shifted = torch.cat([x_prev_expanded, x[:, :-1, :]], dim=1)
            xx = shifted - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        y = self.value(k)
        
        new_x_prev = x[:, -1, :]

        if isinstance(new_x_prev, torch.Tensor) and new_x_prev.dtype != torch.float32:
            new_x_prev = new_x_prev.to(torch.float32)
        
        return y, new_x_prev


class RWKV7Block(nn.Module):
    """RWKV7 Block."""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.skip_ln0 = False
        
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        
        self.att = RWKV7TimeMix(config, layer_id)
        self.ffn = RWKV7ChannelMix(config, layer_id)
    
    @staticmethod
    def _apply_layer_norm(ln_module: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm directly (consistent with official RWKV)."""
        return ln_module(x)
    
    def forward(self, x, v_first):
        if self.layer_id == 0 and not self.skip_ln0:
            x = self._apply_layer_norm(self.ln0, x)

        x_ln1 = self._apply_layer_norm(self.ln1, x)
        x_att, v_first = self.att(x_ln1, v_first)
        x = x + x_att

        x_ln2 = self._apply_layer_norm(self.ln2, x)
        x = x + self.ffn(x_ln2)
        
        return x, v_first
    
    def forward_with_state(self, x, v_first, layer_state):
        """
        Forward pass with state (for inference).
        x_prev stores the value after LayerNorm.
        
        Args:
            x: Input [B, T, C]
            v_first: v_first state
            layer_state: Dict containing att_x_prev, att_kv, ffn_x_prev
        
        Returns:
            x: Output [B, T, C]
            v_first: Updated v_first
            new_layer_state: Updated layer_state dict
        """
        if self.layer_id == 0 and not self.skip_ln0:
            x = self._apply_layer_norm(self.ln0, x)

        x_ln1 = self._apply_layer_norm(self.ln1, x)
        x_att, v_first, new_att_x, new_att_kv = self.att.forward_with_state(
            x_ln1, v_first, 
            layer_state['att_x_prev'],
            layer_state['att_kv']
        )
        x = x + x_att
        
        x_ln2 = self._apply_layer_norm(self.ln2, x)
        x_ffn, new_ffn_x = self.ffn.forward_with_state(x_ln2, layer_state['ffn_x_prev'])
        x = x + x_ffn
        
        new_layer_state = {
            'att_x_prev': new_att_x.to(torch.float32),
            'att_kv': new_att_kv.to(torch.float32) if isinstance(new_att_kv, torch.Tensor) and new_att_kv.dtype != torch.float32 else new_att_kv,
            'ffn_x_prev': new_ffn_x.to(torch.float32)
        }
        
        return x, v_first, new_layer_state


class RWKV7Model(GenerationMixin, nn.Module):
    """RWKV7 Complete Model (supports Transformers-style .generate() method)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKV7Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self._emb_preprocessed = False
    
    def forward(self, idx, logits_to_keep=None):
        """
        Forward pass.
        
        Args:
            idx: Input token IDs [B, T]
            logits_to_keep: Only keep last N tokens' logits (for memory saving)
                           None means return all tokens' logits
        
        Returns:
            logits: [B, T, vocab_size] or [B, logits_to_keep, vocab_size]
        """
        B, T = idx.size()
        assert T <= self.config.ctx_len, f"Sequence length {T} exceeds max length {self.config.ctx_len}"
        
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        
        for block in self.blocks:
            if getattr(self.config, 'grad_cp', 0) == 1 and self.training:
                x, v_first = torch_checkpoint(block, x, v_first, use_reentrant=False)
            else:
                x, v_first = block(x, v_first)
        
        x = self.ln_out(x)
        
        if logits_to_keep is not None and logits_to_keep < T:
            x = x[:, -logits_to_keep:, :]
        
        x = self.head(x)
        
        return x
    
    def forward_with_state(self, idx, states=None):
        """
        Forward pass with state (for inference).
        
        Args:
            idx: Input tokens [B, T]
            states: List of states for all layers, each state [B, H, N, N]
        
        Returns:
            logits: Output logits [B, T, vocab_size]
            new_states: Updated states list
        """
        B, T = idx.size()
        
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        
        if states is None:
            states = self.init_state(B)
        
        new_states = []
        for i, block in enumerate(self.blocks):
            blk = cast(RWKV7Block, block)
            x, v_first, new_layer_state = blk.forward_with_state(x, v_first, states[i])
            new_states.append(new_layer_state)
        
        x = self.ln_out(x)
        x = self.head(x)
        
        return x, new_states
    
    def init_state(self, batch_size):
        """
        Initialize states for all layers.
        Each layer has 3 states:
        - att_x_prev: Attention x_prev [B, C]
        - att_kv: WKV state [B, H, N, N]
        - ffn_x_prev: FFN x_prev [B, C]
        """
        H = self.config.n_embd // 64
        N = 64
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        states = []
        for _ in range(self.config.n_layer):
            layer_state = {
                'att_x_prev': torch.zeros(batch_size, self.config.n_embd, dtype=dtype, device=device),
                'att_kv': torch.zeros(batch_size, H, N, N, dtype=torch.float32, device=device),
                'ffn_x_prev': torch.zeros(batch_size, self.config.n_embd, dtype=dtype, device=device)
            }
            states.append(layer_state)
        return states
    
    def forward_batch_dynamic(self, tokens_list: List[List[int]], states=None, full_output=False):
        """
        Dynamic batch forward pass - process sequences of different lengths.
        
        Core idea:
        1. Maintain processing position for each sequence
        2. Find active (unfinished) sequences
        3. Compute minimum step size for batch processing
        4. Process batch and update states
        5. Repeat until all sequences are done
        
        Args:
            tokens_list: List of token sequences [[tok1, tok2, ...], [tok1, ...], ...]
            states: Initial states, auto-initialized if None
            full_output: Whether to return all positions' output (default: only last)
        
        Returns:
            logits: Output logits
                - full_output=False: List[Tensor] - [B], each [vocab_size]
                - full_output=True: List[Tensor] - [B], each [T_i, vocab_size]
            states: Updated states
        """
        assert isinstance(tokens_list, list), "tokens_list must be a list of token sequences"
        
        batch_size = len(tokens_list)
        lengths = [len(tokens) for tokens in tokens_list]
        
        if states is None:
            states = self.init_state(batch_size)
        
        pos = [0] * batch_size
        
        if full_output:
            outputs = [torch.empty((0, self.config.vocab_size), 
                                  dtype=next(self.parameters()).dtype,
                                  device=next(self.parameters()).device) 
                      for _ in range(batch_size)]
        else:
            outputs = [None] * batch_size
        
        while True:
            active = [i for i in range(batch_size) if pos[i] < lengths[i]]
            if not active:
                break
            
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens_list[i][pos[i]:pos[i]+step] for i in active]
            
            batch_ids = torch.tensor(batch_tokens, dtype=torch.long, 
                                    device=next(self.parameters()).device)
            
            batch_states = self._extract_batch_states(states, active)
            batch_logits, batch_new_states = self.forward_with_state(batch_ids, batch_states)
            
            for k, i in enumerate(active):
                if full_output:
                    outputs[i] = torch.cat([outputs[i], batch_logits[k]], dim=0)
                else:
                    outputs[i] = batch_logits[k, -1, :]
                
                self._update_batch_state(states, i, batch_new_states, k)
                pos[i] += step
        
        return outputs, states
    
    def _extract_batch_states(self, states, indices):
        """Extract batch states for given indices."""
        batch_states = []
        for layer_state in states:
            batch_layer_state = {}
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    batch_layer_state[key] = value[indices]
                else:
                    batch_layer_state[key] = value
            batch_states.append(batch_layer_state)
        return batch_states
    
    def _update_batch_state(self, states, target_idx, batch_states, source_idx):
        """Update states[target_idx] with batch_states[source_idx]."""
        for layer_idx in range(len(states)):
            for key, value in batch_states[layer_idx].items():
                if isinstance(value, torch.Tensor):
                    states[layer_idx][key][target_idx] = value[source_idx]
    
    def load_state_dict(self, state_dict, strict=True, assign: bool = False):
        """
        Load weights, handling RWKV7 special weight format.
        """
        import copy
        new_state_dict = cast(dict[str, Any], copy.deepcopy(state_dict))

        emb_key = 'emb.weight'
        ln0_weight_key = 'blocks.0.ln0.weight'
        ln0_bias_key = 'blocks.0.ln0.bias'

        source_has_ln0 = ln0_weight_key in state_dict and ln0_bias_key in state_dict
        processed_embedding = False

        # Handle r_k flatten to match module parameters
        for key in list(new_state_dict.keys()):
            if key.endswith('att.r_k'):
                tensor = new_state_dict[key]
                if tensor.ndim == 1:
                    H = self.config.dim_att // self.config.head_size_a
                    N = self.config.head_size_a
                    new_state_dict[key] = tensor.view(H, N)

        # Layer 0 v params are ignored in official implementation
        drop_layer0_v_params = False
        if len(self.blocks) > 0:
            first_block = cast(RWKV7Block, self.blocks[0])
            drop_layer0_v_params = not hasattr(first_block.att, 'v1')

        for suffix in ('0', '1', '2'):
            v_key = f'blocks.0.att.v{suffix}'
            a_key = f'blocks.0.att.a{suffix}'
            if v_key in new_state_dict:
                if drop_layer0_v_params:
                    new_state_dict.pop(v_key, None)
                elif a_key in new_state_dict:
                    new_state_dict[v_key] = new_state_dict[a_key]

        load_result = super().load_state_dict(new_state_dict, strict=False, assign=assign)

        emb_preprocessed = processed_embedding or not source_has_ln0
        self._emb_preprocessed = emb_preprocessed

        if len(self.blocks) > 0:
            first_block = cast(RWKV7Block, self.blocks[0])
            first_block.skip_ln0 = emb_preprocessed

        return load_result
