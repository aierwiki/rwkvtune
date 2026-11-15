"""
Official RWKV-7 reference implementation.

Math copied verbatim from:
- RWKV-LM/RWKV-v7/rwkv_v7_demo.py     (GPT/parallel mode, Tmix/CMix/Block/RWKV, non-kernel RWKV7_OP)
- RWKV-LM/RWKV-v7/rwkv_v7_demo_rnn.py (RNN mode state convention)

Only changes vs upstream: parameterized head_size / dim_ffn / layer counts so a small
random model can be built for numerical-equivalence testing. All tensor formulas are
byte-for-byte the official ones.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def RWKV7_OP(r, w, k, v, a, b, HEAD_SIZE):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C).to(dtype=r.dtype)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, args.D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(args.D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, args.D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(args.D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, args.D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(args.D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, args.D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(args.D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

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
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)  # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk * a, self.head_size)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd)  # only used in block 0
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x


# =====================================================================================
# RNN-mode reference (state convention from rwkv_v7_demo_rnn.py)
# state per layer: 0 = att_x_prev (post-ln1 x, dtype), 1 = att_kv [H,N,N] fp32, 2 = ffn_x_prev
# =====================================================================================

def rnn_time_mixing(layer_id, H, N, x, x_prev, v_first, state, z, C):
    """z: dict of parameters keyed like official checkpoint (single-token path)."""
    xx = x_prev - x
    xr, xw = x + xx * z['x_r'], x + xx * z['x_w']
    xk, xv = x + xx * z['x_k'], x + xx * z['x_v']
    xa, xg = x + xx * z['x_a'], x + xx * z['x_g']

    r = z['receptance'] @ xr
    w_lin = torch.tanh(xw @ z['w1']) @ z['w2']
    k = z['key'] @ xk
    v = z['value'] @ xv
    a = torch.sigmoid(z['a0'].view(C) + (xa @ z['a1']) @ z['a2'])
    g = torch.sigmoid(xg @ z['g1']) @ z['g2']

    kk = k * z['k_k']
    kk = F.normalize(kk.view(H, N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a - 1) * z['k_a'])

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(z['v0'] + (xv @ z['v1']) @ z['v2'])

    w = z['w0'] + w_lin.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w))  # official fused version; == exp(-exp(-softplus(-x)-0.5))

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H, N, 1)

    out = F.group_norm(out.view(1, H * N), num_groups=H, weight=z['ln_x_w'], bias=z['ln_x_b'], eps=64e-5).view(H * N)
    out = out + ((r * k * z['r_k']).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    return z['output'] @ (out * g), x, state, v_first


def rnn_channel_mixing(x, x_prev, z):
    xx = x_prev - x
    k = x + xx * z['x_k']
    k = torch.relu(z['key'] @ k) ** 2
    return z['value'] @ k, x


class OfficialRNN:
    """Token-by-token reference with the official state layout."""

    def __init__(self, model: RWKV, dtype=torch.float32):
        self.m = model
        self.dtype = dtype
        args = model.args if hasattr(model, 'args') else None
        # pull dims from modules
        b0 = model.blocks[0]
        self.C = b0.att.x_r.shape[-1]
        self.N = b0.att.head_size
        self.H = b0.att.n_head
        self.n_layer = len(model.blocks)

    def init_state(self, device):
        state = [None for _ in range(self.n_layer * 3)]
        for i in range(self.n_layer):
            state[i * 3 + 0] = torch.zeros(self.C, dtype=self.dtype, device=device)
            state[i * 3 + 1] = torch.zeros((self.H, self.N, self.N), dtype=torch.float, device=device)
            state[i * 3 + 2] = torch.zeros(self.C, dtype=self.dtype, device=device)
        return state

    def forward(self, token: int, state):
        m = self.m
        x = m.emb.weight[token]
        # official demo_rnn folds ln0 into the embedding before the loop
        if hasattr(m.blocks[0], 'ln0'):
            x = F.layer_norm(x, (self.C,), weight=m.blocks[0].ln0.weight, bias=m.blocks[0].ln0.bias)
        v_first = torch.empty_like(x)
        for i, block in enumerate(m.blocks):
            z = {}
            att = block.att
            z.update({
                'x_r': att.x_r.view(self.C), 'x_w': att.x_w.view(self.C), 'x_k': att.x_k.view(self.C),
                'x_v': att.x_v.view(self.C), 'x_a': att.x_a.view(self.C), 'x_g': att.x_g.view(self.C),
                'w0': att.w0.view(self.C), 'w1': att.w1, 'w2': att.w2,
                'a0': att.a0.view(self.C), 'a1': att.a1, 'a2': att.a2,
                'v0': att.v0.view(self.C), 'v1': att.v1, 'v2': att.v2,
                'g1': att.g1, 'g2': att.g2,
                'k_k': att.k_k.view(self.C), 'k_a': att.k_a.view(self.C), 'r_k': att.r_k.reshape(-1),
                'receptance': att.receptance.weight, 'key': att.key.weight,
                'value': att.value.weight, 'output': att.output.weight,
                'ln_x_w': att.ln_x.weight, 'ln_x_b': att.ln_x.bias,
            })
            xx = F.layer_norm(x, (self.C,), weight=block.ln1.weight, bias=block.ln1.bias)
            xx, new_x, new_kv, v_first = rnn_time_mixing(i, self.H, self.N, xx, state[i * 3 + 0], v_first, state[i * 3 + 1], z, self.C)
            x = x + xx
            state[i * 3 + 0] = new_x
            state[i * 3 + 1] = new_kv

            zz = {'x_k': block.ffn.x_k.view(self.C), 'key': block.ffn.key.weight, 'value': block.ffn.value.weight}
            xx2 = F.layer_norm(x, (self.C,), weight=block.ln2.weight, bias=block.ln2.bias)
            xx2, new_fx = rnn_channel_mixing(xx2, state[i * 3 + 2], zz)
            x = x + xx2
            state[i * 3 + 2] = new_fx

        x = F.layer_norm(x, (self.C,), weight=m.ln_out.weight, bias=m.ln_out.bias)
        x = m.head.weight @ x
        return x, state
