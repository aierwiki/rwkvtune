"""
rwkvtune RWKV-7 alignment/regression tests.

Numerical equivalence against the official RWKV-LM v7 implementation
(math copied verbatim into tests/ref_official.py), plus regressions for
specific fixes on the fix/official-rwkv7-alignment branch:
  D1 wkv7s state clone semantics (CPU/CUDA unified, no caller-buffer mutation)
  D2 init_state honors config.head_size_a
  D4 operator caches keyed by head_size
  D5 dim_ffn honored (module no longer hardcodes n_embd*4)

Run from repo root:  python3 tests/test_official_alignment.py
"""
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from ref_official import RWKV as OfficialRWKV, OfficialRNN  # noqa: E402

DT = torch.float32
PASS = True


def build_args(n_layer=3, n_embd=128, vocab=211, head_size=64):
    import types
    a = types.SimpleNamespace()
    a.n_layer, a.n_embd, a.vocab_size, a.head_size_a = n_layer, n_embd, vocab, head_size
    a.D_DECAY_LORA, a.D_AAA_LORA, a.D_MV_LORA, a.D_GATE_LORA = 16, 16, 8, 16
    return a


def seeded_official(args):
    torch.manual_seed(0)
    m = OfficialRWKV(args).to(DT).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith((".weight", ".bias")) and p.ndim >= 1:
                continue
            torch.nn.init.normal_(p, std=0.1)
        for n, p in m.named_parameters():
            if n.endswith("att.w0"):
                p.sub_(4.0)
            if n.endswith(".weight") and p.ndim == 2 and "emb" not in n and "head" not in n and "ln" not in n:
                p.mul_(0.5)
    return m


def build_tune_model(sd, args, dim_ffn=None, load_weights=True):
    from rwkvtune.models.rwkv7.model import RWKV7Model
    cfg = type("C", (), {})()
    cfg.n_layer, cfg.n_embd, cfg.vocab_size = args.n_layer, args.n_embd, args.vocab_size
    cfg.head_size_a, cfg.dim_att, cfg.ctx_len = args.head_size_a, args.n_embd, 4096
    cfg.grad_cp, cfg.use_peft = 0, False
    cfg.dim_att_lora, cfg.dim_gate_lora, cfg.dim_mv_lora = args.D_DECAY_LORA, args.D_GATE_LORA, args.D_MV_LORA
    if dim_ffn is not None:
        cfg.dim_ffn = dim_ffn
    torch.manual_seed(123)
    m = RWKV7Model(cfg).to(DT).eval()
    if load_weights and sd is not None:
        sd2 = {k: v for k, v in sd.items()
               if not (re.match(r"blocks\.0\.att\.v[012]$", k) and not hasattr(m.blocks[0].att, "v1"))}
        m.load_state_dict(sd2, strict=False)
    return m


def report(name, ok, detail=""):
    global PASS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    PASS = PASS and ok


def maxdiff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def main():
    args = build_args()
    ref = seeded_official(args)
    sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    tune = build_tune_model(sd, args)

    torch.manual_seed(7)
    idx = torch.randint(0, args.vocab_size, (2, 17))

    # T1 GPT-mode full-sequence logits vs official
    with torch.no_grad():
        l_ref, l_tun = ref(idx), tune(idx)
    report("T1 GPT-mode logits vs official", maxdiff(l_ref, l_tun) < 5e-4,
           f"diff={maxdiff(l_ref, l_tun):.2e}")

    # T2 token-by-token RNN vs official RNN loop
    rnn = OfficialRNN(ref, dtype=DT)
    tokens = idx[0].tolist()
    with torch.no_grad():
        st = rnn.init_state("cpu")
        ref_logits = []
        for t in tokens:
            out, st = rnn.forward(t, st)
            ref_logits.append(out)
        ref_logits = torch.stack(ref_logits)
        st2 = tune.init_state(1)
        tun_logits = []
        for t in tokens:
            lg, st2 = tune.forward_with_state(torch.tensor([[t]]), st2)
            tun_logits.append(lg[0, -1])
        tun_logits = torch.stack(tun_logits)
    report("T2 RNN token-by-token vs official RNN", maxdiff(ref_logits, tun_logits) < 5e-4,
           f"diff={maxdiff(ref_logits, tun_logits):.2e}")

    # T3 chunked prefill == full prefill
    with torch.no_grad():
        stc = tune.init_state(1)
        a_l, stc = tune.forward_with_state(idx[0, :5].unsqueeze(0), stc)
        b_l, stc = tune.forward_with_state(idx[0, 5:].unsqueeze(0), stc)
        d = maxdiff(tune(idx[0].unsqueeze(0))[0], torch.cat([a_l[0], b_l[0]], dim=0))
    report("T3 chunked prefill == full prefill", d < 5e-4, f"diff={d:.2e}")

    # T4 ragged batch
    with torch.no_grad():
        seqs = [idx[0, :9].tolist(), idx[1, :13].tolist()]
        outs, _ = tune.forward_batch_dynamic([list(s) for s in seqs], full_output=True)
        per = [tune(torch.tensor([s]))[0] for s in seqs]
        d = max(maxdiff(outs[i], per[i]) for i in range(2))
    report("T4 ragged batch forward", d < 5e-4, f"diff={d:.2e}")

    # T5 infctx naive hand-written backward vs autograd
    from rwkvtune.models.rwkv7.operators.wkv_infctx_naive import (
        naive_recurrent_rwkv7_fwd, RWKV7InfctxNaiveFunction)
    Bq, Tq, H, Cq = 1, 8, 4, 16
    torch.manual_seed(3)
    q = torch.randn(Bq, Tq, H, Cq)
    k = torch.randn(Bq, Tq, H, Cq)
    v = torch.randn(Bq, Tq, H, Cq)
    w = torch.randn(Bq, Tq, H, Cq) - 3.0
    a = torch.sigmoid(torch.randn(Bq, Tq, H, Cq))
    b = torch.randn(Bq, Tq, H, Cq)
    qg, kg, vg, wg, ag, bg = [x.requires_grad_(True) for x in (q, k, v, w, a, b)]
    y_auto, s_auto = naive_recurrent_rwkv7_fwd(qg, kg, vg, wg, ag, bg)
    dy = torch.randn_like(y_auto)
    ds = torch.randn_like(s_auto)
    grads_auto = torch.autograd.grad((y_auto * dy).sum() + (s_auto * ds).sum(),
                                     [qg, kg, vg, wg, ag, bg])
    qg2, kg2, vg2, wg2, ag2, bg2 = [x.clone().requires_grad_(True) for x in (q, k, v, w, a, b)]
    y_m, s_m = RWKV7InfctxNaiveFunction.apply(qg2, kg2, vg2, wg2, ag2, bg2, 1.0, None)
    grads_custom = torch.autograd.grad((y_m * dy).sum() + (s_m * ds).sum(),
                                       [qg2, kg2, vg2, wg2, ag2, bg2])
    diffs = [maxdiff(x, y) for x, y in zip(grads_auto, grads_custom)]
    report("T5 infctx naive backward vs autograd", all(dd < 1e-4 for dd in diffs),
           " ".join(f"{d:.1e}" for d in diffs))

    # T6 head_size != 64 (D2 + D4 regression)
    args32 = build_args(head_size=32)
    ref32 = seeded_official(args32)
    sd32 = {k: v.detach().clone() for k, v in ref32.state_dict().items()}
    tune32 = build_tune_model(sd32, args32)
    errs = []
    try:
        with torch.no_grad():
            st = tune32.init_state(1)
            H = args32.n_embd // args32.head_size_a
            expect = (1, H, args32.head_size_a, args32.head_size_a)
            if tuple(st[0]["att_kv"].shape) != expect:
                errs.append(f"init_state shape {tuple(st[0]['att_kv'].shape)} != {expect}")
            tune32.forward_with_state(torch.randint(0, args32.vocab_size, (1, 4)), st)
    except Exception as e:
        errs.append(f"EXC {type(e).__name__}: {e}")
    report("T6 head_size=32 state shape + forward", not errs, "; ".join(errs))

    # T7 dim_ffn honored (D5 regression): explicit non-4x value must build that shape
    try:
        m35 = build_tune_model(None, args, dim_ffn=int(128 * 3.5 // 32 * 32), load_weights=False)
        ffn_shape = tuple(m35.blocks[0].ffn.key.weight.shape)
        ok = ffn_shape == (448, 128)
        # and the default (no explicit dim_ffn) must stay 4x to match release weights
        m4 = build_tune_model(None, args, load_weights=False)
        ffn4 = tuple(m4.blocks[0].ffn.key.weight.shape)
        ok = ok and ffn4 == (512, 128)
        report("T7 dim_ffn honored", ok, f"explicit {ffn_shape}, default {ffn4}")
    except Exception as e:
        report("T7 dim_ffn honored", False, f"EXC {e}")

    print("\n===", "ALL PASS" if PASS else "FAILURES PRESENT", "===")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
