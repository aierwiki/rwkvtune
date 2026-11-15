# FIXES — official RWKV-7 alignment (branch `fix/official-rwkv7-alignment`)

Reference standard: [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) `RWKV-v7`
(`rwkv_v7_demo.py` GPT mode, `rwkv_v7_demo_rnn.py` RNN state convention, `cuda/wkv7.cu`).
Interfaces and project structure are unchanged; only implementation-level defects were fixed.

Verified by `tests/test_official_alignment.py` (numerical equivalence against a verbatim
copy of the official math, plus regressions for each fix below). All tests pass on CPU;
CUDA kernels unchanged in math (verified same formulas as official `wkv7.cu`).

## Fixed

| ID | Severity | Fix |
|---|---|---|
| D1 | P0 | `RUN_CUDA_RWKV7S` (wkv_loader.py): the wkv7s kernel writes the updated state back into the passed buffer **in-place**, and the wrapper returned that caller-owned tensor — under serving, any aliased buffer (state cache, other slots) could be corrupted, and CPU/CUDA mutation semantics differed. Now the wrapper clones the state, runs the kernel on the clone, and returns the updated clone. |
| D2 | P1 | `RWKV7Model.init_state` hardcoded `N=64, H=n_embd//64`, ignoring `config.head_size_a` — wrong state shapes (crash or silent corruption) for head_size≠64 models. Now uses `config.head_size_a`. |
| D4 | P1 | Operator caches were single-slot globals ignoring `head_size` — a second model with a different head size in the same process received the wrong compiled operator. Caches are now keyed by head_size (also fixes JIT module name collisions like `wind_backstepping_h64`). |
| D5 | P1 | `RWKV7ChannelMix` hardcoded `n_embd*4` while `config.dim_ffn` (computed as 3.5×, the official train.py from-scratch default) was never read. The module now honors `config.dim_ffn` (default 4× — all released World/G1 checkpoints use 4×; G1-1.5B verified: intermediate=8192=4×2048). Trainers' implicit default set to 4× for the same reason. `SimpleTrainConfig` gained an explicit `dim_ffn` field (0 = auto). |
| D6 | P3 | `load_state_dict` dead code removed (`processed_embedding` was never set; the layer-0 a→v copy branch was unreachable since layer 0 never creates v params). |
| D7 | P2 | SFT example chat template deviated from the official G1 templates (`RWKV7-G1x-templates.txt`): `Assistant:<think` lacked the space (`Assistant: <think`), and inserting an eos after user turns by default broke the official `User: xxx\n\nAssistant:` shape. Space fixed; eos is now strictly opt-in (default = official template behavior). |
| D8 | P3 | `assert T <= ctx_len` prevented running beyond the trained context; official RWKV runs (degraded, no crash). Now a `RuntimeWarning`. |
| D10 | P3 | Added the official demo's missing shape validation: `dim_att % head_size == 0` and `n_embd == n_head * head_size`. |

## Deliberately not changed

- **Sampling flavor**: temperature applied before top-p (HF/OpenAI style) vs official `sample_logits` (temperature after top-p filtering). Kept HF-style for ecosystem compatibility.
- **Training kernel dtype**: the wind-backstepping training kernel requires bf16 and `T % 16 == 0`; official also supports fp16 ("half better"). Left as-is (fp16 support would need kernel work); documented here.
- **wkv7s/wkv7 CUDA kernel math**: verified identical to official `wkv7.cu` (state update `s = s*w[j] + k[j]*v + sa*b[j]`, decay `exp(-exp(w))`, per-(b,h,i) row layout). Not modified.
