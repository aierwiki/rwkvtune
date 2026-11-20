# -*- coding: utf-8 -*-
"""多模态（视觉）Lightning 模块: 继承 RWKV7LightningModule, 注入 pixel 字段。

batch 契约: {"input_ids", "labels", "pixel_values" [Σt,3,H,W], "grid_thw" [Σn_img,3]}
模型契约: MultimodalRWKV7Model(input_ids, pixel_values, grid_thw) -> logits
"""
import torch

from rwkvtune.training.lightning_module import RWKV7LightningModule


class RWKV7VisionLightningModule(RWKV7LightningModule):
    """RWKV7 多模态训练模块（复用基类的优化器分组/恢复/L2Wrap, 只改前向）。"""

    def _forward_batch(self, batch) -> torch.Tensor:
        return self.model(
            batch["input_ids"],
            pixel_values=batch.get("pixel_values"),
            grid_thw=batch.get("grid_thw"),
            image_feats=batch.get("image_feats"),
        )

    def training_step(self, batch, batch_idx):
        import os
        loss = super().training_step(batch, batch_idx)
        # 步级进度(stdout 块缓冲下无 flush 的进度条整体不可见 —— v50 教训)
        if batch_idx % 10 == 0:
            v = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            print(f"[step {batch_idx}] loss={v:.4f} finite={v == v}", flush=True)
        if os.environ.get("RWKV_DEBUG_NAN") == "1":
            v = float(loss.item()) if torch.is_tensor(loss) else float(loss)
            print(f"[nan-debug] step {batch_idx} loss={v} finite={v == v and abs(v) != float('inf')}",
                  flush=True)
            if v != v:
                # 首个 NaN 步: 权重检查 + 对同一 batch 逐层定位
                bad_params = [n for n, p in self.named_parameters()
                              if p.requires_grad and not torch.isfinite(p).all()]
                print(f"[nan-debug] 非有限参数: {bad_params[:8]}", flush=True)
                with torch.no_grad():
                    logits2 = self._forward_batch(batch)
                    ce2 = torch.nn.functional.cross_entropy(
                        logits2.view(-1, logits2.size(-1)).float(),
                        batch["labels"].view(-1), ignore_index=-100)
                    print(f"[nan-debug] 重算 logits max={logits2.abs().max().item():.1f} "
                          f"CE={ce2.item():.4f} 类数={logits2.shape[-1]} "
                          f"labels 范围=[{batch['labels'].min().item()}, {batch['labels'].max().item()}]",
                          flush=True)
                    core = self.model
                    if hasattr(core, "base_model"):
                        core = core.base_model.model
                    x = core.rwkv.emb(batch["input_ids"])
                    print(f"[nan-debug] emb: finite={torch.isfinite(x).all().item()} "
                          f"max|x|={x.abs().max().item():.1f}")
                    if batch.get("pixel_values") is not None:
                        img = core.embed_images(batch["pixel_values"], batch["grid_thw"]).to(x.dtype)
                        print(f"[nan-debug] 投影输出: finite={torch.isfinite(img).all().item()} "
                              f"max={img.abs().max().item():.1f} shape={tuple(img.shape)}")
                        mask = (batch["input_ids"] == core.image_token_id)
                        x = x.masked_scatter(mask.unsqueeze(-1), img)
                    v_first = torch.empty_like(x)
                    for i, block in enumerate(core.rwkv.blocks):
                        x0 = x
                        if block.layer_id == 0:
                            x0 = block.ln0(x0)
                        att_out, v_first = block.att(block.ln1(x0), v_first)
                        x = x0 + att_out
                        fin = torch.isfinite(x).all().item()
                        print(f"[nan-debug] L{i} att: finite={fin} max|x|={x.abs().max().item():.1f}")
                        if not fin:
                            break
                        x = x + block.ffn(block.ln2(x))
                        fin = torch.isfinite(x).all().item()
                        print(f"[nan-debug] L{i} ffn: finite={fin} max|x|={x.abs().max().item():.1f}")
                        if not fin:
                            break
        return loss

    @torch.no_grad()
    def generate_greedy(self, input_ids, pixel_values=None, grid_thw=None,
                        max_new_tokens: int = 16):
        """训练后贪心生成（整序列重前向; MultimodalRWKV7Model 无 RNN-state API,
        且续写必须重注入像素, 否则占位 token 退化为均值 embedding）。"""
        self.model.eval()
        dev = next(self.model.parameters()).device
        seq = input_ids.to(dev)
        out = []
        for _ in range(max_new_tokens):
            logits = self.model(seq, pixel_values=pixel_values, grid_thw=grid_thw)
            t = int(logits[0, -1].float().argmax().item())
            out.append(t)
            seq = torch.cat([seq, torch.tensor([[t]], device=dev)], dim=1)
        return out
