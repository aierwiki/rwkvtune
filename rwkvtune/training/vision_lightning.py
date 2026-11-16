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
        )

    @torch.no_grad()
    def generate_greedy(self, input_ids, pixel_values=None, grid_thw=None,
                        max_new_tokens: int = 16):
        """训练后贪心生成（逐 token 状态路径; v1 无图像尾部注入时纯文本续写）。"""
        self.model.eval()
        st = self.model.init_state(input_ids.shape[0])
        out = []
        cur = input_ids
        for _ in range(max_new_tokens):
            logits, st = self.model.forward_with_state(cur.to(next(self.model.parameters()).device), st)
            t = int(logits[0, -1].float().argmax().item())
            out.append(t)
            cur = torch.tensor([[t]], device=cur.device)
        return out
