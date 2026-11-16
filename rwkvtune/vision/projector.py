# -*- coding: utf-8 -*-
"""视觉→语言投影器（可训练）: LayerNorm + Linear(+MLP)。"""
import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """embed_dim(1024) -> n_embd(RWKV hidden)。LayerNorm 在前（VLM 标准做法）。

    hidden 匹配时可跳过降维（mlp=False 退化为 LN+Linear 直通降维/同维）。
    """

    def __init__(self, embed_dim: int, n_embd: int, mlp: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        if mlp:
            mid = max(embed_dim, n_embd)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, mid), nn.GELU(), nn.Linear(mid, n_embd))
        else:
            self.fc = nn.Linear(embed_dim, n_embd)
        nn.init.normal_(self.fc[-1].weight if isinstance(self.fc, nn.Sequential)
                        else self.fc.weight, std=0.02)
        if isinstance(self.fc, nn.Sequential):
            nn.init.zeros_(self.fc[-1].bias)
        else:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))
