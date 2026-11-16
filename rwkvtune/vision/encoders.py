# -*- coding: utf-8 -*-
"""视觉编码器注册表。新增编码器: 注册一个 builder(config, tower_dir) -> nn.Module。"""
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


class VisionEncoderRegistry:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str, builder: Callable) -> None:
        cls._registry[name] = builder

    @classmethod
    def build(cls, name: str, **kwargs) -> nn.Module:
        if name not in cls._registry:
            raise KeyError(f"未知视觉编码器 {name!r}; 已注册: {sorted(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def known(cls):
        return sorted(cls._registry)


class Qwen35Tower(nn.Module):
    """Qwen3.5 视觉塔包装（transformers 原生 Qwen3_5VisionModel）。

    - 权重来自 tower_dir/tower.safetensors（键前缀 model.visual. 已剥离）
    - 默认冻结; embed_dim = out_hidden_size（G1k 接线目标维度）
    - forward(pixel_values [B,T帧,3,H,W], grid_thw [n,3]) -> pooler_output [n_merged, out_dim]
    """

    def __init__(self, vision_config, tower_weights: Path = None, freeze: bool = True):
        super().__init__()
        from transformers.models.qwen3_5 import Qwen3_5VisionModel
        self.tower = Qwen3_5VisionModel(vision_config)
        if tower_weights is not None:
            from safetensors.torch import load_file
            sd = load_file(str(tower_weights))
            # tower.safetensors 键为 model.visual.* -> 剥前缀对齐 Qwen3_5VisionModel 子模块
            sd = {k.replace("model.visual.", ""): v for k, v in sd.items()}
            missing, unexpected = self.tower.load_state_dict(sd, strict=False)
            assert len(missing) == 0, f"tower 缺键: {missing[:4]}"
            assert len(unexpected) == 0, f"tower 多余键: {unexpected[:4]}"
        self.embed_dim = vision_config.out_hidden_size
        if freeze:
            for p in self.tower.parameters():
                p.requires_grad_(False)
        self.freeze = freeze

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            out = self.tower(pixel_values, grid_thw)
        return out.pooler_output


def _build_qwen35(tower_dir: str = None, freeze: bool = True, **kwargs):
    import json
    from transformers import AutoConfig
    tower_dir = Path(tower_dir)
    cfg = AutoConfig.from_pretrained(str(tower_dir))
    weights = tower_dir / "tower" / "tower.safetensors"
    if not weights.exists():
        weights = tower_dir / "tower.safetensors"
    return Qwen35Tower(cfg.vision_config, tower_weights=weights, freeze=freeze)


VisionEncoderRegistry.register("qwen3_5", _build_qwen35)
