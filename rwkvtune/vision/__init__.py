# -*- coding: utf-8 -*-
"""rwkvtune.vision — 多模态扩展子包（视觉编码器注册表 + 投影器 + 多模态模型）。

接线规格（V0.4 实测验证）:
    像素 [B, T帧, 3, H, W] (H/W 为 32 倍数)
    → 视觉塔 (原生 Qwen3_5VisionModel, 冻结) → pooler_output [n_merged, out_dim]
    → 投影器 (LayerNorm + Linear, 可训练) → [n_merged, n_embd]
    → 按 image_token_id 占位散射进 RWKV embedding 序列（图像在前/交错均可）
"""
from rwkvtune.vision.encoders import VisionEncoderRegistry, Qwen35Tower
from rwkvtune.vision.projector import VisionProjector
from rwkvtune.models.multimodal import MultimodalRWKV7Model

__all__ = ["VisionEncoderRegistry", "Qwen35Tower", "VisionProjector", "MultimodalRWKV7Model"]
