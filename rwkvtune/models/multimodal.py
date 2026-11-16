# -*- coding: utf-8 -*-
"""多模态 RWKV7 模型: 视觉塔(冻结) + 投影器(可训) + RWKV7(可训/LoRA)。

约定:
- input_ids 中出现 image_token_id 的位置 = 图像占位; 其 embedding 会被
  投影器输出按顺序替换（pixel_values/grid_thw 的图像顺序必须与占位顺序一致）
- 多帧: grid_thw 每行 [t, h_p, w_p]; 像素 [B_total_frames?, ...] 按原生约定打包
  (单图多帧: pixel_values[B=1 图, t帧, 3, H, W], grid 行 [t, h_p, w_p])
"""
import torch
import torch.nn as nn


class MultimodalRWKV7Model(nn.Module):
    def __init__(self, rwkv: nn.Module, vision_encoder: nn.Module,
                 projector: nn.Module, image_token_id: int):
        super().__init__()
        self.rwkv = rwkv
        self.config = rwkv.config
        self.vision_encoder = vision_encoder
        self.projector = projector
        self.image_token_id = image_token_id
        self._expand_embedding_for_image_token()

    def _expand_embedding_for_image_token(self):
        """在词表尾部追加 image_token 行（均值初始化）。"""
        old = self.rwkv.emb
        if self.image_token_id < old.num_embeddings:
            return  # 已存在（如复用罕见 token）
        new_vocab = self.image_token_id + 1
        new_emb = nn.Embedding(new_vocab, old.embedding_dim)
        with torch.no_grad():
            mean = old.weight.mean(dim=0, keepdim=True)
            new_emb.weight[: old.num_embeddings] = old.weight
            new_emb.weight[old.num_embeddings:] = mean
        self.rwkv.emb = new_emb

    def embed_images(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(pixel_values, grid_thw)      # [n_merged, embed_dim]
        return self.projector(feats)                              # [n_merged, n_embd]

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor = None,
                grid_thw: torch.Tensor = None, **kw) -> torch.Tensor:
        x = self.rwkv.emb(input_ids)                              # [B, T, C]
        if pixel_values is not None:
            img = self.embed_images(pixel_values, grid_thw)       # [n_img_tokens, C]
            mask = (input_ids == self.image_token_id)
            n_mask = int(mask.sum().item())
            assert n_mask == img.shape[0], (
                f"图像占位数 {n_mask} != 视觉 token 数 {img.shape[0]} "
                f"(grid_thw 与 <image> 占位顺序必须一致)")
            flat = x[mask]                                        # [n_mask, C]（行序=占位顺序）
            x = x.masked_scatter(mask.unsqueeze(-1), img)
            assert torch.isfinite(x).all(), "图像 embedding 注入后出现非有限值"
        v_first = torch.empty_like(x)
        for block in self.rwkv.blocks:
            if block.layer_id == 0:
                x = block.ln0(x)
            x = x + block.att(block.ln1(x), v_first)[0]
            x = x + block.ffn(block.ln2(x))
        x = self.rwkv.ln_out(x)
        return self.rwkv.head(x)
