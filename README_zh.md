# RWKVTune

RWKV 模型训练工具库 -- 支持预训练、SFT 和 GRPO 三大训练范式。

[![PyPI version](https://badge.fury.io/py/rwkvtune.svg)](https://badge.fury.io/py/rwkvtune)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **[English Documentation](README.md)**

## 核心特性

- **三大训练范式**
  - `PretrainTrainer`: 继续预训练
  - `SFTTrainer`: 监督微调
  - `GRPOTrainer`: GRPO 强化学习对齐

- **高效训练**
  - 多 GPU，DeepSpeed ZeRO Stage 2/3
  - 梯度检查点降低显存
  - 混合精度 (bf16 / fp16 / fp32)
  - 梯度累积

- **参数高效微调 (PEFT)**
  - LoRA 支持，可自定义目标模块
  - 一键合并 LoRA 权重

- **高级功能**
  - 无限上下文训练
  - HuggingFace Datasets 集成
  - 断点续训
  - 多种 GRPO 损失: DAPO / DR-GRPO / BNPO / GRPO
  - Rollout 后处理钩子
  - Rollout 数据保存
  - 日志: SwanLab / WandB / TensorBoard

- **CLI 工具**
  - `rwkvtune-merge-lora`: 合并 LoRA 权重
  - `rwkvtune-create-hub`: 创建标准模型目录

## 安装

```bash
pip install rwkvtune

# 源码安装
git clone https://github.com/rwkv-community/rwkvtune.git
cd rwkvtune && pip install -e .

# DeepSpeed 支持
pip install rwkvtune[deepspeed]
```

## 模型准备

训练前将 RWKV 权重转换为标准 hub 目录。

### 使用 rwkvtune-create-hub

```bash
rwkvtune-create-hub \
    --output-dir models/rwkv7-0.1b \
    --model-file /path/to/rwkv7-0.1b.pth \
    --config-name rwkv7-0.1b
```

目录结构:

```
models/rwkv7-0.1b/
  config.json
  model.pth
  tokenizer_config.json
  vocab.txt
  generation_config.json
```

### 从 ModelScope 下载

```python
from modelscope import snapshot_download
model_dir = snapshot_download('aierwiki/rwkv7-g1d-0.1b')
```

## 快速上手

### 监督微调 (SFT)

```python
from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.training import SFTConfig, SFTTrainer
from datasets import Dataset

model = AutoModel.from_pretrained("/path/to/model")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

def prepare_data(example):
    prompt = f"User: {example['instruction']}\n\nAssistant: "
    completion = example['output']
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels}

dataset = Dataset.from_list([...]).map(prepare_data)

config = SFTConfig(
    ctx_len=2048, micro_bsz=4, epoch_count=3,
    lr_init=2e-5, lr_final=1e-6, precision="bf16",
)

trainer = SFTTrainer(
    model=model, args=config,
    train_dataset=dataset, processing_class=tokenizer,
)
trainer.train()
```

### SFT + LoRA

```python
from rwkvtune import AutoModel
from rwkvtune.peft import LoraConfig, get_peft_model
from rwkvtune.training import SFTConfig, SFTTrainer

model = AutoModel.from_pretrained("/path/to/model")
lora_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0)
model = get_peft_model(model, lora_config)

config = SFTConfig(ctx_len=2048, micro_bsz=4, epoch_count=3)
trainer = SFTTrainer(model=model, args=config, train_dataset=dataset)
trainer.train()
```

### GRPO 训练

```python
from rwkvtune.training import GRPOConfig, GRPOTrainer
from rwkvtune.peft import LoraConfig
from datasets import Dataset

def reward_func(prompts, completions, **kwargs):
    return [1.0 if "correct" in c else 0.0 for c in completions]

dataset = Dataset.from_list([{"prompt": "What is 2+2?", "input_ids": [...]}])

config = GRPOConfig(
    micro_bsz=1, num_generations=8,
    max_prompt_length=2048, max_completion_length=512,
    steps_per_generation=8, temperature=1.0, top_p=0.95,
    lr_init=1e-6, lr_final=1e-7, accumulate_grad_batches=8,
    loss_type="dapo", beta=0.04, save_rollout_steps=16,
)

lora_config = LoraConfig(r=64, lora_alpha=128)
trainer = GRPOTrainer(
    model="/path/to/model", reward_funcs=reward_func,
    args=config, train_dataset=dataset, peft_config=lora_config,
)
trainer.train()
```

### 继续预训练

```python
from rwkvtune.training import PretrainConfig, PretrainTrainer
from datasets import Dataset

dataset = Dataset.from_list([{"input_ids": [...], "labels": [...]}])
config = PretrainConfig(ctx_len=4096, micro_bsz=8, epoch_count=1, lr_init=3e-4)
trainer = PretrainTrainer(model="/path/to/model", args=config, train_dataset=dataset)
trainer.train()
```

## CLI 工具

### rwkvtune-create-hub -- 创建标准模型目录

将 .pth 权重转为 HuggingFace 风格目录。

```bash
rwkvtune-create-hub \
    --output-dir models/rwkv7-0.1b \
    --model-file /path/to/rwkv7-0.1b.pth \
    --config-name rwkv7-0.1b
```

| 参数 | 说明 |
|------|------|
| `--output-dir` | 输出目录(必需) |
| `--model-file` | 权重文件路径(必需) |
| `--config-name` | 配置名称(必需) |
| `--ctx-len` | 覆盖上下文长度 |
| `--chat-template` | 聊天模板(.jinja) |
| `--link-weights` | 符号链接(省空间) |
| `--save-format` | pth / safetensors |
| `--overwrite` | 覆盖已有目录 |

可用配置:

| config-name | 层数 | 维度 | 参数量 |
|-------------|------|------|--------|
| rwkv7-0.1b | 12 | 768 | 0.1B |
| rwkv7-0.4b | 24 | 1024 | 0.4B |
| rwkv7-1.5b | 24 | 2048 | 1.5B |
| rwkv7-2.9b | 32 | 2560 | 2.9B |
| rwkv7-7.2b | 32 | 4096 | 7.2B |
| rwkv7-13.3b | - | - | 13.3B |

### rwkvtune-merge-lora -- 合并 LoRA 权重

将 LoRA 适配器合并到基础模型。

```bash
rwkvtune-merge-lora \
    --base-model models/rwkv7-g1d-0.1b \
    --lora-model output_sft/rwkv7-epoch3 \
    --output models/rwkv7-merged
```

| 参数 | 说明 |
|------|------|
| `--base-model`, `-b` | 基础模型目录(必需) |
| `--lora-model`, `-l` | LoRA 目录(必需) |
| `--output`, `-o` | 输出目录(必需) |
| `--device` | cuda / cpu(默认 cpu) |
| `--precision` | fp32/fp16/bf16(默认 bf16) |
| `--save-format` | pth / safetensors |

合并后自动: 合并权重、更新 config.json、复制分词器、生成 merge_info.json。

## 示例脚本

| 脚本 | 说明 |
|------|------|
| `run_sft_single_gpu.sh` | 单卡 SFT |
| `run_grpo_single_gpu.sh` | 单卡 GRPO |
| `run_sft_grpo_single_gpu.sh` | SFT+GRPO 流水线 |
| `run_test_generation.sh` | 生成测试 |
| `train_sft_single_gpu.py` | SFT Python |
| `train_grpo_single_gpu.py` | GRPO Python |
| `test_generation.py` | 生成测试 Python |

```bash
cd examples/
MODEL_PATH=/path/to/model ./run_sft_single_gpu.sh
./run_grpo_single_gpu.sh
./run_test_generation.sh
```

环境变量覆盖配置:

```bash
MODEL_PATH=models/rwkv7-0.4b CTX_LEN=4096 MICRO_BSZ=4 ./run_sft_single_gpu.sh
```

## 多 GPU 训练

```python
config = SFTConfig(devices=4, strategy="deepspeed_stage_2", precision="bf16")
```

## 配置参数

### SFTConfig / PretrainConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ctx_len` | int | 2048 | 上下文长度 |
| `micro_bsz` | int | 4 | 每 GPU 批次大小 |
| `epoch_count` | int | 10 | 训练轮数 |
| `epoch_steps` | int | None | 每轮步数限制 |
| `epoch_save` | int | 1 | epoch 保存间隔 |
| `save_every_n_batches` | int | 0 | batch 保存间隔 |
| `save_total_limit` | int | 2 | 最大 checkpoint 数 |
| `lr_init` | float | 3e-4 | 初始学习率 |
| `lr_final` | float | 1e-5 | 最终学习率 |
| `warmup_steps` | int | 50 | 预热步数 |
| `weight_decay` | float | 0.0/0.01 | 权重衰减 |
| `accumulate_grad_batches` | int | 1 | 梯度累积 |
| `grad_cp` | int | 1 | 梯度检查点 |
| `grad_clip` | float | 1.0 | 梯度裁剪 |
| `devices` | int | 1 | GPU 数量 |
| `precision` | str | bf16 | 训练精度 |
| `strategy` | str | auto | 训练策略 |
| `train_type` | str | normal | normal / infctx |
| `resume_from_checkpoint` | str | None | 续训路径 |
| `report_to` | str | "" | 日志工具 |

### GRPOConfig

**核心:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_generations` | int | 8 | 每 prompt 生成数 |
| `num_iterations` | int | 1 | 每批次迭代数 |
| `steps_per_generation` | int | 1 | 每 rollout 步数 |
| `micro_bsz` | int | 1 | 每批次 prompt 数 |

**生成:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_prompt_length` | int | 512 | 最大 prompt 长度 |
| `max_completion_length` | int | 256 | 最大 completion 长度 |
| `temperature` | float | 1.0 | 采样温度 |
| `top_p` | float | 1.0 | 核采样 |
| `top_k` | int | 0 | Top-K |
| `repetition_penalty` | float | 1.0 | 重复惩罚 |

**损失与优化:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss_type` | str | dapo | dapo/dr_grpo/bnpo/grpo |
| `epsilon` | float | 0.2 | PPO clip 下界 |
| `epsilon_high` | float | None | PPO clip 上界 |
| `scale_rewards` | str | group | group/batch/none |
| `advantage_clip` | float | None | 优势裁剪 |
| `low_reward_threshold` | float | None | 低质量组阈值 |
| `low_reward_scale` | float | 0.01 | 低质量缩放 |
| `beta` | float | 0.0 | KL 惩罚 |
| `reward_weights` | List | None | 多奖励权重 |

**Checkpoint 与日志:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_every_n_batches` | int | 0 | batch 保存间隔 |
| `save_total_limit` | int | 2 | checkpoint 上限 |
| `save_rollout_steps` | int | 0 | rollout 保存间隔 |
| `save_rollout_path` | str | None | rollout 目录 |
| `report_to` | str | None | swanlab/wandb |

**生成优化:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prefill_chunk_size` | int | 2048 | Prefill 分块 |
| `max_prefill_batch_size` | int | -1 | Prefill 批次限制 |
| `max_decode_batch_size` | int | -1 | Decode 批次限制 |
| `logprob_batch_size` | int | None | Logprob 分块 |

### LoraConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | int | 64 | LoRA 秩 |
| `lora_alpha` | int | 128 | LoRA alpha |
| `lora_dropout` | float | 0.0 | dropout |
| `target_modules` | list | auto | 目标模块 |

## 高级功能

### 无限上下文训练

```python
config = SFTConfig(train_type="infctx", ctx_len=32768, chunk_ctx=512, truncated_bptt=True, grad_cp=1)
```

### GRPO Completion 后处理钩子

在 rollout 后、reward 前插入自定义后处理:

```python
import torch

def my_postprocessor(prompts, completions, completion_ids, masks, tokenizer, **extra_fields):
    new_completions = [process(text) for text in completions]
    B_G, C = completion_ids.shape
    new_ids = torch.zeros(B_G, C, dtype=completion_ids.dtype, device=completion_ids.device)
    new_masks = torch.zeros(B_G, C, dtype=masks.dtype, device=completion_ids.device)
    for i, text in enumerate(new_completions):
        ids = tokenizer.encode(text)
        n = min(len(ids), C)
        new_ids[i, :n] = torch.tensor(ids[:n], dtype=completion_ids.dtype)
        new_masks[i, :n] = True
    return {"completions": new_completions, "completion_ids": new_ids, "masks": new_masks}

trainer = GRPOTrainer(..., completion_postprocess_fn=my_postprocessor)
```

约定:
- 输入: prompts, completions, completion_ids [B*G,C], masks [B*G,C], tokenizer, 额外字段
- 输出: dict 含 completions, completion_ids, masks
- 修改文本后须重新编码; 张量保持 [B*G, C] 形状

### 多奖励函数

```python
trainer = GRPOTrainer(
    reward_funcs=[format_reward, quality_reward],
    args=GRPOConfig(reward_weights=[0.5, 1.0]),
    ...
)
```

### Rollout 数据保存

```python
config = GRPOConfig(save_rollout_steps=16, save_rollout_path="output_grpo/rollouts")
```

## 典型工作流

```bash
# SFT -> 合并 LoRA -> GRPO -> 合并 -> 测试
cd examples/ && ./run_sft_single_gpu.sh
rwkvtune-merge-lora -b models/base -l output_sft/epoch3 -o models/sft-merged
MODEL_PATH=models/sft-merged ./run_grpo_single_gpu.sh
rwkvtune-merge-lora -b models/sft-merged -l output_grpo/batch512 -o models/final
./run_test_generation.sh
```

## 模型支持

- RWKV-7 (0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B)

## 环境要求

- Python >= 3.8, PyTorch >= 2.0.0, Lightning >= 2.0.0
- transformers >= 4.30.0, datasets >= 2.12.0
- CUDA (推荐)

## License

Apache License 2.0

## Acknowledgments

- [RWKV](https://github.com/BlinkDL/RWKV-LM) - [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) - [trl](https://github.com/huggingface/trl)
