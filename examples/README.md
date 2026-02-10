# RWKVTune Training Examples

This directory contains training example scripts for RWKVTune.

## Directory Structure

```
examples/
├── README.md                    # This file
├── train_sft_single_gpu.py      # SFT single GPU training Python script
├── run_sft_single_gpu.sh        # SFT single GPU training Shell script
├── test_generation.py           # Generation test Python script
├── run_test_generation.sh       # Generation test Shell script
└── data/
    └── sharegpt_sample_100.jsonl  # Sample data (100 ShareGPT conversations)
```

## Quick Start

### 1. SFT Single GPU Training

#### Option 1: Using Shell Script (Recommended)

```bash
# Set model path and run
MODEL_PATH=/path/to/rwkv/model ./run_sft_single_gpu.sh

# Or customize more parameters
MODEL_PATH=/path/to/model \
CTX_LEN=1024 \
MICRO_BSZ=4 \
EPOCHS=5 \
USE_LORA=1 \
./run_sft_single_gpu.sh
```

#### Option 2: Using Python Script

```bash
python train_sft_single_gpu.py \
    --model_path /path/to/rwkv/model \
    --data_file data/sharegpt_sample_100.jsonl \
    --output_dir output_sft \
    --ctx_len 2048 \
    --micro_bsz 2 \
    --epoch_count 3 \
    --use_lora
```

### 2. Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | (required) | RWKV model path |
| `--data_file` | `data/sharegpt_sample_100.jsonl` | Training data file |
| `--output_dir` | `output_sft` | Output directory |
| `--ctx_len` | 2048 | Context length |
| `--micro_bsz` | 2 | Batch size per GPU |
| `--accumulate_grad_batches` | 4 | Gradient accumulation steps |
| `--epoch_count` | 3 | Number of epochs |
| `--lr_init` | 2e-5 | Initial learning rate |
| `--lr_final` | 1e-6 | Final learning rate |
| `--warmup_steps` | 10 | Warmup steps |
| `--precision` | bf16 | Training precision (fp32/fp16/bf16) |
| `--use_lora` | False | Enable LoRA |
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA alpha |

## Data Format

### ShareGPT Format

Sample data uses ShareGPT multi-turn conversation format:

```json
{
    "conversations": [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hello! How can I help you today?"},
        {"from": "human", "value": "Tell me about RWKV"},
        {"from": "gpt", "value": "RWKV is a novel language model architecture combining RNN and Transformer advantages..."}
    ]
}
```

### Custom Data

To use your own data, ensure:

1. File format is JSONL (one JSON object per line)
2. Each object contains a `conversations` field
3. `conversations` is an array where each element has:
   - `from`: Role (`system`/`human`/`gpt` or `user`/`assistant`)
   - `value`: Content

## GPU Memory Requirements

| Model Size | Full Fine-tuning | LoRA (r=64) |
|------------|------------------|-------------|
| 0.1B | ~4GB | ~2GB |
| 0.4B | ~8GB | ~4GB |
| 1.5B | ~24GB | ~8GB |
| 3B | ~48GB | ~16GB |

*Note: Actual memory usage depends on context length, batch size, etc.*

## Troubleshooting

### 1. Out of Memory

- Reduce `--micro_bsz` (e.g., set to 1)
- Reduce `--ctx_len` (e.g., set to 1024 or 512)
- Enable `--use_lora`

### 2. Slow Training

- Increase `--micro_bsz` (if memory allows)
- Use `--precision bf16` (if GPU supports it)

### 3. Loss Not Decreasing

- Verify data format is correct
- Try increasing `--lr_init`
- Check if data volume is sufficient

## Sample Data Source

Sample dataset from HuggingFace:
- Dataset: `agentlans/combined-roleplay`
- Subset: `chai_prize_reward_model`
- Samples: 100

## License

Apache License 2.0
