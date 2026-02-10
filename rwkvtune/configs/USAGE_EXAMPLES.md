# RWKV7 Configuration System Usage Examples

## Quick Start

### 1. List Available Models

```bash
# Run from project directory
python -c "from rwkvtune.configs import list_available_models; list_available_models()"
```

Example output:
```
Available model configurations:
--------------------------------------------------------------------------------
  0.1b     -> RWKV7-0.1B           (12 layers, 768 dims)
  0.4b     -> RWKV7-0.4B           (24 layers, 1024 dims)
  1.5b     -> RWKV7-1.5B           (24 layers, 2048 dims)
  2.9b     -> RWKV7-2.9B           (32 layers, 2560 dims)
```

### 2. Test Model Loading

```bash
# Test if all models can be loaded correctly
python examples/test_model_configs.py
```

Expected output:
```
 Test Summary
================================================================================
  0.1b    : ✅ Passed
  0.4b    : ✅ Passed
  1.5b    : ✅ Passed
  2.9b    : ✅ Passed
--------------------------------------------------------------------------------
  Total: 4/4 Passed
================================================================================

🎉 All model tests passed!
```

### 3. Start Training

Train models of any scale using configuration files:

```bash
# Train 0.1B model (quick test)
bash examples/train_with_config.sh 0.1b

# Train 1.5B model (production use)
bash examples/train_with_config.sh 1.5b

# Train 2.9B model (large model)
bash examples/train_with_config.sh 2.9b
```

Training output example:
```
========================================
  RWKV7 General Fine-tuning Training Script
========================================
[1/7] Activating conda rwkv environment...
✓ Environment activated successfully
[2/7] Reading model configuration...
✓ Model config: RWKV7-1.5B
✓ Model file: /path/to/model.pth
[3/7] Checking required files...
✓ Model file: ...
✓ Data file: examples/data/all_sharegpt_data_en_v2.jsonl
[4/7] Setting environment variables...
✓ CUDA_VISIBLE_DEVICES=6,7
✓ RWKV_V7_ON=1
[5/7] Creating output directory...
✓ Output directory: output_1.5b_20251020_090144
[6/7] Training configuration:
  Model: RWKV7-1.5B
  Parameters: 24 layers, 2048 dims, 65536 vocab
  LORA: att=96, gate=256, mv=64
  GPU: 6, 7 (2 GPUs)
  Data: all_sharegpt_data_en_v2.jsonl
  Context length: 1024
  Batch size: 2 per GPU
  Epochs: 1
  Learning rate: 1e-05
[7/7] Starting training...
...
Epoch 0:   7%|▋| 100/1442 [00:43<09:40, 2.31it/s, lr=1e-6, train_loss_step=2.380]
```

## Python API Usage Examples

### Example 1: Basic Usage

```python
from configs.model_loader import load_model_config

# Load configuration
config = load_model_config("1.5b")

# Access configuration info
print(f"Model name: {config.model_name}")
print(f"Layers: {config.n_layer}")
print(f"Hidden dim: {config.n_embd}")
print(f"Vocab size: {config.vocab_size}")
print(f"Model path: {config.model_path}")

# LORA dimensions
print(f"Attention LORA: {config.dim_att_lora}")
print(f"Gate LORA: {config.dim_gate_lora}")
print(f"MV LORA: {config.dim_mv_lora}")

# Recommended training config
print(f"Recommended context length: {config.recommended_ctx_len}")
print(f"Recommended batch size: {config.recommended_batch_size}")
print(f"Recommended learning rate: {config.recommended_lr}")
```

### Example 2: Create Training Configuration

```python
from configs.model_loader import load_model_config
from rwkvtune.models.rwkv7.config import SimpleTrainConfig

# Load model configuration
model_config = load_model_config("1.5b")

# Create training configuration
train_config = SimpleTrainConfig(
    load_model=model_config.model_path,
    n_layer=model_config.n_layer,
    n_embd=model_config.n_embd,
    vocab_size=model_config.vocab_size,
    head_size_a=model_config.head_size_a,
    dim_att_lora=model_config.dim_att_lora,
    dim_gate_lora=model_config.dim_gate_lora,
    dim_mv_lora=model_config.dim_mv_lora,
    ctx_len=model_config.recommended_ctx_len,
    micro_bsz=model_config.recommended_batch_size,
    lr_init=model_config.recommended_lr,
    data_file="examples/data/all_sharegpt_data_en_v2.jsonl",
)

print(f"Training config created: {train_config}")
```

### Example 3: Create and Load Model

```python
import torch
from configs.model_loader import load_model_config
from rwkvtune.models.rwkv7.config import SimpleTrainConfig
from rwkvtune.models.rwkv7.model import RWKV7Model

# Load configuration
model_config = load_model_config("1.5b")

# Create training configuration
train_config = SimpleTrainConfig(
    n_layer=model_config.n_layer,
    n_embd=model_config.n_embd,
    vocab_size=model_config.vocab_size,
    head_size_a=model_config.head_size_a,
    dim_att_lora=model_config.dim_att_lora,
    dim_gate_lora=model_config.dim_gate_lora,
    dim_mv_lora=model_config.dim_mv_lora,
)

# Create model
model = RWKV7Model(train_config)

# Load checkpoint
checkpoint = torch.load(model_config.model_path, map_location='cpu')
model.load_state_dict(checkpoint, strict=False)

print(f"✓ Model loaded successfully!")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

### Example 4: Iterate Over All Models

```python
from configs.model_loader import list_available_models, load_model_config

# Get all available models
available = list_available_models()

for model_size in sorted(available.keys()):
    config = load_model_config(model_size)
    print(f"{config.model_name}:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Dims: {config.n_embd}")
    print(f"  LORA: att={config.dim_att_lora}, gate={config.dim_gate_lora}")
    print(f"  Recommended batch_size: {config.recommended_batch_size}")
    print()
```

## Advanced Usage

### Custom Data Path

```bash
# Modify data path in the training script
vim examples/train_with_config.sh

# Find and modify this line:
# DATA_PATH="examples/data/your_custom_data.jsonl"
```

### Custom Training Parameters

Modify parameters in the training script:

```bash
python -m rwkvtune.cli.train \
    --load_model "$MODEL_PATH" \
    --n_layer $N_LAYER \
    --n_embd $N_EMBD \
    --vocab_size $VOCAB_SIZE \
    --head_size_a $HEAD_SIZE \
    --dim_att_lora $DIM_ATT_LORA \
    --dim_gate_lora $DIM_GATE_LORA \
    --dim_mv_lora $DIM_MV_LORA \
    --ctx_len 2048 \              # Custom: longer context
    --micro_bsz 4 \               # Custom: larger batch
    --epoch_count 5 \             # Custom: more epochs
    --lr_init 5e-5 \              # Custom: learning rate
    --data_file "your_data.jsonl" # Custom: data file
```

### Multi-GPU Training

```bash
# Modify GPU configuration in the training script
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs

# Modify training parameters
--devices 4 \           # Number of GPUs
--strategy ddp \        # Distributed training strategy
```

### Resume from Checkpoint

```bash
# Resume training from checkpoint
python -m rwkvtune.cli.train \
    --load_model "output/epoch_3.pth" \  # Start from saved checkpoint
    --epoch_begin 3 \                    # Starting epoch
    ... # Other parameters remain unchanged
```

## Performance Comparison

Training performance of different model scales (dual GPU 6,7):

| Model | Parameters | Training Speed | GPU Memory | Recommended Use Case |
|-------|------------|----------------|------------|------------------------|
| 0.1B | ~0.1B | 4.2 it/s | ~4GB | Quick testing, prototyping |
| 0.4B | ~0.4B | 3.5 it/s | ~8GB | Small-scale applications |
| 1.5B | ~1.5B | 2.3 it/s | ~16GB | Production |
| 2.9B | ~2.9B | 1.5 it/s | ~24GB | High-quality output |

## Troubleshooting

### 1. CUDA Out of Memory

Reduce batch size:
```bash
bash examples/train_with_config.sh 1.5b
# Edit recommended_batch_size in the config file
```

### 2. Slow Training Speed

- Ensure CUDA kernels are enabled: `RWKV_CUDA_ON=1`
- Use fewer data workers
- Use gradient checkpointing (for large models)

### 3. Loss Not Decreasing

- Check if the learning rate is appropriate
- Verify data format is correct
- Inspect mask behavior in data processing logs

## More Resources

- Config system docs: `configs/README.md`
- Model principles: `docs/RWKV7_Principles.md`
- Training script source: `examples/train_with_config.sh`
- Test script: `examples/test_model_configs.py`

