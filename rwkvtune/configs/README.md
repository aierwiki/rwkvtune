# RWKV7 Model Configuration System

This directory contains configuration files for RWKV7 models of different scales, supporting automatic loading and training.

## Directory Structure

```
rwkvtune/configs/
├── models/               # Model configuration files
│   ├── rwkv7-0.1b.json
│   ├── rwkv7-0.4b.json
│   ├── rwkv7-1.5b.json
│   ├── rwkv7-2.9b.json
│   ├── rwkv7-7.2b.json
│   └── rwkv7-13.3b.json
├── model_loader.py       # Configuration loader
├── __init__.py          # Package init
└── README.md            # This document
```

## Available Models

| Model Scale | Layers | Hidden Dim | Parameters | Config File |
|-------------|--------|------------|------------|-------------|
| 0.1B | 12 | 768 | ~0.1B | rwkv7-0.1b.json |
| 0.4B | 24 | 1024 | ~0.4B | rwkv7-0.4b.json |
| 1.5B | 24 | 2048 | ~1.5B | rwkv7-1.5b.json |
| 2.9B | 32 | 2560 | ~2.9B | rwkv7-2.9b.json |
| 7.2B | 32 | 4096 | ~7.2B | rwkv7-7.2b.json |
| 13.3B | 61 | 4096 | ~13.3B | rwkv7-13.3b.json |

## Configuration File Format

Each configuration file contains the following information:

```json
{
  "model_name": "RWKV7-1.5B",
  "model_file": "rwkv7-g1-1.5b-20250429-ctx4096.pth",
  "description": "RWKV7 1.5B parameter model",
  
  "architecture": {
    "n_layer": 24,
    "n_embd": 2048,
    "vocab_size": 65536,
    "head_size_a": 64,
    "ctx_len": 4096
  },
  
  "lora_dims": {
    "dim_att_lora": 96,
    "dim_gate_lora": 256,
    "dim_mv_lora": 64
  },
  
  "training": {
    "recommended_ctx_len": 1024,
    "recommended_batch_size": 2,
    "recommended_lr": 1e-05
  }
}
```

## Usage

### 1. Training with Configuration Files

The simplest approach is to use the provided training script:

```bash
# Train 0.1B model
bash examples/train_with_config.sh 0.1b

# Train 1.5B model
bash examples/train_with_config.sh 1.5b

# Train 2.9B model  
bash examples/train_with_config.sh 2.9b
```

### 2. Using in Python Code

```python
from rwkvtune.configs.model_loader import load_model_config

# Load 1.5B model configuration
config = load_model_config("1.5b")

print(f"Model: {config.model_name}")
print(f"Layers: {config.n_layer}")
print(f"Hidden dim: {config.n_embd}")
print(f"Model path: {config.model_path}")
```

### 3. List All Available Models

```python
from rwkvtune.configs.model_loader import list_available_models

available = list_available_models()
for key, path in available.items():
    print(f"{key}: {path}")
```

### 4. Test All Model Configurations

Run the test script to verify all models can be loaded correctly:

```bash
python examples/test_model_configs.py
```

## LORA Dimension Guidelines

Different model scales use different LORA dimensions for optimal performance:

- **0.1B/0.4B models**: 
  - dim_att_lora: 64
  - dim_gate_lora: 128
  - dim_mv_lora: 32

- **1.5B/2.9B models**:
  - dim_att_lora: 96
  - dim_gate_lora: 256/320
  - dim_mv_lora: 64

These dimensions are auto-detected from the actual checkpoint parameters to ensure dimension matching when loading models.

## Recommended Training Configuration

Each model has recommended training hyperparameters:

| Model | Context Length | Batch Size | Learning Rate |
|-------|----------------|------------|---------------|
| 0.1B | 1024 | 8 | 3e-4 |
| 0.4B | 1024 | 4 | 1e-4 |
| 1.5B | 1024 | 2 | 1e-5 |
| 2.9B | 1024 | 1 | 5e-6 |
| 7.2B | 1024 | 1 | 1e-6 |
| 13.3B | 512 | 1 | 1e-6 |

Note: These are recommended settings for dual-GPU (6,7) training. Batch size may need adjustment for single-GPU training.

## Adding New Models

To add a new model configuration:

1. Analyze the model checkpoint to obtain parameters:
```python
import torch
ckpt = torch.load("model.pth", map_location='cpu')
# Inspect n_layer, n_embd, LORA dimensions, etc.
```

2. Create a new configuration file `configs/models/rwkv7-xxx.json`

3. Run the test to verify:
```bash
python examples/test_model_configs.py
```

## Troubleshooting

### Dimension Mismatch Error

If you encounter an error like:
```
size mismatch for model.blocks.0.att.w1: copying a param with shape torch.Size([2048, 96])
```

This indicates that the LORA dimensions in the config file do not match the checkpoint. Please:

1. Run the analysis script to obtain correct dimensions
2. Update the `lora_dims` section in the config file
3. Re-run the test

### Model File Not Found

Ensure the model file is in the correct location:
```
/path/to/model.pth
```

If the model is elsewhere, you can modify the `model_file` path in the config file.

## Related Files

- `examples/train_with_config.sh` - Training script using configuration files
- `examples/test_model_configs.py` - Test all model configurations
- `rwkvtune/models/rwkv7/model.py` - Model code with LORA dimension support
- `rwkvtune/models/rwkv7/config.py` - Training configuration class definition

