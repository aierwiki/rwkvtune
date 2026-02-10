# RWKV Model Configuration System

## Overview

A flexible configuration system that supports multiple RWKV architecture versions (rwkv7, rwkv8, etc.) and multiple configuration loading methods.

## Configuration File Format

```json
{
  "architecture_version": "rwkv7",
  "model_name": "RWKV7-1.5B",
  "description": "RWKV7 1.5B parameter model",
  
  "model_file": "/path/to/model.pth",
  
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

### Field Descriptions

#### Required Fields

- `architecture_version`: Architecture version identifier (e.g., "rwkv7", "rwkv8")
- `architecture`: Model architecture parameters
  - `n_layer`: Number of layers
  - `n_embd`: Embedding dimension
  - `vocab_size`: Vocabulary size
  - `head_size_a`: Attention head size
- `lora_dims`: LORA dimension configuration
  - `dim_att_lora`: Attention LORA dimension
  - `dim_gate_lora`: Gate LORA dimension
  - `dim_mv_lora`: MV LORA dimension

#### Optional Fields

- `model_file`: Path to model weight file (optional, can be overridden via command line)
- `model_name`: Model display name
- `description`: Model description
- `training`: Recommended training parameters

## Three Configuration Loading Methods

### Method 1: Short Name

The most concise approach; automatically adds the default architecture prefix.

```python
from rwkvtune.configs.model_loader import load_model_config

config = load_model_config("1.5b")
# Automatically looks up: rwkv7-1.5b.json
```

**Shell script usage:**
```bash
MODEL_CONFIG="1.5b"
```

### Method 2: Full Config Name

Explicitly specify the config file name (without .json extension).

```python
config = load_model_config("rwkv7-0.1b")
# Looks up: rwkv7-0.1b.json
```

**Shell script usage:**
```bash
MODEL_CONFIG="rwkv7-0.1b"
```

### Method 3: Configuration File Path

Use the full path to a custom configuration file.

```python
config = load_model_config("/path/to/my_config.json")
# Or relative path
config = load_model_config("./configs/custom.json")
```

**Shell script usage:**
```bash
MODEL_CONFIG="/path/to/my_config.json"
```

## Model Weight Path Priority

The configuration system supports flexible model weight specification:

1. **Highest priority**: Command line argument `--load_model`
2. **Next priority**: `model_file` in the configuration file
3. **At least one must be specified**; otherwise an error is raised

### Scenario Examples

#### Scenario 1: Use default path from configuration file

```json
// rwkv7-1.5b.json
{
  "model_file": "/path/to/model.pth",
  ...
}
```

```bash
# Shell script
MODEL_CONFIG="1.5b"
MODEL_PATH=""  # Leave empty to use path from config file
```

#### Scenario 2: Override path in configuration file

```bash
# Shell script
MODEL_CONFIG="1.5b"
MODEL_PATH="/custom/path/to/rwkv7-g1a-1.5b-custom.pth"  # Override
```

#### Scenario 3: Config file has no path; specify via command line

```json
// custom_config.json
{
  "model_file": null,  // Or omit this field
  ...
}
```

```bash
# Shell script
MODEL_CONFIG="./custom_config.json"
MODEL_PATH="/path/to/model.pth"  # Must specify
```

## Built-in Configuration Files

Located in `rwkvtune/configs/models/`:

| Config File | Short Name | Full Name | Architecture | Scale |
|-------------|------------|-----------|--------------|-------|
| rwkv7-0.1b.json | 0.1b | rwkv7-0.1b | rwkv7 | 12 layers, 768 dims |
| rwkv7-0.4b.json | 0.4b | rwkv7-0.4b | rwkv7 | 24 layers, 1024 dims |
| rwkv7-1.5b.json | 1.5b | rwkv7-1.5b | rwkv7 | 24 layers, 2048 dims |
| rwkv7-2.9b.json | 2.9b | rwkv7-2.9b | rwkv7 | 32 layers, 2560 dims |
| rwkv7-7.2b.json | 7.2b | rwkv7-7.2b | rwkv7 | 32 layers, 4096 dims |

## Extending Support for New Architectures (e.g., RWKV8)

### 1. Create Configuration File

```json
// rwkvtune/configs/models/rwkv8-1.0b.json
{
  "architecture_version": "rwkv8",
  "model_name": "RWKV8-1.0B",
  "model_file": "/path/to/rwkv8-1.0b.pth",
  
  "architecture": {
    "n_layer": 20,
    "n_embd": 1536,
    ...
  },
  ...
}
```

### 2. Use the New Configuration

```bash
# Method 1: Short name (requires default_arch parameter)
python -m rwkvtune.cli.train --model_config "1.0b" ...

# Method 2: Full name
python -m rwkvtune.cli.train --model_config "rwkv8-1.0b" ...

# Method 3: File path
python -m rwkvtune.cli.train --model_config "./configs/models/rwkv8-1.0b.json" ...
```

## Training Script Integration

### Shell Script Configuration

```bash
# examples/train_rwkv7_multi_gpu.sh

# Method 1: Short name
MODEL_CONFIG="1.5b"
MODEL_PATH=""

# Method 2: Full name
MODEL_CONFIG="rwkv7-0.1b"
MODEL_PATH="/custom/path/model.pth"

# Method 3: Custom configuration
MODEL_CONFIG="/path/to/my_config.json"
MODEL_PATH="/path/to/my_model.pth"
```

### Python API

```python
from rwkvtune.configs.model_loader import load_model_config, print_model_info

# Load configuration
config = load_model_config("1.5b")

# View configuration info
print_model_info(config)

# Get parameters
print(f"Layers: {config.n_layer}")
print(f"Dims: {config.n_embd}")
print(f"Model path: {config.model_path}")

# Convert to training configuration
train_params = config.to_train_config_dict()
```

## Configuration Validation and Error Handling

### Error Messages

When configuration loading fails, the system provides detailed error information:

```
❌ Error: Unable to load model configuration 'unknown-model'
   Configuration not found: 'unknown-model'
   Available configs: ['0.1b', '0.4b', '1.5b', '2.9b', '7.2b', 'rwkv7-0.1b', ...]
   Or provide the full path to a configuration file

💡 Three configuration methods are supported:
   1. Short name: 0.1b, 0.4b, 1.5b, 2.9b, 7.2b
   2. Full name: rwkv7-0.1b, rwkv7-1.5b, ...
   3. File path: /path/to/config.json
```

### Weight Path Verification

```
✓ Using model config: RWKV7-1.5B (rwkv7)
✓ Config source: builtin:1.5b
✓ Model weights: /path/to/model.pth
  (Overriding config file: /old/path/to/model.pth)
✓ Architecture params: 24 layers, 2048 dims, 65536 vocab
✓ LORA dims: att=96, gate=256, mv=64
✓ Training params: ctx_len=1024, batch_size=2, lr=1e-05
```

## Best Practices

1. **Standard models**: Use short names (`0.1b`, `1.5b`, etc.)
2. **Custom weights**: Use built-in config + custom MODEL_PATH
3. **Fully custom**: Create custom config file + custom weight path
4. **Version management**: Include architecture version in config file names (`rwkv7-`, `rwkv8-`, etc.)
5. **Documentation**: Add a `description` field in config files to explain usage

## API Reference

### load_model_config()

```python
def load_model_config(
    config_identifier: str,
    config_dir: Optional[str] = None,
    default_arch: str = "rwkv7"
) -> ModelConfig
```

### ModelConfig Class

```python
class ModelConfig:
    # Architecture info
    architecture_version: str
    model_name: str
    description: str
    
    # Model file
    model_path: Optional[str]
    
    # Architecture parameters
    n_layer: int
    n_embd: int
    vocab_size: int
    head_size_a: int
    ctx_len: int
    
    # LORA dimensions
    dim_att_lora: int
    dim_gate_lora: int
    dim_mv_lora: int
    
    # Training recommendations
    recommended_ctx_len: int
    recommended_batch_size: int
    recommended_lr: float
    
    def to_train_config_dict() -> Dict[str, Any]
```

## Related Documentation

- [Training Script Usage](../../examples/TRAIN_SCRIPT_USAGE.md)
- [Model Configuration Directory](./models/)
- [Config Loader Source](./model_loader.py)

