"""
Model Hub Utilities - Create standard model directory structure

Provides utility functions to convert existing RWKV configs and weights
into standard Transformers-style model directory structure.
"""

import os
import json
import shutil
from typing import Optional, Dict, Any
from pathlib import Path


def create_model_hub(
    output_dir: str,
    model_file: str,
    config_name: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    vocab_file: Optional[str] = None,
    chat_template_file: Optional[str] = None,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
    # Model architecture params
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    vocab_size: Optional[int] = None,
    head_size_a: Optional[int] = None,
    ctx_len: Optional[int] = None,
    # LORA dimensions
    dim_att_lora: Optional[int] = None,
    dim_gate_lora: Optional[int] = None,
    dim_mv_lora: Optional[int] = None,
    # Generation config
    temperature: float = 1.0,
    top_p: float = 0.85,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    max_length: int = 1024,
    # Tokenizer config
    eos_token: str = "\n\n",
    model_max_length: int = 4096,
    # Other options
    copy_weights: bool = True,
    save_format: str = "pth",
    overwrite: bool = False,
):
    """
    Create standard model directory structure
    
    Generates a standard model directory from existing RWKV7 config and weights, including:
    - config.json: Model configuration
    - generation_config.json: Generation configuration
    - tokenizer_config.json: Tokenizer configuration
    - vocab.txt: Vocabulary file (if provided)
    - chat_template.jinja: Chat template (if provided)
    - model.pth / model.safetensors: Weight file
    - README.md: Model documentation
    
    Args:
        output_dir: Output directory path
        model_file: Model weights file path
        config_name: Predefined config name (e.g. "rwkv7-0.1b")
        config_dict: Custom config dictionary (overrides config_name if provided)
        vocab_file: Vocabulary file path (uses default vocab if not provided)
        chat_template_file: Chat template file path (.jinja format)
        model_name: Model name (for README)
        description: Model description (for README)
        
        # Following params override config file values
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        vocab_size: Vocabulary size
        head_size_a: Attention head size
        ctx_len: Context length
        dim_att_lora: Attention LORA dimension
        dim_gate_lora: Gate LORA dimension
        dim_mv_lora: MV LORA dimension
        
        # Generation config params
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        max_length: Max generation length
        
        # Tokenizer config
        eos_token: EOS token
        model_max_length: Tokenizer max length
        
        # Other options
        copy_weights: Whether to copy weight file (False creates symlink)
        save_format: Save format ("pth" or "safetensors")
        overwrite: Whether to overwrite existing directory
    
    Returns:
        None
    
    Example:
        >>> from rwkvtune.utils.model_hub import create_model_hub
        >>> 
        >>> # Use predefined config
        >>> create_model_hub(
        ...     output_dir="models/my-rwkv7-0.1b",
        ...     model_file="checkpoints/rwkv7-0.1b.pth",
        ...     config_name="rwkv7-0.1b",
        ...     model_name="My RWKV7 0.1B Model",
        ...     description="A fine-tuned RWKV7 0.1B model"
        ... )
        >>> 
        >>> # Use custom config
        >>> create_model_hub(
        ...     output_dir="models/my-custom-model",
        ...     model_file="checkpoints/custom.pth",
        ...     n_layer=12,
        ...     n_embd=768,
        ...     vocab_size=65536,
        ...     head_size_a=64,
        ...     ctx_len=4096,
        ...     model_name="Custom RWKV7 Model"
        ... )
    """
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Directory already exists: {output_dir}\n"
            f"Use overwrite=True to overwrite existing directory"
        )
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Create model config
    print("[INFO] Creating model config...")
    config = _create_model_config(
        config_name=config_name,
        config_dict=config_dict,
        model_file=model_file,
        n_layer=n_layer,
        n_embd=n_embd,
        vocab_size=vocab_size,
        head_size_a=head_size_a,
        ctx_len=ctx_len,
        dim_att_lora=dim_att_lora,
        dim_gate_lora=dim_gate_lora,
        dim_mv_lora=dim_mv_lora,
    )
    
    config_file = output_path / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Created: {config_file}")
    
    # 2. Create generation config
    print("[INFO] Creating generation config...")
    gen_config = {
        "bos_token_id": None,
        "eos_token_id": 261,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_length": max_length,
        "transformers_version": "rwkvtune-0.1.0"
    }
    
    gen_config_file = output_path / "generation_config.json"
    with open(gen_config_file, 'w', encoding='utf-8') as f:
        json.dump(gen_config, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Created: {gen_config_file}")
    
    # 3. Create tokenizer config
    print("[INFO] Creating tokenizer config...")
    tokenizer_config = {
        "tokenizer_class": "RWKV_TOKENIZER",
        "model_max_length": model_max_length,
        "bos_token": None,
        "eos_token": eos_token,
        "pad_token": None,
        "unk_token": None,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
    }
    
    # Process chat_template
    chat_template_content = None
    
    if chat_template_file and os.path.exists(chat_template_file):
        print("   Using custom chat_template")
        with open(chat_template_file, 'r', encoding='utf-8') as f:
            chat_template_content = f.read()
    else:
        default_template_path = os.path.join(
            os.path.dirname(__file__), 
            '../configs/chat_templates/rwkv_default.jinja'
        )
        if os.path.exists(default_template_path):
            print("   Using default RWKV chat_template")
            with open(default_template_path, 'r', encoding='utf-8') as f:
                chat_template_content = f.read()
        else:
            print("   [WARN] Default chat_template not found")
    
    if chat_template_content:
        tokenizer_config["chat_template"] = chat_template_content
    
    tokenizer_config_file = output_path / "tokenizer_config.json"
    with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Created: {tokenizer_config_file}")
    
    # 4. Copy vocabulary file
    if vocab_file and os.path.exists(vocab_file):
        print("[INFO] Copying vocabulary file...")
        vocab_dest = output_path / "vocab.txt"
        shutil.copy(vocab_file, vocab_dest)
        print(f"   [OK] Created: {vocab_dest}")
    else:
        print("[INFO] No vocabulary file provided, vocab path needed when loading")
    
    # 5. Copy/link weight file
    if model_file and os.path.exists(model_file):
        print("[INFO] Processing model weight file...")
        
        if save_format == "safetensors":
            weight_dest = output_path / "model.safetensors"
        else:
            weight_dest = output_path / "model.pth"
        
        source_is_safetensors = model_file.endswith('.safetensors')
        target_is_safetensors = save_format == "safetensors"
        
        if source_is_safetensors == target_is_safetensors:
            if copy_weights:
                shutil.copy(model_file, weight_dest)
                print(f"   [OK] Copied: {weight_dest}")
            else:
                if weight_dest.exists():
                    weight_dest.unlink()
                weight_dest.symlink_to(os.path.abspath(model_file))
                print(f"   [OK] Linked: {weight_dest} -> {model_file}")
        else:
            print(f"   [INFO] Converting format: {model_file} -> {weight_dest}")
            _convert_weight_format(model_file, weight_dest)
            print(f"   [OK] Converted: {weight_dest}")
    else:
        print("[WARN] No valid model weight file provided")
    
    # 6. Create README.md
    print("[INFO] Creating README.md...")
    readme_content = _create_readme(
        model_name=model_name or config.get("model_type", "RWKV7"),
        description=description,
        config=config,
        gen_config=gen_config,
    )
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   [OK] Created: {readme_file}")
    
    print(f"\n[OK] Model directory created: {output_dir}")
    print(f"\nUsage:")
    print(f"```python")
    print(f"from rwkvtune import AutoModel, AutoTokenizer")
    print(f"")
    print(f'model = AutoModel.from_pretrained("{output_dir}")')
    print(f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")')
    print(f"```")


def _extract_config_from_model(model_file: str) -> Optional[Dict[str, Any]]:
    """Extract config from model weight file"""
    try:
        import torch
        
        state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        
        config = {}
        
        if "emb.weight" in state_dict:
            emb_weight = state_dict["emb.weight"]
            config["vocab_size"] = emb_weight.shape[0]
            config["n_embd"] = emb_weight.shape[1]
        
        layer_keys = [k for k in state_dict.keys() if k.startswith("blocks.")]
        if layer_keys:
            layer_nums = set()
            for k in layer_keys:
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_nums.add(int(parts[1]))
            if layer_nums:
                config["n_layer"] = max(layer_nums) + 1
        
        for k in state_dict.keys():
            if "att.r_k" in k:
                config["head_size_a"] = state_dict[k].shape[-1]
                break
        
        for k in state_dict.keys():
            if "att.w1" in k:
                config["dim_att_lora"] = state_dict[k].shape[-1]
                break
        
        for k in state_dict.keys():
            if "att.g1" in k:
                config["dim_gate_lora"] = state_dict[k].shape[-1]
                break
        
        for k in state_dict.keys():
            if "att.v1" in k:
                config["dim_mv_lora"] = state_dict[k].shape[-1]
                break
        
        return config if config else None
        
    except Exception as e:
        print(f"   [WARN] Cannot extract config from model weights: {e}")
        return None


def _create_model_config(
    config_name: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    model_file: Optional[str] = None,
    **overrides
) -> Dict[str, Any]:
    """Create model config dictionary"""
    
    if config_dict:
        config = config_dict.copy()
    
    elif config_name:
        from rwkvtune.configs.model_loader import load_model_config
        try:
            model_config = load_model_config(config_name)
            config = {
                "n_layer": model_config.n_layer,
                "n_embd": model_config.n_embd,
                "vocab_size": model_config.vocab_size,
                "head_size_a": model_config.head_size_a,
                "ctx_len": model_config.ctx_len,
                "dim_att_lora": model_config.dim_att_lora,
                "dim_gate_lora": model_config.dim_gate_lora,
                "dim_mv_lora": model_config.dim_mv_lora,
            }
        except Exception as e:
            print(f"[WARN] Cannot load predefined config '{config_name}': {e}")
            if model_file:
                print("   Attempting to extract config from model weights...")
                extracted_config = _extract_config_from_model(model_file)
                if extracted_config:
                    print("   [OK] Successfully extracted config from model weights")
                    config = extracted_config
                else:
                    print("   Using default config...")
                    config = {}
            else:
                print("   Using default config...")
                config = {}
    
    else:
        if model_file:
            print("   Extracting config from model weights...")
            extracted_config = _extract_config_from_model(model_file)
            if extracted_config:
                print("   [OK] Successfully extracted config from model weights")
                config = extracted_config
            else:
                config = {}
        else:
            config = {}
    
    default_config = {
        "model_type": "rwkv7",
        "architectures": ["RWKV7Model"],
        "n_layer": 12,
        "n_embd": 768,
        "vocab_size": 65536,
        "head_size_a": 64,
        "ctx_len": 4096,
        "dim_att_lora": 64,
        "dim_gate_lora": 128,
        "dim_mv_lora": 32,
        "bos_token_id": None,
        "eos_token_id": 261,
        "pad_token_id": 0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "my_testing": "x070",
        "transformers_version": "rwkvtune-0.1.0",
    }
    
    final_config = {**default_config, **config}
    
    for key, value in overrides.items():
        if value is not None:
            final_config[key] = value
    
    if "dim_att" not in final_config or final_config["dim_att"] is None:
        final_config["dim_att"] = final_config["n_embd"]
    
    if "dim_ffn" not in final_config or final_config["dim_ffn"] is None:
        final_config["dim_ffn"] = int((final_config["n_embd"] * 3.5) // 32 * 32)
    
    if "head_size_divisor" not in final_config:
        final_config["head_size_divisor"] = 8
    
    return final_config


def _convert_weight_format(source_file: str, target_file: str):
    """Convert weight file format"""
    import torch
    
    source_is_safetensors = source_file.endswith('.safetensors')
    target_is_safetensors = str(target_file).endswith('.safetensors')
    
    if source_is_safetensors:
        try:
            from safetensors.torch import load_file
            state_dict = load_file(source_file)
        except ImportError:
            raise ImportError(
                "safetensors is required to convert weight format:\n"
                "  pip install safetensors"
            )
    else:
        state_dict = torch.load(source_file, map_location='cpu')
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
    
    if target_is_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(state_dict, str(target_file))
        except ImportError:
            raise ImportError(
                "safetensors is required to save weights:\n"
                "  pip install safetensors"
            )
    else:
        torch.save(state_dict, str(target_file))


def _create_readme(
    model_name: str,
    description: Optional[str],
    config: Dict[str, Any],
    gen_config: Dict[str, Any],
) -> str:
    """Create README.md content"""
    
    readme = f"""# {model_name}

"""
    
    if description:
        readme += f"{description}\n\n"
    
    readme += f"""## Model Information

- **Model Type**: {config.get('model_type', 'RWKV7')}
- **Architecture**: {config.get('architectures', ['RWKV7Model'])[0]}
- **Layers**: {config.get('n_layer', 'N/A')}
- **Embedding Dimension**: {config.get('n_embd', 'N/A')}
- **Vocabulary Size**: {config.get('vocab_size', 'N/A')}
- **Context Length**: {config.get('ctx_len', 'N/A')}

## Usage

### Load Model and Tokenizer

```python
from rwkvtune import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained(".")
tokenizer = AutoTokenizer.from_pretrained(".")

# Generate text
input_text = "Hello, "
input_ids = tokenizer.encode(input_text)

# Use model for inference
# ... (add inference code as needed)
```

### Chat Mode

```python
messages = [
    {{"role": "user", "content": "Hello!"}},
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=False
)
print(prompt)
```

## Generation Parameters

Default generation parameters:

- **Temperature**: {gen_config.get('temperature', 1.0)}
- **Top-p**: {gen_config.get('top_p', 0.85)}
- **Top-k**: {gen_config.get('top_k', 0)}
- **Repetition Penalty**: {gen_config.get('repetition_penalty', 1.0)}
- **Max Length**: {gen_config.get('max_length', 1024)}

## License

Please follow RWKV model license requirements.

## Citation

If you use this model, please cite the RWKV project:

```bibtex
@misc{{rwkv,
  title={{RWKV: Reinventing RNNs for the Transformer Era}},
  author={{Bo Peng}},
  year={{2023}},
  howpublished={{\\url{{https://github.com/BlinkDL/RWKV-LM}}}}
}}
```
"""
    
    return readme
