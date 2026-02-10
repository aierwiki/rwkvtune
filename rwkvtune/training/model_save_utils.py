"""
Model Save Utilities - for saving models during training.

Provides unified interfaces for saving:
- Complete model weights (.pth)
- LoRA weights (adapter_model.bin / lora_weights.pth)
- Config files (config.json / adapter_config.json)
- Tokenizer files (optional)

Following HuggingFace PEFT best practices.
"""
import os
import json
import torch
import shutil
from pathlib import Path
from typing import Optional, Union, Any, Dict

ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
LORA_WEIGHTS_NAME = "lora_weights.pth"
LORA_CONFIG_NAME = "lora_config.json"


def save_model_with_config(
    model: torch.nn.Module,
    save_directory: str,
    model_name: str = "model",
    tokenizer: Optional[Any] = None,
    model_config_obj: Optional[Any] = None,
    save_format: str = "pth",
    is_main_process: bool = True,
):
    """Save complete model structure (config.json, weights, tokenizer)."""
    try:
        if isinstance(model, dict):
            state_dict = model
        elif hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            raise ValueError(f"Cannot get state_dict from {type(model)}")
    except Exception as e:
        if is_main_process:
            print(f"Error: Failed to get state_dict: {e}")
        raise

    if not is_main_process:
        return None

    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config
    if model_config_obj is not None or hasattr(model, 'config'):
        try:
            config = model_config_obj if model_config_obj is not None else model.config
            if hasattr(config, 'save_pretrained'):
                config.save_pretrained(str(save_path))
                print(f"Config saved: {save_path / 'config.json'}")
            elif isinstance(config, dict):
                config_file = save_path / "config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print(f"Config saved: {config_file}")
            else:
                config_dict = {
                    'n_layer': getattr(config, 'n_layer', None),
                    'n_embd': getattr(config, 'n_embd', None),
                    'vocab_size': getattr(config, 'vocab_size', None),
                    'head_size_a': getattr(config, 'head_size_a', None),
                    'ctx_len': getattr(config, 'ctx_len', None),
                    'model_type': 'rwkv7',
                }
                config_dict = {k: v for k, v in config_dict.items() if v is not None}
                config_file = save_path / "config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                print(f"Config saved: {config_file}")
        except Exception as e:
            print(f"Warning: Cannot save config file: {e}")

    # Save weights
    try:
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                weight_file = save_path / f"{model_name}.safetensors"
                save_file(state_dict, str(weight_file))
            except ImportError:
                weight_file = save_path / f"{model_name}.pth"
                torch.save(state_dict, str(weight_file))
        else:
            weight_file = save_path / f"{model_name}.pth"
            torch.save(state_dict, str(weight_file))
        print(f"Weights saved: {weight_file}")
    except Exception as e:
        print(f"Error: Failed to save weights: {e}")
        raise

    # Save tokenizer
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(str(save_path))
                print(f"Tokenizer saved: {save_path}")
        except Exception as e:
            print(f"Warning: Cannot save Tokenizer: {e}")

    print(f"\nModel saved to: {save_directory}")
    return str(weight_file)


def save_model_checkpoint(
    lightning_module,
    save_directory: str,
    checkpoint_name: str,
    tokenizer: Optional[Any] = None,
    model_path: Optional[str] = None,
):
    """Save training checkpoint."""
    checkpoint_dir = Path(save_directory) / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config_obj = None
    if hasattr(lightning_module, 'model') and hasattr(lightning_module.model, 'config'):
        model_config_obj = lightning_module.model.config
    elif hasattr(lightning_module, 'config'):
        model_config_obj = lightning_module.config

    return save_model_with_config(
        model=lightning_module,
        save_directory=str(checkpoint_dir),
        model_name="model",
        tokenizer=tokenizer,
        model_config_obj=model_config_obj,
        save_format="pth",
    )


def get_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Get LoRA parameter state_dict."""
    state_dict = model.state_dict()
    lora_state_dict = {k: v for k, v in state_dict.items() if 'lora_' in k.lower()}
    cleaned_state_dict = {}
    for k, v in lora_state_dict.items():
        clean_key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned_state_dict[clean_key] = v
    return cleaned_state_dict


def save_lora_checkpoint(
    model: torch.nn.Module,
    save_directory: str,
    lora_config: Optional[Dict] = None,
    base_model_path: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    safe_serialization: bool = False,
    is_main_process: bool = True,
):
    """Save LoRA weights and config."""
    try:
        lora_state_dict = get_lora_state_dict(model)
    except Exception as e:
        if is_main_process:
            print(f"Error: Failed to get LoRA state_dict: {e}")
        raise

    if not is_main_process:
        return None

    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    if not lora_state_dict:
        print(f"Warning: No LoRA parameters found")
        return None

    total_params = sum(p.numel() for p in lora_state_dict.values())
    print(f"  LoRA parameters: {total_params / 1e6:.2f}M")

    # Save weights
    if safe_serialization:
        try:
            from safetensors.torch import save_file
            weight_file = save_path / ADAPTER_SAFE_WEIGHTS_NAME
            save_file(lora_state_dict, str(weight_file))
        except ImportError:
            weight_file = save_path / ADAPTER_WEIGHTS_NAME
            torch.save(lora_state_dict, str(weight_file))
    else:
        weight_file = save_path / ADAPTER_WEIGHTS_NAME
        torch.save(lora_state_dict, str(weight_file))

    file_size_mb = weight_file.stat().st_size / (1024 * 1024)
    print(f"LoRA weights saved: {weight_file} ({file_size_mb:.1f} MB)")

    # Save config
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_path,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
    }
    if lora_config:
        adapter_config.update({
            "r": lora_config.get('r', 64),
            "lora_alpha": lora_config.get('lora_alpha', 128),
            "lora_dropout": lora_config.get('lora_dropout', 0.0),
            "target_modules": lora_config.get('target_modules', None),
            "modules_to_save": lora_config.get('modules_to_save', None),
            "use_rslora": lora_config.get('use_rslora', False),
        })

    config_file = save_path / ADAPTER_CONFIG_NAME
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    print(f"LoRA config saved: {config_file}")

    # Save legacy format
    legacy_config_file = save_path / LORA_CONFIG_NAME
    with open(legacy_config_file, 'w', encoding='utf-8') as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    # Save tokenizer
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, 'save_pretrained'):
                tokenizer.save_pretrained(str(save_path))
                print(f"Tokenizer saved: {save_path}")
        except Exception as e:
            print(f"Warning: Cannot save Tokenizer: {e}")

    print(f"\nLoRA checkpoint saved to: {save_directory}")
    return str(weight_file)


def save_checkpoint_with_lora_support(
    lightning_module,
    save_directory: str,
    checkpoint_name: str,
    tokenizer: Optional[Any] = None,
    model_path: Optional[str] = None,
    use_lora: bool = False,
    lora_save_mode: str = "lora_only",
    lora_config: Optional[Dict] = None,
    is_main_process: bool = True,
):
    """
    Save checkpoint with LoRA support.
    
    lora_save_mode options:
    - lora_only: Only LoRA weights (recommended)
    - full: Complete merged model
    - both: Save both
    """
    checkpoint_dir = Path(save_directory) / checkpoint_name
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    model = lightning_module.model if hasattr(lightning_module, 'model') else lightning_module

    if use_lora:
        if lora_save_mode in ["lora_only", "both"]:
            lora_file = save_lora_checkpoint(
                model=lightning_module,
                save_directory=str(checkpoint_dir),
                lora_config=lora_config,
                base_model_path=model_path,
                tokenizer=tokenizer if lora_save_mode == "lora_only" else None,
                is_main_process=is_main_process,
            )
            if lora_file:
                saved_files.append(lora_file)

        if lora_save_mode in ["full", "both"]:
            if is_main_process:
                print(f"  Saving complete model (merged LoRA)...")

            if hasattr(model, 'merge_adapter'):
                model.merge_adapter()

            model_config_obj = model.config if hasattr(model, 'config') else None
            weight_file = save_model_with_config(
                model=lightning_module,
                save_directory=str(checkpoint_dir),
                model_name="model",
                tokenizer=tokenizer,
                model_config_obj=model_config_obj,
                is_main_process=is_main_process,
            )
            saved_files.append(weight_file)

            if hasattr(model, 'unmerge_adapter'):
                model.unmerge_adapter()
    else:
        model_config_obj = model.config if hasattr(model, 'config') else None
        weight_file = save_model_with_config(
            model=lightning_module,
            save_directory=str(checkpoint_dir),
            model_name="model",
            tokenizer=tokenizer,
            model_config_obj=model_config_obj,
            is_main_process=is_main_process,
        )
        saved_files.append(weight_file)

    return saved_files
