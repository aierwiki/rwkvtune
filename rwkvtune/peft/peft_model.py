"""
PEFT Model Utilities - inspired by peft-main design
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .config import LoraConfig
from .lora import LoraLinear


def get_peft_model(
    model: nn.Module,
    peft_config: LoraConfig,
) -> nn.Module:
    """
    Apply LoRA to a model - inspired by peft-main design
    
    Args:
        model: Base model (RWKV7Model)
        peft_config: LoRA configuration
    
    Returns:
        Model with LoRA applied (modified in-place)
    
    Example:
        ```python
        from rwkvtune import AutoModel
        from rwkvtune.peft import LoraConfig, get_peft_model
        
        model = AutoModel.from_pretrained("path/to/model")
        
        lora_config = LoraConfig(r=64, lora_alpha=128)
        model = get_peft_model(model, lora_config)
        
        # Now model has LoRA applied
        # Only LoRA parameters and modules_to_save are trainable
        ```
    """
    replaced_count = _replace_modules_with_lora(model, peft_config)
    
    if replaced_count == 0:
        import warnings
        warnings.warn(
            f"No matching target modules found for LoRA replacement. "
            f"target_modules={peft_config.target_modules}"
        )
    
    _freeze_base_model(model)
    _unfreeze_lora_params(model)
    _unfreeze_modules_to_save(model, peft_config)
    _add_peft_methods(model, peft_config)
    
    trainable_params, total_params = _count_parameters(model)
    print(f"LoRA applied:")
    print(f"  - Replaced {replaced_count} modules")
    print(f"  - Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def is_peft_model(model: nn.Module) -> bool:
    """Check if model has PEFT applied"""
    return getattr(model, '_is_peft_model', False)


def load_peft_model(
    model: nn.Module,
    peft_model_path: Union[str, Path],
    adapter_name: str = "default",
) -> nn.Module:
    """
    Load LoRA weights from directory (similar to peft.PeftModel.from_pretrained)
    
    Automatically handles:
    - Reading adapter_config.json or lora_config.json
    - Applying LoRA structure
    - Loading weights with device and dtype matching
    
    Args:
        model: Base model
        peft_model_path: LoRA weights directory (contains adapter_config.json and adapter_model.bin)
        adapter_name: Adapter name (default "default")
    
    Returns:
        Model with LoRA applied
    
    Example:
        ```python
        from rwkvtune import AutoModel
        from rwkvtune.peft import load_peft_model
        
        model = AutoModel.from_pretrained("path/to/base_model", device="cuda", dtype=torch.bfloat16)
        
        model = load_peft_model(model, "path/to/lora_checkpoint")
        ```
    """
    peft_model_path = Path(peft_model_path)
    
    if not peft_model_path.exists():
        raise ValueError(f"LoRA directory does not exist: {peft_model_path}")
    
    config_file = peft_model_path / "adapter_config.json"
    if not config_file.exists():
        config_file = peft_model_path / "lora_config.json"
    
    if not config_file.exists():
        raise ValueError(
            f"LoRA config file not found: {peft_model_path}\n"
            f"Requires adapter_config.json or lora_config.json"
        )
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    lora_config = LoraConfig(
        r=config_dict.get('r', 64),
        lora_alpha=config_dict.get('lora_alpha', 128),
        lora_dropout=config_dict.get('lora_dropout', 0.0),
        target_modules=config_dict.get('target_modules', None),
        modules_to_save=config_dict.get('modules_to_save', None),
        use_rslora=config_dict.get('use_rslora', False),
    )
    
    model = get_peft_model(model, lora_config)
    
    weights_file = peft_model_path / "adapter_model.bin"
    if not weights_file.exists():
        weights_file = peft_model_path / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = peft_model_path / "lora_weights.pth"
    
    if not weights_file.exists():
        raise ValueError(
            f"LoRA weights file not found: {peft_model_path}\n"
            f"Requires adapter_model.bin, adapter_model.safetensors or lora_weights.pth"
        )
    
    model.load_lora_weights(str(weights_file))
    
    return model


def _replace_modules_with_lora(model: nn.Module, config: LoraConfig) -> int:
    """
    Replace target modules with LoraLinear
    
    Returns:
        Number of replaced modules
    """
    target_modules = config.target_modules
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    replaced_count = 0
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if _is_target_module(name, target_modules):
                modules_to_replace.append((name, module))
    
    for name, module in modules_to_replace:
        parent = _get_parent_module(model, name)
        attr_name = name.split('.')[-1]
        
        lora_layer = LoraLinear(
            base_layer=module,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
        )
        
        setattr(parent, attr_name, lora_layer)
        replaced_count += 1
    
    return replaced_count


def _is_target_module(name: str, target_modules: List[str]) -> bool:
    """Check if module is a target module (name contains any target string)"""
    for target in target_modules:
        if target in name:
            return True
    return False


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """Get parent module of a named module"""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def _freeze_base_model(model: nn.Module):
    """Freeze all parameters"""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_lora_params(model: nn.Module):
    """Unfreeze LoRA parameters"""
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True


def _unfreeze_modules_to_save(model: nn.Module, config: LoraConfig):
    """Unfreeze modules_to_save"""
    if config.modules_to_save is None:
        return
    
    for name, param in model.named_parameters():
        for module_name in config.modules_to_save:
            if module_name in name:
                param.requires_grad = True
                break


def _count_parameters(model: nn.Module) -> tuple:
    """Count trainable and total parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def _add_peft_methods(model: nn.Module, config: LoraConfig):
    """Add PEFT helper methods and attributes"""
    model._peft_config = config
    model._is_peft_model = True
    
    def disable_adapter():
        """Disable all LoRA adapters"""
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.disable_adapter()
    
    def enable_adapter():
        """Enable all LoRA adapters"""
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.enable_adapter()
    
    def merge_adapter():
        """Merge all LoRA weights into base layers"""
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.merge()
    
    def get_merged_state_dict() -> Dict[str, torch.Tensor]:
        """
        Get merged state_dict (without LoRA structure)
        
        Used for saving merged complete model, automatically:
        - Removes lora_A and lora_B parameters
        - Removes base_layer prefix, restores original parameter names
        """
        state_dict = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                continue
            if '.base_layer.' in name:
                clean_name = name.replace('.base_layer.', '.')
            else:
                clean_name = name
            state_dict[clean_name] = param.data.clone()
        return state_dict
    
    def unmerge_adapter():
        """Remove all LoRA weights from base layers"""
        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.unmerge()
    
    def get_lora_state_dict() -> Dict[str, torch.Tensor]:
        """Get all LoRA parameters"""
        state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                state_dict[f"{name}.lora_A"] = module.lora_A.data.clone()
                state_dict[f"{name}.lora_B"] = module.lora_B.data.clone()
        return state_dict
    
    def save_lora_weights(path: str):
        """Save LoRA weights"""
        state_dict = get_lora_state_dict()
        torch.save({
            'lora_state_dict': state_dict,
            'config': {
                'r': config.r,
                'lora_alpha': config.lora_alpha,
                'lora_dropout': config.lora_dropout,
                'target_modules': config.target_modules,
                'modules_to_save': config.modules_to_save,
                'use_rslora': config.use_rslora,
            }
        }, path)
        print(f"LoRA weights saved to: {path}")
    
    def load_lora_weights(path: str):
        """
        Load LoRA weights
        
        Automatically handles device and dtype matching with base model.
        Supports adapter_model.bin and lora_weights.pth formats.
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('lora_state_dict', checkpoint)
        
        missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                a_key = f"{name}.lora_A"
                b_key = f"{name}.lora_B"
                
                if a_key in state_dict:
                    weight = state_dict[a_key].to(device=device, dtype=dtype)
                    module.lora_A.data.copy_(weight)
                else:
                    missing_keys.append(a_key)
                
                if b_key in state_dict:
                    weight = state_dict[b_key].to(device=device, dtype=dtype)
                    module.lora_B.data.copy_(weight)
                else:
                    missing_keys.append(b_key)
        
        if missing_keys:
            import warnings
            warnings.warn(f"Missing LoRA weights: {missing_keys}")
        
        print(f"LoRA weights loaded: {path} (device: {device}, dtype: {dtype})")
    
    model.disable_adapter = disable_adapter
    model.enable_adapter = enable_adapter
    model.merge_adapter = merge_adapter
    model.unmerge_adapter = unmerge_adapter
    model.get_lora_state_dict = get_lora_state_dict
    model.get_merged_state_dict = get_merged_state_dict
    model.save_lora_weights = save_lora_weights
    model.load_lora_weights = load_lora_weights
