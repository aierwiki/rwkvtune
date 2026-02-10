"""
AutoModel - Automatic RWKV model loading
Inspired by Transformers AutoModel design
"""

import os
import torch
from typing import Optional, Union
from .configuration import RWKVConfig


MODEL_MAPPING = {
    "rwkv7": "RWKV7Model",
}


class AutoModel:
    """
    AutoModel - Automatically select and load the appropriate RWKV model.
    
    Selects the corresponding model class based on the model_type field in config.
    
    Usage:
        # Load from directory
        model = AutoModel.from_pretrained("path/to/model")
        
        # Load from config and weights
        config = RWKVConfig.from_json_file("config.json")
        model = AutoModel.from_config(config)
        model.load_state_dict(torch.load("model.pth"))
    """
    
    @staticmethod
    def _get_model_class(model_type: str):
        """
        Get the model class for the given model_type.
        
        Args:
            model_type: Model type (e.g., "rwkv7")
            
        Returns:
            Model class
        """
        if model_type not in MODEL_MAPPING:
            raise ValueError(
                f"Unsupported model type: {model_type}\n"
                f"Supported model types: {list(MODEL_MAPPING.keys())}"
            )
        
        model_class_name = MODEL_MAPPING[model_type]
        
        if model_type == "rwkv7":
            from rwkvtune.models.rwkv7 import RWKV7Model
            return RWKV7Model
        
        raise NotImplementedError(f"Model type {model_type} is not yet implemented")
    
    @classmethod
    def from_config(cls, config: RWKVConfig, **kwargs):
        """
        Create model from config (without loading weights).
        
        Args:
            config: RWKV config object
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Model instance
        """
        model_class = cls._get_model_class(config.model_type)
        return model_class(config, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Load model from path (supports directory or file).
        
        Args:
            model_path: Model path, can be:
                - Directory: contains config.json and weights (model.pth, etc.)
                - File: .pth weight file (will look for config.json in same directory)
            device: Device ("cpu", "cuda", "cuda:0", etc.)
            dtype: Data type ("fp32", "fp16", "bf16" or torch.dtype)
            **kwargs: Additional arguments
                - strict: Whether to strictly match weights (default False)
                - load_weights: Whether to load weights (default True)
                
        Returns:
            Model instance with loaded weights
            
        Examples:
            # Load from directory
            model = AutoModel.from_pretrained("path/to/model_dir")
            
            # Load from weight file
            model = AutoModel.from_pretrained("path/to/model.pth", dtype="bf16")
        """
        if os.path.isdir(model_path):
            config_dir = model_path
            weight_file = None
            
            for filename in ["model.pth", "pytorch_model.pth", "model.safetensors"]:
                candidate = os.path.join(model_path, filename)
                if os.path.exists(candidate):
                    weight_file = candidate
                    break
        
        elif os.path.isfile(model_path):
            weight_file = model_path
            config_dir = os.path.dirname(model_path)
            
            if not config_dir:
                config_dir = "."
        
        else:
            raise ValueError(
                f"Path does not exist: {model_path}\n"
                f"model_path should be a model directory or weight file (.pth)"
            )
        
        config = RWKVConfig.from_pretrained(config_dir)
        model = cls.from_config(config)
        
        if dtype is not None:
            if isinstance(dtype, str):
                dtype_map = {
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                }
                dtype = dtype_map.get(dtype.lower(), torch.float32)
            
            model = model.to(dtype=dtype)
        
        load_weights = kwargs.pop("load_weights", True)
        if load_weights:
            if weight_file is None:
                raise FileNotFoundError(
                    f"Weight file not found. Supported filenames:\n"
                    f"  - model.pth\n"
                    f"  - pytorch_model.pth\n"
                    f"  - model.safetensors\n"
                    f"In directory: {config_dir}"
                )
            
            strict = kwargs.pop("strict", False)
            
            if weight_file.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_file)
                except ImportError:
                    raise ImportError(
                        "safetensors is required to load .safetensors format:\n"
                        "  pip install safetensors"
                    )
            else:
                map_location = "cpu"
                state_dict = torch.load(weight_file, map_location=map_location)
            
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
            if missing_keys and strict:
                print(f"Warning: Missing keys in model: {missing_keys[:5]}...")
            if unexpected_keys and strict:
                print(f"Warning: Unexpected keys in weights: {unexpected_keys[:5]}...")
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def save_pretrained(
        cls,
        model,
        save_directory: str,
        save_format: str = "pth",
        **kwargs
    ):
        """
        Save model to directory.
        
        Args:
            model: Model to save
            save_directory: Save directory
            save_format: Save format ("pth" or "safetensors")
            **kwargs: Additional arguments
        """
        os.makedirs(save_directory, exist_ok=True)
        
        if hasattr(model, "config"):
            model.config.save_pretrained(save_directory)
        else:
            print("Warning: Model has no config attribute, cannot save config")
        
        if save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                weight_file = os.path.join(save_directory, "model.safetensors")
                save_file(model.state_dict(), weight_file)
            except ImportError:
                raise ImportError(
                    "safetensors is required to save .safetensors format:\n"
                    "  pip install safetensors"
                )
        else:
            weight_file = os.path.join(save_directory, "model.pth")
            torch.save(model.state_dict(), weight_file)
        
        print(f"Model saved to: {save_directory}")
