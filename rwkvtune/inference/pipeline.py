"""
Inference pipeline for RWKV models
"""

import os
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union

from rwkvtune.models.rwkv7 import RWKV7Model, RWKV7Config
from rwkvtune.data.tokenizers import get_tokenizer


class InferencePipeline:
    """RWKV7 inference pipeline"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_config: Optional[str] = None,
        device: str = "cuda",
        precision: str = "bf16",
        vocab_path: Optional[str] = None,
        debug_output: bool = False,
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Model checkpoint path (optional if model_config provides it)
            model_config: Model config name (e.g., "0.1b", "rwkv7-1.5b" or config file path)
            device: Device (cuda/cpu)
            precision: Precision (fp32/fp16/bf16)
            vocab_path: Vocabulary path (optional, uses built-in vocab by default)
            debug_output: Whether to print debug info (token encoding/decoding validation)
        """
        self.device = torch.device(device)
        self.precision = precision
        self.debug_output = debug_output
        self._first_generate = True
        
        n_layer = None
        n_embd = None
        vocab_size = None
        head_size_a = None
        dim_att_lora = None
        dim_gate_lora = None
        dim_mv_lora = None

        # Load from config system if provided
        if model_config:
            try:
                from rwkvtune.configs.model_loader import load_model_config
                config_obj = load_model_config(model_config)
                
                # Priority: command line model_path > config file model_path
                if not model_path:
                    if config_obj.model_path:
                        model_path = config_obj.model_path
                    else:
                        raise ValueError(
                            f"No model_file specified in config and no --model_path provided\n"
                            f"Please use --model_path to specify the model weights path"
                        )
                
                print(f"[OK] Using model config: {config_obj.model_name} ({config_obj.architecture_version})")
                print(f"[OK] Config source: {config_obj.config_source}")
                if model_path != config_obj.model_path:
                    print(f"[OK] Model weights: {model_path} (override)")
                else:
                    print(f"[OK] Model weights: {model_path}")
                
                # Get model parameters from config
                n_layer = config_obj.n_layer
                n_embd = config_obj.n_embd
                vocab_size = config_obj.vocab_size
                head_size_a = config_obj.head_size_a
                dim_att_lora = config_obj.dim_att_lora
                dim_gate_lora = config_obj.dim_gate_lora
                dim_mv_lora = config_obj.dim_mv_lora
                
            except Exception as e:
                print(f"[ERROR] Failed to load model config '{model_config}'")
                print(f"   {str(e)}")
                print(f"\nSupported config formats:")
                print(f"   1. Short name: 0.1b, 0.4b, 1.5b, 2.9b, 7.2b")
                print(f"   2. Full name: rwkv7-0.1b, rwkv7-1.5b, ...")
                print(f"   3. File path: /path/to/config.json")
                raise
        else:
            # Legacy mode: infer config from checkpoint
            if not model_path:
                raise ValueError("Must provide model_path or model_config")
            
            print(f"[WARN] Not using config system, inferring model params from checkpoint")
            print(f"Model path: {model_path}")
            head_size_a = 64
            dim_att_lora = 0
            dim_gate_lora = 0
            dim_mv_lora = 0
        
        # Load tokenizer
        if vocab_path is None:
            self.tokenizer = get_tokenizer()
        else:
            from rwkvtune.data.tokenizers import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(vocab_path)
        
        # Load model checkpoint
        print(f"Loading model checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Infer parameters from checkpoint if not using config
        if not model_config:
            vocab_size, n_embd = checkpoint["emb.weight"].shape
            n_layer = max(int(k.split(".")[1]) for k in checkpoint if k.startswith("blocks.")) + 1
        
        # Try to parse LORA dimensions from checkpoint
        def _shape_or_default(key: str, default: int = 0) -> int:
            tensor = checkpoint.get(key)
            if tensor is None:
                return default
            return tensor.shape[1] if tensor.ndim >= 2 else default

        if dim_att_lora is None or dim_att_lora <= 0:
            dim_att_lora = _shape_or_default("blocks.0.att.w1", 0)
        if dim_gate_lora is None or dim_gate_lora <= 0:
            dim_gate_lora = _shape_or_default("blocks.0.att.g1", 0)
        if dim_mv_lora is None or dim_mv_lora <= 0:
            dim_mv_lora = _shape_or_default("blocks.1.att.v1", _shape_or_default("blocks.0.att.v1", 0))

        # Infer head_size_a from checkpoint if missing
        if head_size_a is None or head_size_a <= 0:
            r_k = checkpoint.get("blocks.0.att.r_k")
            if r_k is not None and r_k.ndim == 2:
                head_size_a = r_k.shape[1]
            else:
                head_size_a = 64

        if model_path is None:
            raise ValueError("Cannot determine model weights path, please check input parameters")

        if n_layer is None or n_embd is None or vocab_size is None:
            raise ValueError("Cannot determine model architecture parameters, please check config or checkpoint")

        # Create model config
        config = RWKV7Config(
            n_layer=n_layer,
            n_embd=n_embd,
            vocab_size=vocab_size,
            ctx_len=4096,
            accelerator="cuda" if device == "cuda" else "cpu",
            precision=precision,
            head_size_a=head_size_a,
            dim_att_lora=dim_att_lora,
            dim_gate_lora=dim_gate_lora,
            dim_mv_lora=dim_mv_lora,
            load_model=model_path,
        )
        
        # Create and load model
        self.model = RWKV7Model(config)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        
        # Set precision
        if device == "cuda":
            if precision == "fp16":
                self.model.half()
            elif precision == "bf16":
                self.model.to(torch.bfloat16)
            else:
                self.model.float()
        else:
            self.model.float()
        
        self.model.eval()
        print(f"[OK] Model loaded: {n_layer} layers, {n_embd} dim, {vocab_size} vocab")
        
        # Initialize batch generator
        from rwkvtune.inference.batch_generator import BatchGenerator
        self.batch_generator = BatchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=None,
            use_state_cache=True,
            batch_strategy="auto"
        )
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(tokens)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        num_generations: int = 1,
        stream: bool = False,
    ) -> Union[str, List[str]]:
        """
        Unified inference interface - supports single and batch requests
        
        Args:
            prompt: Single prompt string or list of prompts
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_generations: Number of generations per prompt (default 1)
            stream: Stream output (not supported, interface reserved)
        
        Returns:
            If input is single string:
                - num_generations=1: returns single string
                - num_generations>1: returns list of strings
            If input is list:
                - returns list of strings (length = len(prompt) * num_generations)
        """
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        from rwkvtune.inference.batch_generator import GenerationConfig
        
        gen_config = GenerationConfig(
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_generations=num_generations,
        )
        
        results = self.batch_generator.generate(
            prompts=prompts,
            generation_config=gen_config
        )
        
        completions = results['completions']
        
        if stream:
            for text in completions:
                print(text, end="", flush=True)
                print()
        
        if is_single and num_generations == 1:
            return completions[0]
        else:
            return completions
    
    def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[tuple]] = None,
        **kwargs
    ) -> str:
        """
        Chat interface
        
        Args:
            user_input: User input
            system_prompt: System prompt (optional)
            history: Conversation history [(user, assistant), ...]
            **kwargs: Other parameters passed to generate
        
        Returns:
            Assistant response
        """
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n\n")
        
        if history:
            for user_msg, assistant_msg in history:
                prompt_parts.append(f"User: {user_msg}\n\nAssistant: {assistant_msg}\n\n")
        
        prompt_parts.append(f"User: {user_input}\n\nAssistant:")
        
        prompt = "".join(prompt_parts)
        response = self.generate(prompt, **kwargs)
        
        return response.strip()
