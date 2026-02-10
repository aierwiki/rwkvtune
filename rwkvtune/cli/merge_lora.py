"""
Command-line interface for merging LoRA weights into base model

Usage:
    rwkvtune-merge-lora --base-model /path/to/base --lora-model /path/to/lora --output /path/to/output
"""

import argparse
import sys
import os
import json
import shutil
from pathlib import Path

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "0")  # Disable JIT to avoid merge issues


def main():
    """Entry point for rwkvtune-merge-lora command"""
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base RWKV model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    rwkvtune-merge-lora --base-model /path/to/base --lora-model /path/to/lora --output /path/to/merged

    # Specify device and precision
    rwkvtune-merge-lora --base-model ./base --lora-model ./lora --output ./merged --device cpu --precision fp32
        """
    )
    
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        required=True,
        help="Path to base model directory (must contain config.json and model weights)"
    )
    parser.add_argument(
        "--lora-model", "-l",
        type=str,
        required=True,
        help="Path to LoRA model directory (must contain adapter_config.json and adapter_model.bin)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for merging (default: cpu)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for model weights (default: bf16)"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="pth",
        choices=["pth", "safetensors"],
        help="Output format for merged weights (default: pth)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    base_model_path = Path(args.base_model)
    lora_model_path = Path(args.lora_model)
    output_path = Path(args.output)
    
    if not base_model_path.exists():
        print(f"[ERROR] Base model path does not exist: {base_model_path}")
        return 1
    
    if not lora_model_path.exists():
        print(f"[ERROR] LoRA model path does not exist: {lora_model_path}")
        return 1
    
    print("=" * 60)
    print("RWKVTune LoRA Merge Tool")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    print(f"Output dir: {output_path}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print("=" * 60)
    
    try:
        import torch
        from rwkvtune import AutoModel
        from rwkvtune.peft import load_peft_model
        
        # Set precision
        if args.precision == "bf16":
            dtype = torch.bfloat16
        elif args.precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # 1. Load base model
        print("\n[INFO] Loading base model...")
        model = AutoModel.from_pretrained(
            str(base_model_path),
            device=args.device,
            dtype=dtype,
        )
        print(f"[OK] Base model loaded")
        
        # 2. Load LoRA weights
        print("\n[INFO] Loading LoRA weights...")
        model = load_peft_model(model, str(lora_model_path))
        print(f"[OK] LoRA weights loaded")
        
        # 3. Merge LoRA weights into base model
        print("\n[INFO] Merging LoRA weights...")
        model.merge_adapter()
        print(f"[OK] LoRA weights merged")
        
        # 4. Get merged state_dict
        print("\n[INFO] Extracting merged weights...")
        merged_state_dict = model.get_merged_state_dict()
        print(f"[OK] Merged weight count: {len(merged_state_dict)}")
        
        # 5. Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 6. Save merged weights
        print("\n[INFO] Saving merged model...")
        if args.save_format == "safetensors":
            try:
                from safetensors.torch import save_file
                weights_file = output_path / "model.safetensors"
                save_file(merged_state_dict, str(weights_file))
            except ImportError:
                print("[WARN] safetensors not installed, falling back to pth format")
                weights_file = output_path / "model.pth"
                torch.save(merged_state_dict, weights_file)
        else:
            weights_file = output_path / "model.pth"
            torch.save(merged_state_dict, weights_file)
        
        print(f"[OK] Weights saved: {weights_file}")
        
        # 7. Process config files (merge base model and LoRA model configs)
        print("\n[INFO] Processing config files...")
        
        # Load base model's config.json (model architecture params)
        base_config = {}
        base_config_file = base_model_path / "config.json"
        if base_config_file.exists():
            with open(base_config_file, 'r', encoding='utf-8') as f:
                base_config = json.load(f)
        
        # Load LoRA model's tokenizer_config.json (may contain new eos_token etc.)
        lora_tokenizer_config = {}
        lora_tokenizer_config_file = lora_model_path / "tokenizer_config.json"
        if lora_tokenizer_config_file.exists():
            with open(lora_tokenizer_config_file, 'r', encoding='utf-8') as f:
                lora_tokenizer_config = json.load(f)
        
        # If LoRA model's tokenizer_config.json has eos_token, update eos_token_id in config.json
        if lora_tokenizer_config.get('eos_token'):
            eos_token = lora_tokenizer_config['eos_token']
            # Find token_id for eos_token
            additional_tokens = lora_tokenizer_config.get('additional_special_tokens', [])
            for token_info in additional_tokens:
                if isinstance(token_info, dict) and token_info.get('token') == eos_token:
                    new_eos_id = token_info.get('id')
                    if new_eos_id is not None:
                        old_eos_id = base_config.get('eos_token_id')
                        base_config['eos_token_id'] = new_eos_id
                        print(f"  Updated eos_token_id: {old_eos_id} -> {new_eos_id} (for '{eos_token}')")
                    break
        
        # Save merged config.json
        config_dst = output_path / "config.json"
        with open(config_dst, 'w', encoding='utf-8') as f:
            json.dump(base_config, f, indent=2, ensure_ascii=False)
        print(f"[OK] config.json saved (based on base model, token_id updated)")
        
        # Copy tokenizer files (prefer LoRA model's since SFT may have updated chat_template)
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
        ]
        for fname in tokenizer_files:
            # Prefer LoRA model's tokenizer files
            src = lora_model_path / fname
            source_type = "LoRA"
            if not src.exists():
                src = base_model_path / fname
                source_type = "base"
            
            if src.exists():
                dst = output_path / fname
                shutil.copy(src, dst)
                print(f"[OK] {fname} copied (from {source_type} model)")
        
        # 8. Create merge info file
        merge_info = {
            "base_model": str(base_model_path.absolute()),
            "lora_model": str(lora_model_path.absolute()),
            "precision": args.precision,
            "merged_by": "rwkvtune-merge-lora",
        }
        with open(output_path / "merge_info.json", "w", encoding="utf-8") as f:
            json.dump(merge_info, f, indent=2, ensure_ascii=False)
        print(f"[OK] merge_info.json created")
        
        print("\n" + "=" * 60)
        print("[OK] Merge complete!")
        print(f"Merged model saved at: {output_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
