"""
Command-line tool: Create standard RWKV model directory

Usage:
    rwkvtune-create-hub --help
    
    # Use predefined config
    rwkvtune-create-hub \\
        --output-dir models/my-rwkv7 \\
        --model-file checkpoints/rwkv7.pth \\
        --config-name rwkv7-0.1b
    
    # Use custom config
    rwkvtune-create-hub \\
        --output-dir models/custom \\
        --model-file checkpoints/custom.pth \\
        --config-name rwkv7-0.1b \\
        --ctx-len 8192
"""

import argparse
import sys
from pathlib import Path


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Create standard RWKV model directory (for RWKVTune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available predefined configs:
  - rwkv7-0.1b  (12 layers, 768 dim, 0.1B params)
  - rwkv7-0.4b  (24 layers, 1024 dim, 0.4B params)
  - rwkv7-1.5b  (24 layers, 2048 dim, 1.5B params)
  - rwkv7-2.9b  (32 layers, 2560 dim, 2.9B params)
  - rwkv7-7.2b  (32 layers, 4096 dim, 7.2B params)

Examples:
  # Basic usage (with predefined config)
  %(prog)s --output-dir models/my-rwkv7 \\
           --model-file checkpoints/rwkv7.pth \\
           --config-name rwkv7-0.1b

  # Override some config params
  %(prog)s --output-dir models/custom \\
           --model-file checkpoints/custom.pth \\
           --config-name rwkv7-0.1b \\
           --ctx-len 8192 \\
           --model-name "Custom RWKV7"

  # Add chat template and description
  %(prog)s --output-dir models/chat \\
           --model-file checkpoints/chat.pth \\
           --config-name rwkv7-0.1b \\
           --chat-template templates/chat.jinja \\
           --description "Fine-tuned chat model"
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="Model weights file path"
    )
    
    # Config options
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Predefined config name (required, e.g. rwkv7-0.1b, rwkv7-0.4b, rwkv7-1.5b)"
    )
    
    # Model architecture params
    arch_group = parser.add_argument_group("Model architecture params (optional, override predefined config)")
    arch_group.add_argument("--n-layer", type=int, help="Number of transformer layers")
    arch_group.add_argument("--n-embd", type=int, help="Embedding dimension")
    arch_group.add_argument("--vocab-size", type=int, help="Vocabulary size")
    arch_group.add_argument("--head-size-a", type=int, help="Attention head size")
    arch_group.add_argument("--ctx-len", type=int, help="Context length")
    arch_group.add_argument("--dim-att-lora", type=int, help="Attention LORA dimension")
    arch_group.add_argument("--dim-gate-lora", type=int, help="Gate LORA dimension")
    arch_group.add_argument("--dim-mv-lora", type=int, help="MV LORA dimension")
    
    # File paths
    file_group = parser.add_argument_group("File paths")
    file_group.add_argument(
        "--vocab-file",
        type=str,
        help="Vocabulary file path (uses default vocab if not specified)"
    )
    file_group.add_argument(
        "--chat-template",
        type=str,
        help="Chat template file path (.jinja format)"
    )
    
    # Model info
    info_group = parser.add_argument_group("Model info")
    info_group.add_argument(
        "--model-name",
        type=str,
        help="Model name (for README)"
    )
    info_group.add_argument(
        "--description",
        type=str,
        help="Model description (for README)"
    )
    
    # Generation config
    gen_group = parser.add_argument_group("Generation config")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_group.add_argument("--top-p", type=float, default=0.85, help="Top-p sampling")
    gen_group.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    gen_group.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    gen_group.add_argument("--max-length", type=int, default=1024, help="Max generation length")
    
    # Tokenizer config
    tok_group = parser.add_argument_group("Tokenizer config")
    tok_group.add_argument("--eos-token", type=str, default="\n\n", help="EOS token")
    tok_group.add_argument("--model-max-length", type=int, default=4096, help="Tokenizer max length")
    
    # Other options
    other_group = parser.add_argument_group("Other options")
    other_group.add_argument(
        "--copy-weights",
        action="store_true",
        default=True,
        help="Copy weights file (default)"
    )
    other_group.add_argument(
        "--link-weights",
        action="store_true",
        help="Create symlink instead of copying weights (saves space)"
    )
    other_group.add_argument(
        "--save-format",
        type=str,
        choices=["pth", "safetensors"],
        default="pth",
        help="Weights save format"
    )
    other_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing directory"
    )
    other_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Check required files
    if not Path(args.model_file).exists():
        print(f"[ERROR] Model file does not exist: {args.model_file}", file=sys.stderr)
        sys.exit(1)
    
    # Validate config-name (although required, provide friendly hint)
    valid_configs = ["rwkv7-0.1b", "rwkv7-0.4b", "rwkv7-1.5b", "rwkv7-2.9b", "rwkv7-7.2b", "rwkv7-13.3b"]
    if args.config_name not in valid_configs:
        print(f"[WARN] Config name '{args.config_name}' not in predefined list", file=sys.stderr)
        print(f"    Available configs: {', '.join(valid_configs)}", file=sys.stderr)
        print(f"    Will attempt to continue, but config loading may fail", file=sys.stderr)
    
    if args.chat_template and not Path(args.chat_template).exists():
        print(f"[ERROR] Chat template file does not exist: {args.chat_template}", file=sys.stderr)
        sys.exit(1)
    
    # Get vocab file
    vocab_file = args.vocab_file
    if vocab_file is None:
        try:
            from rwkvtune.data.tokenizers import get_vocab_path
            vocab_file = get_vocab_path()
            if args.verbose:
                print(f"[INFO] Using default vocab: {vocab_file}")
        except Exception as e:
            print(f"[WARN] Could not get default vocab: {e}", file=sys.stderr)
    
    # Import utility function
    try:
        from rwkvtune.utils.model_hub import create_model_hub
    except ImportError as e:
        print(f"[ERROR] Cannot import create_model_hub: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare arguments
    kwargs = {
        "output_dir": args.output_dir,
        "model_file": args.model_file,
        "config_name": args.config_name,
        "vocab_file": vocab_file,
        "chat_template_file": args.chat_template,
        "model_name": args.model_name,
        "description": args.description,
        # Model architecture
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "vocab_size": args.vocab_size,
        "head_size_a": args.head_size_a,
        "ctx_len": args.ctx_len,
        "dim_att_lora": args.dim_att_lora,
        "dim_gate_lora": args.dim_gate_lora,
        "dim_mv_lora": args.dim_mv_lora,
        # Generation config
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_length": args.max_length,
        # Tokenizer config
        "eos_token": args.eos_token,
        "model_max_length": args.model_max_length,
        # Other options
        "copy_weights": not args.link_weights,
        "save_format": args.save_format,
        "overwrite": args.overwrite,
    }
    
    # Show config info
    if args.verbose:
        print("\n" + "=" * 80)
        print("Configuration:")
        print("=" * 80)
        for key, value in kwargs.items():
            if value is not None:
                print(f"  {key}: {value}")
        print("=" * 80 + "\n")
    
    # Create model directory
    try:
        create_model_hub(**kwargs)
        print(f"\n[OK] Success! Model directory created: {args.output_dir}")
        print(f"\nUsage:")
        print(f"```python")
        print(f"from rwkvtune import AutoModel, AutoTokenizer")
        print(f"")
        print(f'model = AutoModel.from_pretrained("{args.output_dir}")')
        print(f'tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")')
        print(f"```")
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
