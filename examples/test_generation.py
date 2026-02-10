#!/usr/bin/env python3
"""
RWKVTune Generation Test Script

Test generation capabilities of a trained model using samples from training data.

Usage:
    python test_generation.py --model_path ./output_sft/rwkv7-epoch8-merged
    
    # Using LoRA model (not merged)
    python test_generation.py \
        --model_path models/rwkv7-g1d-0.1b \
        --lora_path ./output_sft/rwkv7-epoch8
"""

import os
import json
import argparse

import torch


def load_test_samples(data_file: str, num_samples: int = 3):
    """Load test samples from training data."""
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip():
                data = json.loads(line)
                samples.append(data)
    return samples


def extract_prompt(conversations: list) -> str:
    """
    Extract prompt from conversation (system + first user turn).
    
    Note: RWKV7 supports thinking mode. To skip thinking and respond directly,
    add empty <think>\n\n</think> tags after Assistant:
    """
    prompt_parts = []
    
    # RWKV7 thinking mode: empty think tags to skip thinking
    THINK_PREFIX = "<think>\n\n</think>"
    
    for turn in conversations:
        role = turn.get('from', '').lower()
        content = turn.get('value', '')
        
        if role == 'system':
            prompt_parts.append(f"System: {content}\n\n")
        elif role in ('human', 'user'):
            prompt_parts.append(f"User: {content}\n\nAssistant:{THINK_PREFIX}")
            break  # Only take first user turn
    
    return ''.join(prompt_parts)


def extract_reference(conversations: list) -> str:
    """Extract first assistant reply as reference."""
    for i, turn in enumerate(conversations):
        role = turn.get('from', '').lower()
        if role in ('gpt', 'assistant'):
            return turn.get('value', '')[:200] + "..."
    return ""


def main():
    parser = argparse.ArgumentParser(description="RWKVTune Generation Test")
    
    parser.add_argument("--model_path", type=str, 
                        default="./output_sft/rwkv7-epoch8-merged",
                        help="Model path (merged model directory)")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA weights path (optional, for unmerged models)")
    parser.add_argument("--data_file", type=str, 
                        default="data/sharegpt_sample_100.jsonl",
                        help="Test data file")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of test samples")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--eos_token_id", type=int, default=261,
                        help="EOS token ID (RWKV default 261 = \\n\\n)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RWKVTune Generation Test")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    if args.lora_path:
        print(f"LoRA path: {args.lora_path}")
    print(f"Test data: {args.data_file}")
    print(f"Test samples: {args.num_samples}")
    print(f"Generation params: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_p={args.top_p}")
    print("=" * 60)
    
    # 1. Load model and tokenizer
    print("\nLoading model...")
    from rwkvtune import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained(args.model_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load LoRA if specified
    if args.lora_path:
        print(f"Loading LoRA weights: {args.lora_path}")
        from rwkvtune.peft import load_peft_model
        model = load_peft_model(model, args.lora_path)
    
    model.eval()
    print(f"Model loaded on {args.device}")
    
    # 2. Load test samples
    print(f"\nLoading test samples...")
    samples = load_test_samples(args.data_file, args.num_samples)
    print(f"Loaded {len(samples)} test samples")
    
    # 3. Test generation
    print("\n" + "=" * 60)
    print("Starting Generation Test")
    print("=" * 60)
    
    for i, sample in enumerate(samples):
        conversations = sample.get('conversations', [])
        if not conversations:
            continue
        
        prompt = extract_prompt(conversations)
        reference = extract_reference(conversations)
        
        print(f"\n{'─' * 60}")
        print(f"[Sample {i + 1}]")
        print(f"{'─' * 60}")
        
        prompt_display = prompt[:500] + "..." if len(prompt) > 500 else prompt
        print(f"\nPrompt:\n{prompt_display}")
        
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=args.device)
        
        print(f"\nGenerating... (input_length={len(input_ids)})")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=args.eos_token_id,
            )
        
        output_ids = output_ids[0].tolist()
        generated_ids = output_ids[len(input_ids):]
        generated_text = tokenizer.decode(generated_ids)
        
        print(f"\nGenerated:\n{generated_text}")
        
        if reference:
            print(f"\nReference (from training data):\n{reference}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
