#!/usr/bin/env python3
"""
RWKVTune SFT Single GPU Training Example

Demonstrates single GPU SFT (Supervised Fine-Tuning) using RWKVTune.
Uses ShareGPT multi-turn conversation format.

Usage:
    python train_sft_single_gpu.py \
        --model_path /path/to/rwkv/model \
        --data_file data/sharegpt_sample_100.jsonl \
        --output_dir output_sft

Requirements:
    - PyTorch >= 2.0
    - CUDA GPU
    - rwkvtune installed
"""

import os
import json
import argparse
from typing import List, Dict, Any

import torch
from datasets import Dataset

from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.training import SFTConfig, SFTTrainer
from rwkvtune.peft import LoraConfig, get_peft_model


def load_sharegpt_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load JSONL data in ShareGPT format.
    
    ShareGPT format:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
            ...
        ]
    }
    """
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} conversations")
    return data


def format_conversation(conversations: List[Dict], tokenizer) -> Dict[str, List[int]]:
    """
    Convert ShareGPT conversation format to input_ids and labels for training.
    
    Following rwkvtune data processing convention:
    - Data preparation applies shift: input_ids = tokens[:-1], labels = tokens[1:]
    - lightning_module computes loss without shift
    - This is the standard RWKV official training approach
    
    Strategy:
    - Build complete token sequence with role markers
    - Apply shift operation
    - Mask labels based on role (non-assistant parts set to -100)
    - Add <think>\n\n</think> before assistant replies for RWKV7 thinking mode compatibility
    
    Conversation template:
        System: {system_message}
        
        User: {user_message}
        
        Assistant:<think>
        
        </think>{assistant_message}
    """
    IGNORE_INDEX = -100
    
    # RWKV7 thinking mode: empty think tags to skip thinking and respond directly
    THINK_PREFIX = "<think>\n\n</think>"
    
    # Step 1: Build complete token sequence with role markers
    all_tokens = []
    token_roles = []  # 'mask' or 'gpt', marks which positions compute loss
    
    for turn in conversations:
        role = turn.get('from', '').lower()
        content = turn.get('value', '')
        
        if role == 'system':
            # System prompt - no loss computation
            text = f"System: {content}\n\n"
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            token_roles.extend(['mask'] * len(tokens))
            
        elif role in ('human', 'user'):
            # User input (includes Assistant: and think prefix) - no loss computation
            text = f"User: {content}\n\nAssistant:{THINK_PREFIX}"
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            token_roles.extend(['mask'] * len(tokens))
            
        elif role in ('gpt', 'assistant'):
            # Assistant reply - compute loss
            text = f"{content}\n\n"
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            token_roles.extend(['gpt'] * len(tokens))
    
    # Step 2: Apply shift operation (consistent with rwkvtune convention)
    # input_ids: used to predict next token
    # labels: tokens to be predicted (shifted sequence)
    if len(all_tokens) < 2:
        return {'input_ids': [], 'labels': []}
    
    input_ids = all_tokens[:-1]  # Remove last token
    labels = all_tokens[1:]      # Remove first token
    
    # Step 3: Mask labels based on role markers
    # labels[i] corresponds to original token[i+1], so use token_roles[i+1]
    for i in range(len(labels)):
        token_idx = i + 1
        if token_idx < len(token_roles):
            if token_roles[token_idx] != 'gpt':
                labels[i] = IGNORE_INDEX
        else:
            labels[i] = IGNORE_INDEX
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


def prepare_dataset(
    data: List[Dict], 
    tokenizer, 
    max_length: int = 2048
) -> Dataset:
    """
    Prepare training dataset.
    
    Args:
        data: ShareGPT format conversation data
        tokenizer: Tokenizer
        max_length: Maximum sequence length (truncates if exceeded)
    
    Returns:
        HuggingFace Dataset with input_ids and labels fields
    """
    processed_samples = []
    skipped = 0
    
    for item in data:
        conversations = item.get('conversations', [])
        if not conversations:
            skipped += 1
            continue
        
        sample = format_conversation(conversations, tokenizer)
        
        # Truncate to max length
        if len(sample['input_ids']) > max_length:
            sample['input_ids'] = sample['input_ids'][:max_length]
            sample['labels'] = sample['labels'][:max_length]
        
        # Skip samples that are too short
        if len(sample['input_ids']) < 10:
            skipped += 1
            continue
        
        processed_samples.append(sample)
    
    print(f"Processed {len(processed_samples)} samples (skipped {skipped})")
    
    dataset = Dataset.from_list(processed_samples)
    
    lengths = [len(s['input_ids']) for s in processed_samples]
    print(f"  Sequence length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    return dataset


def print_first_sample(dataset: Dataset, tokenizer) -> None:
    """
    Print first sample's input_ids and labels for debugging and validation.
    """
    if len(dataset) == 0:
        print("Dataset is empty, cannot print sample")
        return
    
    sample = dataset[0]
    input_ids = sample['input_ids']
    labels = sample['labels']
    
    print("\n" + "="*60)
    print("First Sample Details (for data processing validation)")
    print("="*60)
    
    input_text = tokenizer.decode(input_ids)
    print(f"\n[input_ids] Length: {len(input_ids)}")
    print("-" * 40)
    print(input_text)
    print("-" * 40)
    
    label_tokens = []
    for label in labels:
        if label == -100:
            label_tokens.append("[MASK]")
        else:
            label_tokens.append(tokenizer.decode([label]))
    
    valid_labels = [l for l in labels if l != -100]
    label_text = tokenizer.decode(valid_labels) if valid_labels else "(all masked)"
    
    print(f"\n[labels] Length: {len(labels)}, valid tokens: {len(valid_labels)}")
    print("-" * 40)
    print(f"Loss computed on (assistant reply):")
    print(label_text)
    print("-" * 40)
    
    print(f"\n[Token-level alignment] (first 50 tokens)")
    print("-" * 70)
    print(f"{'Pos':<6} {'input_id':<10} {'label':<10} {'input text':<20} {'prediction target'}")
    print("-" * 70)
    print("Note: input[i] predicts label[i], i.e., the next token")
    print("-" * 70)
    for i in range(min(50, len(input_ids))):
        input_text = tokenizer.decode([input_ids[i]]).replace('\n', '\\n').replace('\t', '\\t')
        if labels[i] != -100:
            label_text = tokenizer.decode([labels[i]]).replace('\n', '\\n').replace('\t', '\\t')
            label_str = f"{labels[i]} ({repr(label_text)})"
        else:
            label_str = "-100 (masked)"
        print(f"{i:<6} {input_ids[i]:<10} {labels[i] if labels[i] != -100 else -100:<10} {repr(input_text):<20} {label_str if labels[i] != -100 else 'no loss'}")
    if len(input_ids) > 50:
        print(f"... omitting {len(input_ids) - 50} remaining tokens")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RWKVTune SFT Single GPU Training")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="RWKV model path (directory or .pth file)")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="data/sharegpt_sample_100.jsonl",
                        help="Training data file path (ShareGPT JSONL format)")
    parser.add_argument("--ctx_len", type=int, default=2048,
                        help="Context length")
    
    # Training parameters
    parser.add_argument("--micro_bsz", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epoch_count", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr_init", type=float, default=2e-5,
                        help="Initial learning rate")
    parser.add_argument("--lr_final", type=float, default=1e-6,
                        help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Warmup steps")
    
    # LoRA parameters
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output_sft",
                        help="Output directory")
    
    # Checkpoint parameters
    parser.add_argument("--epoch_save", type=int, default=1,
                        help="Save interval (epochs), 0 = save only at end")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum checkpoints to keep (0 = unlimited)")
    
    # Hardware parameters
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Training precision")
    
    # Logging parameters
    parser.add_argument("--report_to", type=str, default="",
                        choices=["", "swanlab", "tensorboard"],
                        help="Logging tool (swanlab, tensorboard, or empty)")
    parser.add_argument("--run_name", type=str, default="",
                        help="Run name (for logging, auto-generated if empty)")
    parser.add_argument("--log_every_n_steps", type=int, default=1,
                        help="Log metrics every N optimizer steps")
    
    args = parser.parse_args()
    
    # ========== 1. Load Tokenizer ==========
    print("\n" + "="*60)
    print("Step 1: Load Tokenizer")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    
    # ========== 2. Load and Process Data ==========
    print("\n" + "="*60)
    print("Step 2: Load and Process Data")
    print("="*60)
    
    raw_data = load_sharegpt_data(args.data_file)
    dataset = prepare_dataset(raw_data, tokenizer, max_length=args.ctx_len)
    
    print_first_sample(dataset, tokenizer)
    
    # ========== 3. Load Model ==========
    print("\n" + "="*60)
    print("Step 3: Load Model")
    print("="*60)
    
    model = AutoModel.from_pretrained(args.model_path)
    print(f"Model loaded")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Dimensions: {model.config.n_embd}")
    print(f"  Vocab size: {model.config.vocab_size}")
    
    # ========== 4. Apply LoRA (Optional) ==========
    lora_config_dict = None
    if args.use_lora:
        print("\n" + "="*60)
        print("Step 4: Apply LoRA")
        print("="*60)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        lora_config_dict = {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
        }
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"LoRA applied")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Total params: {total_params:,}")
    
    # ========== 5. Create Training Config ==========
    print("\n" + "="*60)
    print("Step 5: Create Training Config")
    print("="*60)
    
    config = SFTConfig(
        ctx_len=args.ctx_len,
        micro_bsz=args.micro_bsz,
        accumulate_grad_batches=args.accumulate_grad_batches,
        epoch_count=args.epoch_count,
        epoch_save=args.epoch_save,
        save_total_limit=args.save_total_limit,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        devices=1,  # Single GPU
        precision=args.precision,
        proj_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_save_mode="lora_only" if args.use_lora else "full",
        report_to=args.report_to,
        run_name=args.run_name,
        log_every_n_steps=args.log_every_n_steps,
    )
    
    print(f"Training config:")
    print(f"  Context length: {config.ctx_len}")
    print(f"  Batch size: {config.micro_bsz}")
    print(f"  Gradient accumulation: {config.accumulate_grad_batches}")
    print(f"  Effective batch size: {config.micro_bsz * config.accumulate_grad_batches}")
    print(f"  Epochs: {config.epoch_count}")
    print(f"  Learning rate: {config.lr_init} -> {config.lr_final}")
    print(f"  Precision: {config.precision}")
    
    # ========== 6. Create Trainer and Start Training ==========
    print("\n" + "="*60)
    print("Step 6: Start Training")
    print("="*60)
    
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        lora_config=lora_config_dict,
    )
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
