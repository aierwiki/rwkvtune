#!/usr/bin/env python3
"""
RWKVTune GRPO Single GPU Training Example

This script demonstrates GRPO (Group Relative Policy Optimization) training
on a small ShareGPT-style dataset using a RWKV7 base model.

Dataset:
    - Input: examples/data/sharegpt_sample_100.jsonl
      (ShareGPT multi-turn conversation format)

Base model:
    - Default: models/rwkv7-g1d-0.1b (created via rwkvtune-create-hub)

GRPO training idea:
    - For each multi-turn chat session, create multiple samples:
        (system, user1, ai1),
        (system, user1, ai1, user2, ai2),
        ...
    - For each prefix, drop the last assistant reply and use the remaining
      turns as prompt. The original last assistant reply is treated as
      the "reference answer" for reward design.
    - The model generates a new reply; a reward function scores the
      generated text based on:
        * Correct <think>...</think> structure
        * Separation of actions (inside * ... *) and speech (outside *)
        * Length of thinking part (≈100–300 tokens)
        * Length of reply part (≈10–30 tokens)
        * Approximate similarity to the reference answer

Usage:
    python train_grpo_single_gpu.py \
        --model_path models/rwkv7-g1d-0.1b \
        --data_file data/sharegpt_sample_100.jsonl \
        --output_dir output_grpo

Requirements:
    - PyTorch >= 2.0
    - CUDA GPU
    - rwkvtune installed (editable or installed from wheel)
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset

from rwkvtune import AutoTokenizer
from rwkvtune.peft import LoraConfig
from rwkvtune.training import GRPOConfig, GRPOTrainer


def load_sharegpt_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load JSONL data in ShareGPT format.

    Each line:
        {
            "conversations": [
                {"from": "system", "value": "..."},
                {"from": "human",  "value": "..."},
                {"from": "gpt",    "value": "..."},
                ...
            ]
        }
    """
    data: List[Dict[str, Any]] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"[DATA] Loaded {len(data)} conversations from {data_file}")
    return data


def _extract_turn_pairs(conversations: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract (system_prompt, [(user, assistant), ...]) from ShareGPT conversation list.
    """
    system_prompt = ""
    pairs: List[Tuple[str, str]] = []

    # Find first system message (optional)
    for turn in conversations:
        role = turn.get("from", "").lower()
        if role == "system":
            system_prompt = turn.get("value", "")
            break

    # Collect user/assistant pairs in order
    last_user: str | None = None
    for turn in conversations:
        role = turn.get("from", "").lower()
        content = turn.get("value", "")
        if role in ("human", "user"):
            last_user = content
        elif role in ("gpt", "assistant") and last_user is not None:
            pairs.append((last_user, content))
            last_user = None

    return system_prompt, pairs


def build_grpo_dataset_from_sharegpt(
    data: List[Dict[str, Any]],
    tokenizer,
    max_prompt_length: int = 1024,
) -> Dataset:
    """
    Build a GRPO training dataset from ShareGPT-style conversations.

    For each session:
        system, (user1, ai1), (user2, ai2), ...

    We create samples:
        Sample k:
            Prompt:
                System: {system}\n\n
                User: {user1}\n\n
                Assistant: {ai1}\n\n
                ...
                User: {user_k}\n\n
                Assistant:
            ground_truth_answer: ai_k

    The GRPO trainer expects a HuggingFace Dataset with:
        - 'prompt' (string)
        - 'input_ids' (list[int]) pre-tokenized
        - additional fields (e.g. 'ground_truth_answer') passed through to reward functions
    """
    samples: List[Dict[str, Any]] = []
    skipped_sessions = 0

    for idx, item in enumerate(data):
        conversations = item.get("conversations", [])
        if not conversations:
            skipped_sessions += 1
            continue

        system_prompt, pairs = _extract_turn_pairs(conversations)
        if not pairs:
            skipped_sessions += 1
            continue

        # System block 与 infer_with_rwkv_pip.py 保持一致：Role/Task/Tone/Length/Format/Example + Roleplay Background
        format_instruction = (
            "**Role:** NPC in an interactive story  \n"
            "**Task:** Respond in character to the user's input  \n"
            "**Tone:** Immersive, concise, consistent with established context  \n"
            "**Length:** ~30 words  \n"
            "**Format:**  \n"
            "- Use *asterisks* for actions  \n"
            "- Reply directly, no greetings or explanations  \n"
            "- No extra formatting or meta-commentary  \n"
            "- Never use quotation marks around speech  \n"
            "- Never use parentheses  \n"
            "- Never start with a role prefix like \"Name:\"  \n"
            "**Example:**  \n"
            "*leans closer with a crooked smile* A better idea than movie night? Now I'm curious... you'd better show me.\n"
            "**Roleplay Background:**\n"
        )
        system_block = "System: " + format_instruction
        if system_prompt:
            system_block += "\n\n" + system_prompt.strip()
        system_block += "\n\n"
        history_parts: List[str] = [system_block]

        # For each pair k, build prompt that ends with "Assistant: <think"（与 infer_with_rwkv_pip 一致）
        for turn_idx, (user_text, ai_text) in enumerate(pairs):
            if turn_idx > 0:
                pass

            history_parts_copy = list(history_parts)
            history_parts_copy.append(f"User: {user_text}\n\n")
            history_parts_copy.append("Assistant: <think")

            prompt = "".join(history_parts_copy)

            # Tokenize prompt
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) > max_prompt_length:
                input_ids = input_ids[:max_prompt_length]

            if len(input_ids) < 16:
                # Skip extremely short prompts
                continue

            samples.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "ground_truth_answer": ai_text,
                    "session_index": idx,
                    "turn_index": turn_idx,
                }
            )

            # After creating sample for this (user, ai), now append full turn to history
            history_parts.append(f"User: {user_text}\n\n")
            history_parts.append(f"Assistant: {ai_text}\n\n")

    if not samples:
        raise RuntimeError("No valid GRPO samples generated from dataset.")

    print(f"[DATA] Built {len(samples)} GRPO samples (skipped {skipped_sessions} sessions)")

    ds = Dataset.from_list(samples)
    lengths = [len(s["input_ids"]) for s in samples]
    print(
        f"[DATA] Prompt length tokens: min={min(lengths)}, max={max(lengths)}, "
        f"avg={sum(lengths)/len(lengths):.1f}"
    )
    return ds


THINK_CLOSE = "</think>"

# =====================================================================
# Completion post-processing: truncate at next-turn delimiters
# =====================================================================
# During GRPO rollout the model sometimes does not emit EOS (\n\n) at
# the end of its reply and instead continues to generate the next
# turn's "System:" or "User:" header.  This postprocessor truncates
# each completion at the first occurrence of such a delimiter so that
# reward functions only see the actual reply and training loss is
# computed on clean tokens.
#
# This function serves as the reference implementation of a
# ``completion_postprocess_fn`` hook.  You can write your own as long
# as it follows the same input / output contract documented below.
# =====================================================================

# Substrings that signal "the model has leaked into the next turn"
_STOP_SUBSTRINGS = ("\nSystem:", "?System:", "\nUser:", "System:", "\n\nUser:", "\n\nSystem:", "?User:")

# RWKV EOS: "\n\n" is a single token (id=261 by default)
_EOS_TEXT = "\n\n"


def truncate_at_stop_postprocessor(
    prompts: List[str],
    completions: List[str],
    completion_ids: "torch.Tensor",
    masks: "torch.Tensor",
    tokenizer,
    **extra_fields,
) -> Dict[str, Any]:
    """Truncate completions at the first next-turn delimiter.

    This is a ``completion_postprocess_fn`` hook — called after rollout
    generation and before reward computation.

    Input contract (provided by the framework)
    -------------------------------------------
    prompts          : List[str]        – [B*G] prompt texts (read-only).
    completions      : List[str]        – [B*G] generated completion texts.
    completion_ids   : torch.Tensor     – [B*G, C] int token IDs produced by
                                          the model during rollout, 0-padded.
    masks            : torch.Tensor     – [B*G, C] bool mask where True marks
                                          a valid (generated) token position.
    tokenizer                           – tokenizer with ``encode()`` / ``decode()``.
    **extra_fields                      – any additional dataset columns
                                          (e.g. ``ground_truth_answer``).

    Output contract (must be returned)
    ----------------------------------
    A ``dict`` with exactly three keys::

        {
            'completions':    List[str],         # [B*G]  modified texts
            'completion_ids': torch.Tensor,      # [B*G, C] same shape & device
            'masks':          torch.Tensor,      # [B*G, C] same shape & device
        }

    Processing steps:
      1. Truncate each completion at the earliest next-turn delimiter
         (e.g. "System:", "User:").
      2. Ensure every completion ends with ``\\n\\n`` (RWKV EOS).
      3. Re-encode the final text to produce new ``completion_ids``,
         and rebuild ``masks`` accordingly.  The returned tensors keep
         the same ``[B*G, C]`` shape — sequences shorter than *C* are
         0-padded, sequences longer than *C* are truncated to *C*.
    """
    B_G, C = completion_ids.shape
    device = completion_ids.device

    # --- Step 1: text-level processing ---
    new_completions: List[str] = []
    for text in completions:
        if not text:
            new_completions.append(text)
            continue

        # Truncate at earliest stop substring
        cut = len(text)
        for stop in _STOP_SUBSTRINGS:
            idx = text.find(stop)
            if idx != -1 and idx < cut:
                cut = idx
        if cut < len(text):
            text = text[:cut].rstrip()

        # Ensure ends with \n\n
        if text and not text.endswith(_EOS_TEXT):
            text = text + _EOS_TEXT

        new_completions.append(text)

    # --- Step 2: re-encode all completions → new completion_ids & masks ---
    new_ids = torch.zeros(B_G, C, dtype=completion_ids.dtype, device=device)
    new_masks = torch.zeros(B_G, C, dtype=masks.dtype, device=device)

    for i, text in enumerate(new_completions):
        if not text:
            continue
        token_ids = tokenizer.encode(text)
        n = min(len(token_ids), C)
        new_ids[i, :n] = torch.tensor(token_ids[:n], dtype=completion_ids.dtype)
        new_masks[i, :n] = True

    return {
        "completions": new_completions,
        "completion_ids": new_ids,
        "masks": new_masks,
    }


def _range_score(
    x: int,
    soft_min: int,
    target_min: int,
    target_max: int,
    soft_max: int,
) -> float:
    """Smooth range reward: target range -> 1.0, soft range -> decay, else penalty."""
    if x <= 0:
        return -0.5
    if target_min <= x <= target_max:
        return 1.0
    if x < soft_min or x > soft_max:
        return -0.5
    if x < target_min:
        return (x - soft_min) / max(1.0, target_min - soft_min)
    return (soft_max - x) / max(1.0, soft_max - target_max)


def _parse_think_reply(completion: str) -> Tuple[str, str]:
    """Split completion into think part and reply part by </think>. Prompt ends with <think> so we only check for </think>."""
    text = (completion or "").strip()
    if THINK_CLOSE not in text:
        return "", text
    idx = text.find(THINK_CLOSE)
    think_text = text[:idx].strip()
    reply_text = text[idx + len(THINK_CLOSE) :].strip()
    return think_text, reply_text


def build_think_reply_reward_fns(tokenizer):
    """
    Build multiple reward functions (each returns List[float]) and default weights.

    Design principles:
      - All scores in [-1.0, 1.0] for consistent gradient scale.
      - has_think is the structural gate; other functions that depend on
        think/reply structure return -0.5 when </think> is missing, so
        they don't accidentally reward structurally broken completions.
      - format_clean only targets actual quotation marks (not apostrophes)
        to avoid false positives on English contractions like "I'm".
    """
    def _token_count(text: str) -> int:
        return len(tokenizer.encode(text))

    def _word_tokens(text: str) -> List[str]:
        return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]

    def _has_think_structure(c: str) -> bool:
        return THINK_CLOSE in (c or "").strip()

    # 1) Has thinking structure
    def reward_has_think(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        out = []
        for c in completions:
            if c is None:
                out.append(-1.0)
                continue
            out.append(1.0 if _has_think_structure(c) else -1.0)
        return out

    # 2) Think length in target range
    def reward_think_len(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        out = []
        for c in completions:
            if c is None or not _has_think_structure(c):
                out.append(-0.5)
                continue
            think_text, _ = _parse_think_reply(c)
            n = _token_count(think_text)
            out.append(_range_score(n, soft_min=60, target_min=100, target_max=300, soft_max=400))
        return out

    # 3) Reply length in target range (requires </think> structure)
    def reward_reply_len(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        out = []
        for c in completions:
            if c is None or not _has_think_structure(c):
                out.append(-0.5)
                continue
            _, reply_text = _parse_think_reply(c)
            n = _token_count(reply_text)
            out.append(_range_score(n, soft_min=5, target_min=10, target_max=30, soft_max=40))
        return out

    # 4) Actions in *...* and speech outside; asterisks must be paired
    def reward_actions_style(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        out = []
        for c in completions:
            if c is None:
                out.append(-0.5)
                continue
            _, reply_text = _parse_think_reply(c)
            asterisk_count = reply_text.count("*")
            if asterisk_count > 0 and asterisk_count % 2 != 0:
                out.append(-0.5)
            elif asterisk_count >= 2:
                segments = reply_text.split("*")
                has_action = any(i % 2 == 1 and seg.strip() for i, seg in enumerate(segments))
                has_speech = any(i % 2 == 0 and seg.strip() for i, seg in enumerate(segments))
                if has_action and has_speech:
                    out.append(1.0)
                elif has_action or has_speech:
                    out.append(0.3)
                else:
                    out.append(-0.3)
            else:
                out.append(-0.3)
        return out

    # 5) Similarity to reference answer
    def reward_similarity(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        gt_list = extra_fields.get("ground_truth_answer", [None] * len(completions))
        out = []
        for c, gt in zip(completions, gt_list):
            if c is None:
                out.append(-0.5)
                continue
            _, reply_text = _parse_think_reply(c)
            if not isinstance(gt, str) or not gt.strip():
                out.append(0.0)
                continue
            ref_tokens = _word_tokens(gt)
            gen_tokens = _word_tokens(reply_text)
            if not ref_tokens or not gen_tokens:
                out.append(-0.3)
                continue
            overlap = len(set(ref_tokens) & set(gen_tokens)) / max(1.0, len(ref_tokens))
            out.append(min(1.0, overlap * 2.0) - 0.2)  # shift so zero-overlap → -0.2
        return out

    # 6) Penalize quotation marks, role-prefix colons, and parentheses in reply.
    #    Only targets actual quotation marks — NOT apostrophes (\u2019) which
    #    appear in normal contractions like "I'm", "don't", "it's".
    _QUOTE_CHARS = set('"\u201c\u201d')
    _PAREN_CHARS = set('()\uff08\uff09')
    _ROLE_COLON_RE = re.compile(r"^\s*\w[\w\s]{0,20}:")

    def reward_format_clean(prompts: List[str], completions: List[str], **extra_fields) -> List[float]:
        out = []
        for c in completions:
            if c is None:
                out.append(0.0)
                continue
            _, reply_text = _parse_think_reply(c)
            if not reply_text:
                out.append(0.0)
                continue
            violations = 0
            if any(ch in _QUOTE_CHARS for ch in reply_text):
                violations += 1
            if any(ch in _PAREN_CHARS for ch in reply_text):
                violations += 1
            if _ROLE_COLON_RE.search(reply_text):
                violations += 1
            if violations == 0:
                out.append(1.0)
            elif violations == 1:
                out.append(-0.5)
            else:
                out.append(-1.0)
        return out

    reward_fns = [
        reward_has_think,
        reward_think_len,
        reward_reply_len,
        reward_actions_style,
        reward_similarity,
        reward_format_clean,
    ]
    default_weights = [1.0, 0.5, 0.3, 0.8, 0.6, 0.8]
    return reward_fns, default_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKVTune GRPO Single GPU Training Example")

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/rwkv7-g1d-0.1b",
        help="Base RWKV7 model directory (created by rwkvtune-create-hub)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/sharegpt_sample_100.jsonl",
        help="ShareGPT-style JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_grpo",
        help="Output directory for GRPO training",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max training samples to use (0 = use all). Useful for debugging or small runs.",
    )

    # GRPO core parameters
    parser.add_argument("--micro_bsz", type=int, default=2, help="Prompts per GPU batch")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt (G)")
    parser.add_argument(
        "--steps_per_generation",
        type=int,
        default=4,
        help="Training steps per generation (should divide micro_bsz * num_generations)",
    )
    parser.add_argument("--epoch_count", type=int, default=1, help="Number of GRPO epochs")

    # Generation parameters
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt length (tokens)")
    parser.add_argument("--max_completion_length", type=int, default=256, help="Max completion length (tokens)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for repeating tokens (>1 suppresses repetition, 1=off).",
    )

    # Optimization parameters
    parser.add_argument("--lr_init", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--lr_final", type=float, default=1e-7, help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Training precision")

    # Hardware
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")

    # LoRA (GRPO almost always uses LoRA; we hardcode it to be enabled)
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # Logging
    parser.add_argument(
        "--report_to",
        type=str,
        default="",
        choices=["", "swanlab", "wandb"],
        help="Logging backend (empty for none)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="rwkvtune-grpo",
        help="Run name for logging / experiment tracking",
    )

    # Reward weights (one per reward function: has_think, think_len, reply_len, actions_style, similarity)
    parser.add_argument(
        "--reward_weights",
        type=str,
        default="",
        help="Comma-separated weights for reward functions (default: 1.0,0.5,0.3,0.8,0.6,0.8). "
             "Order: has_think, think_len, reply_len, actions_style, similarity, format_clean.",
    )

    # Advantage clipping and low-quality group suppression
    parser.add_argument(
        "--advantage_clip",
        type=float,
        default=None,
        help="Clamp advantages to [-clip, +clip] after normalization (default: disabled, recommended: 5.0~10.0).",
    )
    parser.add_argument(
        "--low_reward_threshold",
        type=float,
        default=None,
        help="Groups with max_reward below this threshold get suppressed advantages (default: disabled).",
    )
    parser.add_argument(
        "--low_reward_scale",
        type=float,
        default=0.01,
        help="Advantage multiplier for low-quality groups (default: 0.01).",
    )

    # Checkpoint saving
    parser.add_argument(
        "--save_every_n_batches",
        type=int,
        default=0,
        help="Save model checkpoint every N batches; 0 = only save at epoch end.",
    )

    # Rollout data saving (prompt, completion, score, etc.)
    parser.add_argument(
        "--save_rollout_steps",
        type=int,
        default=0,
        help="Save GRPO rollout (prompt, completion, rewards) every N steps; 0 = disabled.",
    )
    parser.add_argument(
        "--save_rollout_path",
        type=str,
        default="",
        help="Directory for rollout .jsonl files (default: output_dir/rollouts when save_rollout_steps > 0).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("========== GRPO Single GPU Training ==========")
    print(f"Model path:         {args.model_path}")
    print(f"Data file:          {args.data_file}")
    print(f"Output dir:         {args.output_dir}")
    print(f"max_samples:        {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"micro_bsz:          {args.micro_bsz}")
    print(f"num_generations:    {args.num_generations}")
    print(f"steps_per_generation: {args.steps_per_generation}")
    print(f"epoch_count:        {args.epoch_count}")
    print(f"max_prompt_length:  {args.max_prompt_length}")
    print(f"max_completion_len: {args.max_completion_length}")
    print(f"temperature:        {args.temperature}")
    print(f"top_p:              {args.top_p}")
    print(f"repetition_penalty: {args.repetition_penalty}")
    print(f"accumulate_grad_batches: {args.accumulate_grad_batches}")
    print(f"precision:          {args.precision}")
    print(f"devices:            {args.devices}")
    print(f"use_lora:           True (forced for GRPO)")
    print(f"  lora_r:           {args.lora_r}")
    print(f"  lora_alpha:       {args.lora_alpha}")
    print(f"  lora_dropout:     {args.lora_dropout}")
    if args.save_every_n_batches > 0:
        print(f"save_every_n_batches: {args.save_every_n_batches}")
    if args.save_rollout_steps > 0:
        path = args.save_rollout_path or os.path.join(args.output_dir, "rollouts")
        print(f"save_rollout_steps: {args.save_rollout_steps}")
        print(f"save_rollout_path: {path}")
    print("==============================================\n")

    # 1. Load data
    raw_data = load_sharegpt_data(args.data_file)

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("[INIT] Tokenizer loaded")

    # 3. Build GRPO dataset
    dataset = build_grpo_dataset_from_sharegpt(
        raw_data,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
    )
    if args.max_samples > 0:
        n = min(len(dataset), args.max_samples)
        dataset = dataset.select(range(n))
        print(f"[DATA] Using {n} samples (max_samples={args.max_samples})")

    # 4. Build reward functions (multiple components + weights; each score recorded in JSON)
    reward_fns, default_weights = build_think_reply_reward_fns(tokenizer)
    if args.reward_weights:
        reward_weights = [float(x.strip()) for x in args.reward_weights.split(",") if x.strip()]
        if len(reward_weights) != len(reward_fns):
            raise ValueError(
                f"reward_weights has {len(reward_weights)} values but there are {len(reward_fns)} reward functions. "
                f"Expected: has_think, think_len, reply_len, actions_style, similarity"
            )
    else:
        reward_weights = default_weights
    print(f"[REWARD] Functions: {[f.__name__ for f in reward_fns]}, weights: {reward_weights}")

    # 5. Configure GRPO
    config = GRPOConfig(
        num_generations=args.num_generations,
        num_iterations=1,
        micro_bsz=args.micro_bsz,
        steps_per_generation=args.steps_per_generation,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        repetition_penalty=args.repetition_penalty,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        proj_dir=args.output_dir,
        epoch_count=args.epoch_count,
        save_every_n_batches=args.save_every_n_batches,
        save_total_limit=2,
        report_to=(args.report_to or None),
        run_name=args.run_name,
        use_lora=True,
        reward_weights=reward_weights,
        save_rollout_steps=args.save_rollout_steps,
        save_rollout_path=(args.save_rollout_path or None),
        advantage_clip=args.advantage_clip,
        low_reward_threshold=args.low_reward_threshold,
        low_reward_scale=args.low_reward_scale,
    )

    # 6. LoRA configuration (always enabled for GRPO)
    # Note: rwkvtune's LoraConfig does not use bias/task_type arguments
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # 7. Create GRPO trainer
    # The truncate_at_stop_postprocessor is an optional hook called after rollout
    # generation and before reward computation.  It truncates completions at
    # next-turn delimiters ("System:", "User:") to prevent the model from
    # "leaking" into the next turn.  Pass None to disable post-processing.
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_fns,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        completion_postprocess_fn=truncate_at_stop_postprocessor,
    )

    # 8. Start training
    trainer.train()


if __name__ == "__main__":
    main()

