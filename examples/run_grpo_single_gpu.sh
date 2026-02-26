#!/bin/bash
#
# RWKVTune GRPO Single GPU Training Script
#

set -e

# ==================== Configuration ====================

# Model path (hub-style directory created by rwkvtune-create-hub https://modelscope.cn/models/aierwiki/rwkv7-g1a3-2.9b-20251103-ctx8192)
MODEL_PATH="./output_sft_grpo/rwkv7-epoch3"

# Data file (ShareGPT-style JSONL; sample data: download from ModelScope https://modelscope.cn/datasets/aierwiki/sharegpt_roleplay_sample_100, put in data/ and rename if needed)
DATA_FILE="data/sharegpt_sample_100.jsonl"

# Output directory for GRPO training
OUTPUT_DIR="output_grpo"

# Data: max samples to use (0 = use all)
MAX_SAMPLES=100

# Checkpoint saving: save model checkpoint every N batches (0 = disabled, only save at epoch end)
SAVE_EVERY_N_BATCHES=64

# Rollout saving: save prompt/completion/rewards every N steps (0 = disabled)
SAVE_ROLLOUT_STEPS=16
# Optional: directory for rollout .jsonl (default: ${OUTPUT_DIR}/rollouts)
# SAVE_ROLLOUT_PATH=""

# Reward weights: comma-separated (has_think, think_len, reply_len, actions_style, similarity, format_clean, eos_ending). Empty = default
# REWARD_WEIGHTS="1.0,0.5,0.6,0.8,0.5,0.8,1.2"

# Advantage clipping: clamp advantages to [-clip, +clip] after normalization.
# Prevents extreme gradients from near-zero std groups. Set to empty to disable.
ADVANTAGE_CLIP=5.0

# Low-quality group suppression: groups with max_reward < threshold get near-zero advantages
# Set to empty to disable (default). Recommended starting value: 0.5~1.0
LOW_REWARD_THRESHOLD=0.5
LOW_REWARD_SCALE=0.01

# GRPO core parameters
MICRO_BSZ=1              # Prompts per GPU batch
NUM_GENERATIONS=16        # Completions per prompt (G)
STEPS_PER_GENERATION=16   # Training steps per generation
EPOCH_COUNT=5            # Number of GRPO epochs

# Generation parameters
MAX_PROMPT_LENGTH=2048      # Max prompt length (tokens)
MAX_COMPLETION_LENGTH=1536  # Max completion length (tokens, think 500-1000 + reply ~30)
TEMPERATURE=1.0
TOP_P=0.95
REPETITION_PENALTY=1.0

# Optimization parameters
LR_INIT=1e-6
LR_FINAL=1e-7
WARMUP_STEPS=8
ACCUMULATE_GRAD_BATCHES=8  # Gradient accumulation steps
PRECISION="bf16"         # fp32 / fp16 / bf16

# Hardware
DEVICES=1
CUDA_DEVICE=6

# LoRA parameters (always enabled for GRPO)
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# KL penalty: constrain policy from drifting too far from the reference model.
# With LoRA, the reference is computed by disabling the adapter (no extra VRAM).
# Typical range: 0.01 ~ 0.1. Set to empty or 0 to disable.
BETA=0.04

# Logging
REPORT_TO="swanlab"      # "" / "swanlab" / "wandb"
RUN_NAME="rwkvtune-grpo"

# ==================== Check Configuration ====================

echo "=================================================="
echo "       RWKVTune GRPO Single GPU Training"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Model path:          ${MODEL_PATH}"
echo "  Data file:           ${DATA_FILE}"
echo "  Output dir:          ${OUTPUT_DIR}"
echo "  max_samples:         ${MAX_SAMPLES} (0=all)"
echo "  save_every_n_batches:${SAVE_EVERY_N_BATCHES}"
echo "  save_rollout_steps:  ${SAVE_ROLLOUT_STEPS}"
echo "  micro_bsz:           ${MICRO_BSZ}"
echo "  num_generations:     ${NUM_GENERATIONS}"
echo "  steps_per_generation:${STEPS_PER_GENERATION}"
echo "  epoch_count:         ${EPOCH_COUNT}"
echo "  max_prompt_length:   ${MAX_PROMPT_LENGTH}"
echo "  max_completion_len:  ${MAX_COMPLETION_LENGTH}"
echo "  temperature:         ${TEMPERATURE}"
echo "  top_p:               ${TOP_P}"
echo "  repetition_penalty: ${REPETITION_PENALTY}"
echo "  accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  precision:           ${PRECISION}"
echo "  devices:             ${DEVICES}"
echo "  use_lora:            true"
echo "  lora_r:              ${LORA_R}"
echo "  lora_alpha:          ${LORA_ALPHA}"
echo "  lora_dropout:        ${LORA_DROPOUT}"
echo "  beta (KL penalty):   ${BETA:-0}"
echo "  logging:             ${REPORT_TO:-none}"
echo "  run_name:            ${RUN_NAME}"
echo "  GPU device:          ${CUDA_DEVICE}"
echo ""

# Check model path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${SCRIPT_DIR}/${MODEL_PATH}" ] && [ ! -e "${MODEL_PATH}" ]; then
  echo "Error: Model path does not exist: ${MODEL_PATH}"
  echo "Please ensure the hub-style model directory exists, e.g.:"
  echo "  models/rwkv7-g1d-0.1b"
  exit 1
fi

# Check data file
if [ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ] && [ ! -f "${DATA_FILE}" ]; then
  echo "Error: Data file does not exist: ${DATA_FILE}"
  echo "Please ensure data/sharegpt_sample_100.jsonl exists"
  exit 1
fi

# ==================== Setup Environment ====================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

cd "${SCRIPT_DIR}"

echo "Starting GRPO training..."
echo ""

# ==================== Run Training ====================

CMD="python train_grpo_single_gpu.py \
  --model_path ${MODEL_PATH} \
  --data_file ${DATA_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --max_samples ${MAX_SAMPLES} \
  --micro_bsz ${MICRO_BSZ} \
  --num_generations ${NUM_GENERATIONS} \
  --steps_per_generation ${STEPS_PER_GENERATION} \
  --epoch_count ${EPOCH_COUNT} \
  --max_prompt_length ${MAX_PROMPT_LENGTH} \
  --max_completion_length ${MAX_COMPLETION_LENGTH} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --repetition_penalty ${REPETITION_PENALTY} \
  --lr_init ${LR_INIT} \
  --lr_final ${LR_FINAL} \
  --warmup_steps ${WARMUP_STEPS} \
  --accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
  --precision ${PRECISION} \
  --devices ${DEVICES} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --save_rollout_steps ${SAVE_ROLLOUT_STEPS} \
  --save_every_n_batches ${SAVE_EVERY_N_BATCHES}"

if [ -n "${SAVE_ROLLOUT_PATH}" ]; then
  CMD="${CMD} --save_rollout_path ${SAVE_ROLLOUT_PATH}"
fi
if [ -n "${REWARD_WEIGHTS}" ]; then
  CMD="${CMD} --reward_weights ${REWARD_WEIGHTS}"
fi
if [ -n "${ADVANTAGE_CLIP}" ]; then
  CMD="${CMD} --advantage_clip ${ADVANTAGE_CLIP}"
fi
if [ -n "${LOW_REWARD_THRESHOLD}" ]; then
  CMD="${CMD} --low_reward_threshold ${LOW_REWARD_THRESHOLD} --low_reward_scale ${LOW_REWARD_SCALE}"
fi
if [ -n "${BETA}" ] && [ "${BETA}" != "0" ]; then
  CMD="${CMD} --beta ${BETA}"
fi
if [ -n "${REPORT_TO}" ]; then
  CMD="${CMD} --report_to ${REPORT_TO}"
fi

if [ -n "${RUN_NAME}" ]; then
  CMD="${CMD} --run_name ${RUN_NAME}"
fi

echo "Executing command:"
echo "${CMD}"
echo ""

eval ${CMD}

