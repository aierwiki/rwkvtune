#!/bin/bash
#
# RWKVTune SFT Single GPU Training Script (pre-GRPO)
#

set -e

# ==================== Configuration ====================
# All settings are fixed in script; no external overrides.

# Base model used for GRPO
MODEL_PATH="models/rwkv7-g1a3-2.9b-20251103-ctx8192"

# Data file (ShareGPT with reasoning; download from ModelScope: https://modelscope.cn/datasets/aierwiki/sharegpt_roleplay_with_reasoning)
DATA_FILE="data/sharegpt_roleplay_with_reasoning.jsonl"

# Output directory for SFT checkpoints
OUTPUT_DIR="output_sft_grpo"

# Context length
CTX_LEN=8192

# Batch size / grad accum
MICRO_BSZ=2
ACCUMULATE_GRAD=4

# Training epochs
EPOCHS=3

# Learning rate schedule
LR_INIT=2e-5
LR_FINAL=1e-6
WARMUP_STEPS=50

# Precision
PRECISION="bf16"

# Full-parameter fine-tuning (no LoRA)
USE_LORA=0
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0

# Checkpoint saving
EPOCH_SAVE=1
SAVE_TOTAL_LIMIT=3

# Logging
REPORT_TO="swanlab"
RUN_NAME="rwkvtune-sft-grpo"
LOG_EVERY_N_STEPS=2

# GPU
CUDA_DEVICE=6

# ==================== Check Configuration ====================

echo "=================================================="
echo "   RWKVTune SFT Single GPU Training (pre-GRPO)"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Model path:      ${MODEL_PATH}"
echo "  Data file:       ${DATA_FILE}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "  Context length:  ${CTX_LEN}"
echo "  Batch size:      ${MICRO_BSZ}"
echo "  Grad accum:      ${ACCUMULATE_GRAD}"
echo "  Effective batch: $((MICRO_BSZ * ACCUMULATE_GRAD))"
echo "  Epochs:          ${EPOCHS}"
echo "  Learning rate:   ${LR_INIT} -> ${LR_FINAL}"
echo "  Precision:       ${PRECISION}"
echo "  Use LoRA:        ${USE_LORA}"
if [ "${USE_LORA}" = "1" ]; then
  echo "  LoRA rank:       ${LORA_R}"
  echo "  LoRA alpha:      ${LORA_ALPHA}"
  echo "  LoRA dropout:    ${LORA_DROPOUT}"
fi
echo "  Save interval:   every ${EPOCH_SAVE} epoch(s)"
echo "  Max checkpoints: ${SAVE_TOTAL_LIMIT}"
echo "  Logging:         ${REPORT_TO}"
echo "  Run name:        ${RUN_NAME}"
echo "  Log frequency:   every ${LOG_EVERY_N_STEPS} step(s)"
echo "  GPU device:      ${CUDA_DEVICE}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -e "${SCRIPT_DIR}/${MODEL_PATH}" ] && [ ! -e "${MODEL_PATH}" ]; then
  echo "Error: Model path does not exist: ${MODEL_PATH}"
  exit 1
fi

if [ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ] && [ ! -f "${DATA_FILE}" ]; then
  echo "Error: Data file does not exist: ${DATA_FILE}"
  exit 1
fi

# ==================== Setup Environment ====================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
cd "${SCRIPT_DIR}"

echo "Starting SFT (pre-GRPO)..."
echo ""

# ==================== Run Training ====================

CMD="python train_sft_single_gpu.py \
  --model_path ${MODEL_PATH} \
  --data_file ${DATA_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --ctx_len ${CTX_LEN} \
  --micro_bsz ${MICRO_BSZ} \
  --accumulate_grad_batches ${ACCUMULATE_GRAD} \
  --epoch_count ${EPOCHS} \
  --epoch_save ${EPOCH_SAVE} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --lr_init ${LR_INIT} \
  --lr_final ${LR_FINAL} \
  --warmup_steps ${WARMUP_STEPS} \
  --precision ${PRECISION} \
  --last_reply_only"

if [ "${USE_LORA}" = "1" ]; then
  CMD="${CMD} --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT}"
fi

if [ -n "${REPORT_TO}" ]; then
  CMD="${CMD} --report_to ${REPORT_TO}"
fi
if [ -n "${RUN_NAME}" ]; then
  CMD="${CMD} --run_name ${RUN_NAME}"
fi
CMD="${CMD} --log_every_n_steps ${LOG_EVERY_N_STEPS}"

echo "Executing command:"
echo "${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=================================================="
echo "SFT Training (pre-GRPO) Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=================================================="

