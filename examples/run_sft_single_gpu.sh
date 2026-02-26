#!/bin/bash
#
# RWKVTune SFT Single GPU Training Script
#
# ==================== Model Preparation ====================
#
# Before training, convert RWKV model weights to RWKVTune standard format.
# Use rwkvtune-create-hub command for conversion:
#
# 1. If you have original .pth weight files, convert first:
#
#    rwkvtune-create-hub \
#        --output-dir models/rwkv7-0.1b \
#        --model-file /path/to/rwkv7-0.1b.pth \
#        --config-name rwkv7-0.1b \
#        --ctx-len 4096
#
#    Available preset configs (--config-name):
#      - rwkv7-0.1b  (12 layers, 768 dim, 0.1B params)
#      - rwkv7-0.4b  (24 layers, 1024 dim, 0.4B params)
#      - rwkv7-1.5b  (24 layers, 2048 dim, 1.5B params)
#      - rwkv7-2.9b  (32 layers, 2560 dim, 2.9B params)
#      - rwkv7-7.2b  (32 layers, 4096 dim, 7.2B params)
#
#    Other common options:
#      --link-weights    # Use symlinks instead of copying (saves disk space)
#      --ctx-len 8192    # Override default context length
#      --verbose         # Show detailed info
#
# 2. Converted model directory structure:
#    models/rwkv7-0.1b/
#    ├── config.json
#    ├── model.pth
#    ├── tokenizer_config.json
#    ├── vocab.txt
#    └── generation_config.json
#
# 3. Or download pre-converted models from ModelScope:
#    from modelscope import snapshot_download
#    model_dir = snapshot_download('aierwiki/rwkv7-g1d-0.1b')
#
# ==================== Usage ====================
#
#   chmod +x run_sft_single_gpu.sh
#   ./run_sft_single_gpu.sh
#
# Or with custom model path:
#   MODEL_PATH=/path/to/model ./run_sft_single_gpu.sh
#
# ==================== Merging LoRA ====================
#
# After training, use rwkvtune-merge-lora to merge LoRA weights:
#
#   rwkvtune-merge-lora \
#       --base-model models/rwkv7-g1d-0.1b \
#       --lora-model output_sft/rwkv7-epoch16 \
#       --output models/rwkv7-g1d-0.1b-sft-merged
#
# Options:
#   --base-model, -b   Base model path
#   --lora-model, -l   LoRA checkpoint path (contains adapter_model.bin)
#   --output, -o       Merged model output directory
#   --precision        Precision (fp32/fp16/bf16, default bf16)
#   --save-format      Save format (pth/safetensors, default pth)
#

set -e

# ==================== Configuration ====================

# Model path (required)
MODEL_PATH="${MODEL_PATH:-models/rwkv7-g1d-0.1b}"

# Data file path (download from ModelScope: https://modelscope.cn/datasets/aierwiki/sharegpt_roleplay_sample_100)
DATA_FILE="${DATA_FILE:-data/sharegpt_roleplay_sample_100.jsonl}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-output_sft}"

# Context length
CTX_LEN="${CTX_LEN:-8192}"

# Batch size (adjust based on GPU memory)
MICRO_BSZ="${MICRO_BSZ:-2}"

# Gradient accumulation steps
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-4}"

# Training epochs
EPOCHS="${EPOCHS:-3}"

# Learning rate
LR_INIT="${LR_INIT:-2e-5}"
LR_FINAL="${LR_FINAL:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"

# Training precision (fp32, fp16, bf16)
PRECISION="${PRECISION:-bf16}"

# LoRA config (set to 1 to enable LoRA)
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

# Checkpoint save config
EPOCH_SAVE="${EPOCH_SAVE:-1}"              # Save every N epochs (0 = save only at end)
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"  # Max checkpoints to keep (0 = unlimited)

# Logging config (swanlab, tensorboard, or empty)
REPORT_TO="${REPORT_TO:-swanlab}"
RUN_NAME="${RUN_NAME:-rwkvtune-sft}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-2}"

# GPU device
CUDA_DEVICE="${CUDA_DEVICE:-6}"

# ==================== Check Configuration ====================

echo "=================================================="
echo "       RWKVTune SFT Single GPU Training"
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
fi
echo "  Save interval:   every ${EPOCH_SAVE} epoch(s)"
echo "  Max checkpoints: ${SAVE_TOTAL_LIMIT}"
echo "  Logging:         ${REPORT_TO:-none}"
if [ -n "${RUN_NAME}" ]; then
    echo "  Run name:        ${RUN_NAME}"
fi
echo "  Log frequency:   every ${LOG_EVERY_N_STEPS} step(s)"
echo "  GPU device:      ${CUDA_DEVICE}"
echo ""

# Check model path
if [ "${MODEL_PATH}" = "/path/to/rwkv/model" ]; then
    echo "Error: Please set MODEL_PATH environment variable or modify the default value"
    echo "Example: MODEL_PATH=/path/to/model ./run_sft_single_gpu.sh"
    exit 1
fi

if [ ! -e "${MODEL_PATH}" ]; then
    echo "Error: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

# Check data file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ] && [ ! -f "${DATA_FILE}" ]; then
    echo "Error: Data file does not exist: ${DATA_FILE}"
    echo "Please ensure data/sharegpt_sample_100.jsonl exists"
    exit 1
fi

# ==================== Setup Environment ====================

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

cd "${SCRIPT_DIR}"

echo "Starting training..."
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
    --precision ${PRECISION}"

# Add LoRA parameters
if [ "${USE_LORA}" = "1" ]; then
    CMD="${CMD} --use_lora \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT}"
fi

# Add logging parameters
if [ -n "${REPORT_TO}" ]; then
    CMD="${CMD} --report_to ${REPORT_TO}"
fi
if [ -n "${RUN_NAME}" ]; then
    CMD="${CMD} --run_name ${RUN_NAME}"
fi
CMD="${CMD} --log_every_n_steps ${LOG_EVERY_N_STEPS}"

# Execute training
echo "Executing command:"
echo "${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=================================================="
echo "Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=================================================="
