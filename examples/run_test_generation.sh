#!/bin/bash
#
# RWKVTune Generation Test Script
#
# Usage:
#   chmod +x run_test_generation.sh
#   ./run_test_generation.sh
#

set -e

# ==================== Configuration ====================

# Model path (merged model)
MODEL_PATH="./output_sft/rwkv7-epoch3"
# MODEL_PATH="models/rwkv7-g1d-0.1b"

# LoRA config (USE_LORA=1 for unmerged LoRA model)
USE_LORA=1
BASE_MODEL="models/rwkv7-g1d-0.1b"
LORA_PATH="./output_sft/rwkv7-epoch3"

# Test data
DATA_FILE="data/sharegpt_sample_100.jsonl"
NUM_SAMPLES=3

# Generation parameters
MAX_NEW_TOKENS=256
TEMPERATURE=0.8
TOP_P=0.9
EOS_TOKEN_ID=261  # RWKV default EOS is \n\n, token ID 261

# GPU device
CUDA_DEVICE=6

# ==================== No modification needed below ====================

echo "=================================================="
echo "       RWKVTune Generation Test"
echo "=================================================="
echo ""
echo "Configuration:"
if [ "${USE_LORA}" = "1" ]; then
    echo "  Base model:      ${BASE_MODEL}"
    echo "  LoRA path:       ${LORA_PATH}"
else
    echo "  Model path:      ${MODEL_PATH}"
fi
echo "  Test data:       ${DATA_FILE}"
echo "  Test samples:    ${NUM_SAMPLES}"
echo "  Max generation:  ${MAX_NEW_TOKENS} tokens"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Top-p:           ${TOP_P}"
echo "  EOS Token:       ${EOS_TOKEN_ID}"
echo "  GPU device:      ${CUDA_DEVICE}"
echo ""

# Check model path
if [ "${USE_LORA}" = "1" ]; then
    if [ ! -e "${BASE_MODEL}" ]; then
        echo "Error: Base model path does not exist: ${BASE_MODEL}"
        exit 1
    fi
    if [ ! -e "${LORA_PATH}" ]; then
        echo "Error: LoRA path does not exist: ${LORA_PATH}"
        exit 1
    fi
else
    if [ ! -e "${MODEL_PATH}" ]; then
        echo "Error: Model path does not exist: ${MODEL_PATH}"
        echo "Tip: Run training and merge LoRA first, or set USE_LORA=1 for unmerged model"
        exit 1
    fi
fi

# Check test data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/${DATA_FILE}" ] && [ ! -f "${DATA_FILE}" ]; then
    echo "Error: Test data file does not exist: ${DATA_FILE}"
    exit 1
fi

# Setup environment
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
cd "${SCRIPT_DIR}"

echo "Starting test..."
echo ""

# Run test
if [ "${USE_LORA}" = "1" ]; then
    python test_generation.py \
        --model_path "${BASE_MODEL}" \
        --lora_path "${LORA_PATH}" \
        --data_file "${DATA_FILE}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --eos_token_id "${EOS_TOKEN_ID}"
else
    python test_generation.py \
        --model_path "${MODEL_PATH}" \
        --data_file "${DATA_FILE}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --eos_token_id "${EOS_TOKEN_ID}"
fi

echo ""
echo "=================================================="
echo "Test Complete!"
echo "=================================================="
