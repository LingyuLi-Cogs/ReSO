#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model directory or identifier}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/dpo}"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-${REPO_ROOT}/training/dpo_train.py}"
NUM_STEPS="${NUM_STEPS:-3000}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAINER_SCRIPT}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --replay_data "${REPO_ROOT}/dataset/replay.jsonl" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  "$@"

