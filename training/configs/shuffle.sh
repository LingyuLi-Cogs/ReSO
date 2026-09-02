#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model directory or identifier}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/reso_shuffle}"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-${REPO_ROOT}/training/reso_train.py}"
NUM_STEPS="${NUM_STEPS:-3000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BETA="${BETA:-0.1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAINER_SCRIPT}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --shuffle_labels \
  --shuffle_seed "${SHUFFLE_SEED}" \
  "$@"

