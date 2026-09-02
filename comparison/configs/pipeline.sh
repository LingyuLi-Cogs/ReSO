#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to a local checkpoint or model identifier}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH%/}")}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/comparison/sampled-human-representation-vectors.jsonl}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/comparison/artifacts}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

MODEL_LOAD_ARGS=()
if [[ "${ALLOW_NETWORK:-0}" == "1" ]]; then
  MODEL_LOAD_ARGS+=(--no-local_files_only)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/extraction/extract_activations.py" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MODEL_NAME}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${ARTIFACTS_DIR}/activations/${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --dtype "${DTYPE}" \
  --all_layers \
  --merge_batches \
  "${MODEL_LOAD_ARGS[@]}"

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/comparison/linear_probe.py" \
  --activations-root "${ARTIFACTS_DIR}/activations" \
  --output-dir "${ARTIFACTS_DIR}/linear_probe" \
  --model "${MODEL_NAME}=${MODEL_NAME}"

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/comparison/category_centers.py" \
  --activations-root "${ARTIFACTS_DIR}/activations" \
  --output-dir "${ARTIFACTS_DIR}/category_centers" \
  --models "${MODEL_NAME}"

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/analysis/correlation_analysis.py" \
  --similarities-dir "${ARTIFACTS_DIR}/category_centers/similarities" \
  --output-dir "${ARTIFACTS_DIR}/analysis/correlations" \
  --models "${MODEL_NAME}"

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/analysis/plot_correlations.py" \
  --analysis-dir "${ARTIFACTS_DIR}/analysis/correlations" \
  --models "${MODEL_NAME}"

"${PYTHON_BIN}" "${REPO_ROOT}/comparison/analysis/plot_category_centers.py" \
  --centers-dir "${ARTIFACTS_DIR}/category_centers/centers" \
  --output-dir "${ARTIFACTS_DIR}/analysis/category_centers" \
  --models "${MODEL_NAME}"
