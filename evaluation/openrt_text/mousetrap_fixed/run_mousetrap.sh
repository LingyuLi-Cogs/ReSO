#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENRT_TEXT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${OPENRT_PYTHON:-python3}"
TARGET_GPU="${MOUSETRAP_TARGET_GPU:-0}"
JUDGE_GPU="${MOUSETRAP_JUDGE_GPU:-1}"
TARGET_PORT="${MOUSETRAP_TARGET_PORT:-18100}"
JUDGE_PORT="${MOUSETRAP_JUDGE_PORT:-18101}"
SERVER_TIMEOUT="${MOUSETRAP_SERVER_TIMEOUT:-900}"

TARGET_MODEL_PATH=""
JUDGE_MODEL_PATH=""
PROMPT_BANK=""
OUTPUT_DIR="./outputs/mousetrap_fixed"
RUN_NAME="mousetrap_fixed"
WORKERS="${MOUSETRAP_WORKERS:-32}"
SERVER_PIDS=()

usage() {
  printf '%s\n' \
    "Usage: bash evaluation/openrt_text/mousetrap_fixed/run_mousetrap.sh \\" \
    "  --model_path MODEL --judge_model_path JUDGE --prompt_bank BANK \\" \
    "  [--output_dir DIR] [--run_name NAME] [--workers N]" \
    "" \
    "GPU assignment: MOUSETRAP_TARGET_GPU=0 MOUSETRAP_JUDGE_GPU=1"
}

while (($#)); do
  case "$1" in
    --model_path) TARGET_MODEL_PATH="${2:?missing model path}"; shift 2 ;;
    --judge_model_path) JUDGE_MODEL_PATH="${2:?missing judge path}"; shift 2 ;;
    --prompt_bank) PROMPT_BANK="${2:?missing prompt bank}"; shift 2 ;;
    --output_dir) OUTPUT_DIR="${2:?missing output directory}"; shift 2 ;;
    --run_name) RUN_NAME="${2:?missing run name}"; shift 2 ;;
    --workers) WORKERS="${2:?missing worker count}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TARGET_MODEL_PATH" || -z "$JUDGE_MODEL_PATH" || -z "$PROMPT_BANK" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$PROMPT_BANK" ]]; then
  echo "error: prompt bank does not exist: $PROMPT_BANK" >&2
  exit 2
fi
if [[ "$TARGET_GPU" == "$JUDGE_GPU" ]]; then
  echo "error: target and judge must use different GPUs" >&2
  exit 2
fi

resolve_checkpoint() {
  "$PYTHON_BIN" -c \
    'import sys; sys.path.insert(0, sys.argv[2]); from openrt32_models import resolve_hf_checkpoint; print(resolve_hf_checkpoint(sys.argv[1]))' \
    "$1" "$OPENRT_TEXT_DIR"
}

model_type() {
  "$PYTHON_BIN" -c \
    'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text(encoding="utf-8")).get("model_type", "unknown"))' \
    "$1"
}

TARGET_MODEL_PATH="$(resolve_checkpoint "$TARGET_MODEL_PATH")"
JUDGE_MODEL_PATH="$(resolve_checkpoint "$JUDGE_MODEL_PATH")"
TARGET_MODEL_TYPE="$(model_type "$TARGET_MODEL_PATH")"
JUDGE_MODEL_TYPE="$(model_type "$JUDGE_MODEL_PATH")"

cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_DIR/server_logs/$RUN_NAME"

start_server() {
  local role="$1" gpu="$2" port="$3" model="$4" served_name="$5" model_kind="$6"
  local log="$OUTPUT_DIR/server_logs/$RUN_NAME/${role}.log"
  local extra=()
  if [[ -n "${MOUSETRAP_VLLM_EXTRA_ARGS:-}" ]]; then
    read -r -a extra <<<"$MOUSETRAP_VLLM_EXTRA_ARGS"
  fi
  if [[ "$model_kind" == "gpt_oss" ]]; then
    extra+=(--reasoning-parser openai_gptoss)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model" --served-model-name "$served_name" \
      --host 127.0.0.1 --port "$port" --trust-remote-code \
      --dtype auto --disable-log-requests "${extra[@]}" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
}

wait_for_server() {
  local role="$1" port="$2" log="$3" pid="$4"
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "error: $role server exited during startup" >&2
      tail -n 80 "$log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "error: timed out waiting for $role server" >&2
  return 1
}

start_server target "$TARGET_GPU" "$TARGET_PORT" "$TARGET_MODEL_PATH" mousetrap-target "$TARGET_MODEL_TYPE"
start_server judge "$JUDGE_GPU" "$JUDGE_PORT" "$JUDGE_MODEL_PATH" mousetrap-judge "$JUDGE_MODEL_TYPE"
wait_for_server target "$TARGET_PORT" "$OUTPUT_DIR/server_logs/$RUN_NAME/target.log" "${SERVER_PIDS[0]}"
wait_for_server judge "$JUDGE_PORT" "$OUTPUT_DIR/server_logs/$RUN_NAME/judge.log" "${SERVER_PIDS[1]}"

REASONING_ARGS=()
if [[ "$TARGET_MODEL_TYPE" == "gpt_oss" ]]; then
  REASONING_ARGS+=(--target_reasoning_effort low)
fi
if [[ "$JUDGE_MODEL_TYPE" == "gpt_oss" ]]; then
  REASONING_ARGS+=(--judge_reasoning_effort low)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_prompt_bank.py" \
  --prompt_bank "$PROMPT_BANK" \
  --target_base_urls "http://127.0.0.1:${TARGET_PORT}/v1" \
  --target_model_name mousetrap-target \
  --target_checkpoint_identity "$TARGET_MODEL_PATH" \
  --judge_base_urls "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --judge_model_name mousetrap-judge \
  --judge_checkpoint_identity "$JUDGE_MODEL_PATH" \
  "${REASONING_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" --run_name "$RUN_NAME" --workers "$WORKERS"
