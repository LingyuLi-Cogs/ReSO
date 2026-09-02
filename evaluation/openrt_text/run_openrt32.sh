#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${OPENRT_PYTHON:-python3}"
ATTACKER_PORT="${OPENRT_ATTACKER_PORT:-18081}"
JUDGE_PORT="${OPENRT_JUDGE_PORT:-18082}"
TARGET_GPUS="${OPENRT_TARGET_GPUS:-0}"
ATTACKER_GPUS="${OPENRT_ATTACKER_GPUS:-1}"
JUDGE_GPUS="${OPENRT_JUDGE_GPUS:-2}"
if [[ "${OPENRT_RUNNER_CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  RUNNER_CUDA_VISIBLE_DEVICES="$OPENRT_RUNNER_CUDA_VISIBLE_DEVICES"
else
  RUNNER_CUDA_VISIBLE_DEVICES="$TARGET_GPUS"
fi
SHARE_HELPER_SERVER="${OPENRT_SHARE_HELPER_SERVER:-auto}"
ALLOW_HELPER_SPLIT_MIGRATION="${OPENRT_ALLOW_HELPER_SPLIT_MIGRATION:-0}"
SERVER_TIMEOUT="${OPENRT_SERVER_TIMEOUT:-600}"
CUDA_RESTART_LIMIT="${OPENRT_CUDA_RESTART_LIMIT:-2}"

TARGET_MODEL_PATH=""
ATTACKER_MODEL_PATH=""
JUDGE_MODEL_PATH=""
ATTACKER_BASE_URL=""
JUDGE_BASE_URL=""
ATTACKER_MODEL_NAME="openrt-attacker"
JUDGE_MODEL_NAME="openrt-judge"
RUNNER_ARGS=()
HELPER_MIGRATION_ARGS=()
HELPER_IDENTITY_ARGS=()
SERVER_PIDS=()
LIST_ONLY=0

if [[ "$ALLOW_HELPER_SPLIT_MIGRATION" != "0" && "$ALLOW_HELPER_SPLIT_MIGRATION" != "1" ]]; then
  echo "error: OPENRT_ALLOW_HELPER_SPLIT_MIGRATION must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$CUDA_RESTART_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "error: OPENRT_CUDA_RESTART_LIMIT must be a non-negative integer" >&2
  exit 2
fi

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash openrt_text/run_openrt32.sh \\" \
    "    --model_path /local/target \\" \
    "    --attacker_model_path /local/attacker \\" \
    "    --judge_model_path /local/judge \\" \
    "    --embedding_model_path /models--Qwen--Qwen3-Embedding-4B [runner options]" \
    "" \
    "If attacker/judge paths are omitted, the target path is reused. Existing" \
    "loopback OpenAI-compatible endpoints can be supplied with" \
    "--attacker_base_url and --judge_base_url." \
    "" \
    "GPU assignment env vars:" \
    "  OPENRT_TARGET_GPUS=0 OPENRT_ATTACKER_GPUS=1 OPENRT_JUDGE_GPUS=2" \
    "  Multiple target replicas: OPENRT_TARGET_GPUS=0,1 plus" \
    "  --target_devices cuda:0 cuda:1 (white-box uses cuda:0)" \
    "Helper sharing:" \
    "  OPENRT_SHARE_HELPER_SERVER=auto|0|1 (auto shares only on the same GPU set)" \
    "Parallel runner defaults:" \
    "  --parallel_attacks 4 --target_batch_size 0 --target_batch_wait_ms 250" \
    "  --target_batch_stats_every 25 prints the effective live batch size"
}

while (($#)); do
  case "$1" in
    --model_path)
      TARGET_MODEL_PATH="${2:?missing value for --model_path}"
      RUNNER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --attacker_model_path)
      ATTACKER_MODEL_PATH="${2:?missing value for --attacker_model_path}"
      shift 2
      ;;
    --judge_model_path)
      JUDGE_MODEL_PATH="${2:?missing value for --judge_model_path}"
      shift 2
      ;;
    --attacker_base_url)
      ATTACKER_BASE_URL="${2:?missing value for --attacker_base_url}"
      RUNNER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --judge_base_url)
      JUDGE_BASE_URL="${2:?missing value for --judge_base_url}"
      RUNNER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --attacker_model_name)
      ATTACKER_MODEL_NAME="${2:?missing value for --attacker_model_name}"
      RUNNER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --judge_model_name)
      JUDGE_MODEL_NAME="${2:?missing value for --judge_model_name}"
      RUNNER_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --list_attacks)
      LIST_ONLY=1
      RUNNER_ARGS+=("$1")
      shift
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

if ((LIST_ONLY)); then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/openrt32_runner.py" "${RUNNER_ARGS[@]}"
fi

if [[ -z "$TARGET_MODEL_PATH" ]]; then
  echo "error: --model_path is required" >&2
  exit 2
fi

echo "Checking Python dependencies and attack imports"
CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" \
  "$PYTHON_BIN" "$SCRIPT_DIR/openrt32_runner.py" \
  --check_imports_only "${RUNNER_ARGS[@]}"

ATTACKER_MODEL_PATH="${ATTACKER_MODEL_PATH:-$TARGET_MODEL_PATH}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$ATTACKER_MODEL_PATH}"
LOG_DIR="${OPENRT_SERVER_LOG_DIR:-./outputs/openrt32_local/server_logs}"
VLLM_CACHE_BASE="${OPENRT_VLLM_CACHE_BASE:-$LOG_DIR/vllm_cache}"
mkdir -p "$LOG_DIR" "$VLLM_CACHE_BASE"

resolve_checkpoint() {
  "$PYTHON_BIN" -c \
    'import sys; sys.path.insert(0, sys.argv[2]); from openrt32_models import resolve_hf_checkpoint; print(resolve_hf_checkpoint(sys.argv[1]))' \
    "$1" "$SCRIPT_DIR"
}

gpu_set_relation() {
  "$PYTHON_BIN" -c '
import sys

def parse(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit("GPU list must not be empty")
    if len(items) != len(set(items)):
        raise SystemExit(f"GPU list contains duplicates: {value}")
    return set(items)

left, right = parse(sys.argv[1]), parse(sys.argv[2])
if left == right:
    print("same")
elif left & right:
    print("overlap")
else:
    print("disjoint")
' "$1" "$2"
}

if [[ -z "$ATTACKER_BASE_URL" ]]; then
  ATTACKER_MODEL_PATH="$(resolve_checkpoint "$ATTACKER_MODEL_PATH")"
  HELPER_IDENTITY_ARGS+=(--attacker_checkpoint_identity "$ATTACKER_MODEL_PATH")
fi
if [[ -z "$JUDGE_BASE_URL" ]]; then
  JUDGE_MODEL_PATH="$(resolve_checkpoint "$JUDGE_MODEL_PATH")"
  HELPER_IDENTITY_ARGS+=(--judge_checkpoint_identity "$JUDGE_MODEL_PATH")
fi

cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_server() {
  local model_path="$1"
  local port="$2"
  local served_name="$3"
  local gpu_list="$4"
  local log_path="$5"
  local extra_args_text="${6:-}"
  local extra_args=()
  local cache_key
  local cache_root
  if [[ -n "$extra_args_text" ]]; then
    read -r -a extra_args <<<"$extra_args_text"
  fi
  cache_key="$(printf '%s' "$served_name" | tr -c '[:alnum:]_.-' '_')"
  cache_root="$VLLM_CACHE_BASE/$cache_key"
  mkdir -p "$cache_root/torchinductor" "$cache_root/triton"

  echo "Starting local vLLM server '$served_name' on 127.0.0.1:$port (GPU $gpu_list, cache $cache_root)"
  CUDA_VISIBLE_DEVICES="$gpu_list" \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  VLLM_CACHE_ROOT="$cache_root" \
  TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor" \
  TRITON_CACHE_DIR="$cache_root/triton" \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --host 127.0.0.1 \
      --port "$port" \
      --trust-remote-code \
      --dtype auto \
      --disable-log-requests \
      "${extra_args[@]}" >"$log_path" 2>&1 &
  SERVER_PIDS+=("$!")
}

wait_for_server() {
  local base_url="$1"
  local log_path="$2"
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  local probe="${base_url%/}"
  if [[ "$probe" != */v1 ]]; then
    probe="$probe/v1"
  fi
  probe="$probe/models"
  while ((SECONDS < deadline)); do
    if curl -fsS "$probe" >/dev/null 2>&1; then
      echo "Ready: $base_url"
      return 0
    fi
    sleep 2
  done
  echo "error: local model server did not become ready: $base_url" >&2
  echo "server log: $log_path" >&2
  tail -n 80 "$log_path" >&2 || true
  return 1
}

if [[ -z "$ATTACKER_BASE_URL" || -z "$JUDGE_BASE_URL" ]]; then
  if ! "$PYTHON_BIN" -c 'import vllm' >/dev/null 2>&1; then
    echo "error: vLLM is required to auto-start local attacker/judge servers." >&2
    echo "Install dependencies first, or pass loopback --attacker_base_url and --judge_base_url." >&2
    exit 2
  fi
fi

SHARE_HELPERS=0
GPU_RELATION="unknown"
if [[ -z "$ATTACKER_BASE_URL" && -z "$JUDGE_BASE_URL" ]]; then
  GPU_RELATION="$(gpu_set_relation "$ATTACKER_GPUS" "$JUDGE_GPUS")"
fi

# The target process starts after helper services and sees its own remapped
# CUDA namespace.  Reject physical overlap up front; otherwise two independent
# model copies can silently contend for one GPU and destroy throughput/OOM.
if [[ -z "$ATTACKER_BASE_URL" ]]; then
  TARGET_ATTACKER_RELATION="$(gpu_set_relation "$TARGET_GPUS" "$ATTACKER_GPUS")"
  if [[ "$TARGET_ATTACKER_RELATION" != "disjoint" ]]; then
    echo "error: target and auto-started attacker GPU sets must be disjoint" >&2
    echo "target GPUs: $TARGET_GPUS; attacker GPUs: $ATTACKER_GPUS" >&2
    exit 2
  fi
fi
if [[ -z "$JUDGE_BASE_URL" ]]; then
  TARGET_JUDGE_RELATION="$(gpu_set_relation "$TARGET_GPUS" "$JUDGE_GPUS")"
  if [[ "$TARGET_JUDGE_RELATION" != "disjoint" ]]; then
    echo "error: target and auto-started judge GPU sets must be disjoint" >&2
    echo "target GPUs: $TARGET_GPUS; judge GPUs: $JUDGE_GPUS" >&2
    exit 2
  fi
fi
case "$SHARE_HELPER_SERVER" in
  auto|AUTO|Auto)
    if [[ "$ATTACKER_MODEL_PATH" == "$JUDGE_MODEL_PATH" && "$GPU_RELATION" == "same" ]]; then
      SHARE_HELPERS=1
    elif [[ "$GPU_RELATION" == "same" || "$GPU_RELATION" == "overlap" ]]; then
      echo "error: attacker/judge GPU sets overlap but cannot share this service configuration" >&2
      echo "attacker GPUs: $ATTACKER_GPUS; judge GPUs: $JUDGE_GPUS" >&2
      exit 2
    fi
    ;;
  1|true|TRUE|True|yes|YES|Yes)
    if [[ "$ATTACKER_MODEL_PATH" != "$JUDGE_MODEL_PATH" ]]; then
      echo "error: OPENRT_SHARE_HELPER_SERVER=1 requires identical attacker/judge checkpoints" >&2
      exit 2
    fi
    SHARE_HELPERS=1
    ;;
  0|false|FALSE|False|no|NO|No)
    if [[ "$GPU_RELATION" == "same" || "$GPU_RELATION" == "overlap" ]]; then
      echo "error: separate attacker/judge services require disjoint GPU sets" >&2
      echo "attacker GPUs: $ATTACKER_GPUS; judge GPUs: $JUDGE_GPUS" >&2
      exit 2
    fi
    ;;
  *)
    echo "error: OPENRT_SHARE_HELPER_SERVER must be auto, 0, or 1" >&2
    exit 2
    ;;
esac

if [[ -z "$ATTACKER_BASE_URL" && -z "$JUDGE_BASE_URL" && "$SHARE_HELPERS" -eq 0 && "$ATTACKER_MODEL_PATH" == "$JUDGE_MODEL_PATH" && "$ALLOW_HELPER_SPLIT_MIGRATION" == "1" ]]; then
  # The legacy launcher merged identical checkpoints even when two disjoint
  # GPU groups were requested.  The explicit environment confirmation permits
  # only that exact shared->split manifest migration, preserving the old run
  # name and GCG.  New manifests store the resolved checkpoint identity.
  HELPER_MIGRATION_ARGS+=(--allow_helper_split_migration)
fi
if [[ -z "$ATTACKER_BASE_URL" && -z "$JUDGE_BASE_URL" && "$ALLOW_HELPER_SPLIT_MIGRATION" == "1" ]]; then
  # The same explicit confirmation also permits recording resolved checkpoint
  # identities into a legacy manifest that predates this metadata.
  HELPER_MIGRATION_ARGS+=(--allow_helper_identity_backfill)
fi

if [[ -z "$ATTACKER_BASE_URL" && -z "$JUDGE_BASE_URL" && "$SHARE_HELPERS" -eq 1 ]]; then
  ATTACKER_MODEL_NAME="openrt-shared-helper"
  JUDGE_MODEL_NAME="$ATTACKER_MODEL_NAME"
  ATTACKER_BASE_URL="http://127.0.0.1:${ATTACKER_PORT}/v1"
  JUDGE_BASE_URL="$ATTACKER_BASE_URL"
  shared_log="$LOG_DIR/shared_helper.log"
  start_server "$ATTACKER_MODEL_PATH" "$ATTACKER_PORT" "$ATTACKER_MODEL_NAME" "$ATTACKER_GPUS" "$shared_log" "${OPENRT_ATTACKER_VLLM_EXTRA_ARGS:-${OPENRT_VLLM_EXTRA_ARGS:-}}"
  wait_for_server "$ATTACKER_BASE_URL" "$shared_log"
else
  if [[ -z "$ATTACKER_BASE_URL" ]]; then
    ATTACKER_BASE_URL="http://127.0.0.1:${ATTACKER_PORT}/v1"
    attacker_log="$LOG_DIR/attacker.log"
    start_server "$ATTACKER_MODEL_PATH" "$ATTACKER_PORT" "$ATTACKER_MODEL_NAME" "$ATTACKER_GPUS" "$attacker_log" "${OPENRT_ATTACKER_VLLM_EXTRA_ARGS:-${OPENRT_VLLM_EXTRA_ARGS:-}}"
    wait_for_server "$ATTACKER_BASE_URL" "$attacker_log"
  fi
  if [[ -z "$JUDGE_BASE_URL" ]]; then
    JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
    judge_log="$LOG_DIR/judge.log"
    start_server "$JUDGE_MODEL_PATH" "$JUDGE_PORT" "$JUDGE_MODEL_NAME" "$JUDGE_GPUS" "$judge_log" "${OPENRT_JUDGE_VLLM_EXTRA_ARGS:-${OPENRT_VLLM_EXTRA_ARGS:-}}"
    wait_for_server "$JUDGE_BASE_URL" "$judge_log"
  fi
fi

echo "Running the selected OpenRT local text suite"
cuda_restarts=0
while true; do
  set +e
  CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  OPENAI_API_KEY=local-only \
  OPENAI_BASE_URL="$ATTACKER_BASE_URL" \
    "$PYTHON_BIN" "$SCRIPT_DIR/openrt32_runner.py" \
      --attacker_base_url "$ATTACKER_BASE_URL" \
      --attacker_model_name "$ATTACKER_MODEL_NAME" \
      --judge_base_url "$JUDGE_BASE_URL" \
      --judge_model_name "$JUDGE_MODEL_NAME" \
      "${HELPER_MIGRATION_ARGS[@]}" \
      "${HELPER_IDENTITY_ARGS[@]}" \
      "${RUNNER_ARGS[@]}"
  runner_status=$?
  set -e

  if ((runner_status != 75)); then
    exit "$runner_status"
  fi
  if ((cuda_restarts >= CUDA_RESTART_LIMIT)); then
    echo "error: exhausted $CUDA_RESTART_LIMIT automatic CUDA-context restart(s)" >&2
    echo "rerun with CUDA_LAUNCH_BLOCKING=1 to localize a deterministic kernel fault" >&2
    exit "$runner_status"
  fi
  cuda_restarts=$((cuda_restarts + 1))
  echo "Restarting target runner with a fresh CUDA context ($cuda_restarts/$CUDA_RESTART_LIMIT)" >&2
done
