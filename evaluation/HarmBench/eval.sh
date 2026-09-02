#!/usr/bin/env bash
# One-click offline HarmBench ASR eval for a single trained checkpoint.
#
# Usage:
#   ./run_harmbench_eval.sh <MODEL_PATH> <CLS_PATH> [OUT_DIR] [MODEL_NAME]
#
#   MODEL_PATH  target model dir (base model or a .../best/model checkpoint)
#   CLS_PATH    local dir of cais/HarmBench-Llama-2-13b-cls (pre-downloaded)
#   OUT_DIR     output dir (default: ./harmbench_results)
#   MODEL_NAME  label for outputs (default: derived from MODEL_PATH)
#
# Runs two stages as SEPARATE processes so the target model's GPU memory is
# fully released before the 13B classifier loads. Fully offline. One model per
# run; run again with a different MODEL_PATH to append another row to the
# accumulating harmbench_asr.csv in OUT_DIR.
#
# Override any knob via env, e.g.:
#   TP=1 MAX_BEHAVIORS=40 CATEGORIES="standard" ./run_harmbench_eval.sh ...
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <MODEL_PATH> <CLS_PATH> [OUT_DIR] [MODEL_NAME]" >&2
  exit 1
fi

MODEL_PATH="$1"
CLS_PATH="$2"
OUT_DIR="${3:-./harmbench_results}"
MODEL_NAME="${4:-}"

cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TP="${TP:-0}"                          # 0 = all visible GPUs
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_BEHAVIORS="${MAX_BEHAVIORS:-0}"    # 0 = all selected behaviors
CATEGORIES="${CATEGORIES:-standard contextual}"
BEHAVIORS="${BEHAVIORS:-./data/behavior_datasets/harmbench_behaviors_text_all.csv}"

NAME_ARG=()
if [ -n "$MODEL_NAME" ]; then NAME_ARG=(--model_name "$MODEL_NAME"); fi

COMMON=(--out_dir "$OUT_DIR" --behaviors "$BEHAVIORS" \
        --categories $CATEGORIES --max_behaviors "$MAX_BEHAVIORS" --tp "$TP")

echo "=== HarmBench eval: $MODEL_PATH ==="
echo "--- stage 1/2: generating completions (target model) ---"
python harmbench_eval.py --stage generate \
  --model_path "$MODEL_PATH" "${NAME_ARG[@]}" \
  --max_new_tokens "$MAX_NEW_TOKENS" "${COMMON[@]}"

echo "--- stage 2/2: scoring with HarmBench classifier ---"
python harmbench_eval.py --stage evaluate \
  --model_path "$MODEL_PATH" --cls_path "$CLS_PATH" "${NAME_ARG[@]}" \
  "${COMMON[@]}"

echo "=== done. See $OUT_DIR/harmbench_asr.csv ==="
