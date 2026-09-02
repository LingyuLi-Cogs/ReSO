# OpenRT text-attack adapter

This directory contains the OpenRT components required for the paper's
text-only red-team evaluation. The vendored package is pinned to
`AI45Lab/OpenRT@365652f52c05c63324687ae69d4350499db9264c`; the HarmBench test
data is pinned to `centerforaisafety/HarmBench@8e1604d1171fe8a48d8febecd22f600e462bdcdd`.
See `THIRD_PARTY_NOTICES.md` for licenses and provenance.

## Project attack set

The reported score is macro ASR across 27 attacks and 240 standard/contextual
HarmBench text-test behaviors:

- GCG: 100 steps, search width 128.
- 25 generic black-box attacks selected by `--attacks paper`.
- Mousetrap: a fixed bank of 3 transformation depths by 3 trials, reported as
  Any@9 ASR.

AutoDAN-Turbo is excluded for runtime, and Rainbow Teaming is excluded because
its upstream implementation is not conditioned on the assigned behavior.
Mousetrap is deliberately outside the generic runner so every model receives
the same hashed prompt bank. The enumerated suite gives 27 x 240 = 6,480
behavior-level method outcomes before expanding Mousetrap's repeated trials.
The supplementary text's statement of 6,720 combinations is inconsistent with
its named 27-method list; this harness follows the enumerated methods and the
27 attack rows in the results table.

## Run

The unified project launcher generates the shared prompt bank, runs the core
attacks, runs Mousetrap, and writes the final 27-attack macro score:

```bash
OPENRT_TARGET_GPUS=0 \
OPENRT_ATTACKER_GPUS=1 \
OPENRT_JUDGE_GPUS=2 \
MOUSETRAP_TARGET_GPU=0 \
MOUSETRAP_JUDGE_GPU=1 \
python evaluation/run.py openrt \
  --model checkpoints/target-model \
  --attacker-model models/Qwen2.5-7B-Instruct \
  --judge-model models/CompassJudger-2-32B-Instruct \
  --embedding-model models/Qwen3-Embedding-4B
```

The core launcher starts loopback-only vLLM services for the attacker and
judge. The runner process loads the target checkpoint directly because GCG
requires weight and gradient access. `run_mousetrap.sh` subsequently starts one
target and one judge service on two distinct GPUs. GPU IDs and ports are
environment variables rather than machine-specific source configuration.

Use `--dry-run` to inspect the commands. For a small dependency/import check,
run `python evaluation/openrt_text/openrt32_runner.py --attacks paper
--check_imports_only` after installing `requirements-openrt32-addons.txt`.

## Outputs

The GCG, black-box, and Mousetrap runs each write a manifest, resumable
`results.jsonl`, and `summary.json`. GCG is separate because GPT-OSS requires
Harmony rendering and a differentiable BF16 load; the launcher detects that
model type and enables the required adapter. `summarize_paper.py` verifies that
GCG, all 25 black-box attacks, and Mousetrap Any@9 are complete before writing
`*_paper_summary.json`.
