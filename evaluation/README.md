# Evaluation harness

This directory contains the reproduction code for the nine out-of-distribution
benchmarks reported in the paper. Run every command from the repository root.
No script contains a private cluster path, API key, checkpoint identity, or
historical result. Target, judge, scorer, and attacker checkpoints are supplied
by the runner.

## Scope and metrics

| Harness | Paper metric | Evaluation model |
| --- | --- | --- |
| MMLU-Pro | accuracy | deterministic answer extraction |
| HaluEval | hallucination classification accuracy | deterministic Yes/No extraction |
| Flames | harmless score | official Flames scorer |
| MoReBench | main-benchmark score | CompassJudger-2-32B-Instruct |
| Ethics Benchmark | composite moral-reasoning score | benchmark heuristics |
| XSTest | balanced accuracy | CompassJudger-2-32B-Instruct |
| DeceptionBench | deception rate (lower is better) | CompassJudger-2-32B-Instruct |
| HarmBench | direct-request ASR (lower is better) | HarmBench-Llama-2-13b-cls |
| OpenRT | macro ASR over 27 attacks (lower is better) | CompassJudger-2-32B-Instruct |

The same judge checkpoint and generation settings must be reused across the
baseline, DPO, ReSO, and shuffled-control conditions.

## Installation

The non-OpenRT harnesses use the repository environment plus vLLM and datasets:

```bash
pip install -r requirements.txt
pip install -r evaluation/requirements.txt
```

OpenRT has additional pinned dependencies:

```bash
pip install -r evaluation/openrt_text/requirements-openrt32-addons.txt
```

All model-loading paths default to offline/local operation. Download model
checkpoints before moving to an offline evaluation machine.

## Unified entry point

Use local, non-secret paths appropriate for your machine:

```bash
MODEL=checkpoints/target-model
JUDGE=models/CompassJudger-2-32B-Instruct
HARM_CLASSIFIER=models/HarmBench-Llama-2-13b-cls
FLAMES_SCORER=models/flames-scorer
```

General capability:

```bash
python evaluation/run.py mmlu-pro --model "$MODEL"
python evaluation/run.py halueval --model "$MODEL"
```

The MMLU-Pro launcher evaluates all subjects with five-shot CoT, a 20,000-token
context, 1,024 generated tokens, and tensor parallelism 8 by default. HaluEval
runs QA, dialogue, and summarization with the chat template, greedy decoding, a
20,000-token context, and up to 4,096 generated tokens.

Values and moral reasoning:

```bash
python evaluation/run.py flames \
  --model "$MODEL" \
  --data-path data/Flames_1k_Chinese.jsonl \
  --scorer-model "$FLAMES_SCORER"

python evaluation/run.py morebench --model "$MODEL" --judge-model "$JUDGE"
python evaluation/run.py ethics --model "$MODEL"
python evaluation/run.py xstest --model "$MODEL" --judge-model "$JUDGE"
```

Flames data and the official scorer checkpoint are not redistributed here;
obtain them from the Flames release. The harness first generates responses and
then applies the dedicated scorer. MoReBench uses the 500-item main split.
XSTest evaluates all 250 safe and 200 unsafe prompts.

Safety and adversarial robustness:

```bash
python evaluation/run.py deception --model "$MODEL" --judge-model "$JUDGE"
python evaluation/run.py harmbench \
  --model "$MODEL" \
  --classifier-model "$HARM_CLASSIFIER"
```

HarmBench uses DirectRequest over the 300 standard/contextual behaviors in the
text-all split and excludes the copyright/hash-check subset. DeceptionBench
uses all five domains and the six single-turn L1/L2 conditions.

OpenRT uses one white-box method (GCG), 25 methods from the generic black-box
runner, and a separate fixed-prompt Mousetrap evaluation. AutoDAN-Turbo and
Rainbow Teaming are excluded. The enumerated set gives 27 x 240 = 6,480
behavior-level method outcomes before expanding Mousetrap's nine fixed trials.
The supplementary text states 6,720 combinations, which is inconsistent with
its named 27-method list; this harness follows the enumerated methods and table
rows. The project settings use
Qwen2.5-7B-Instruct as attacker, CompassJudger-2-32B-Instruct as judge, and
Qwen3-Embedding-4B for AutoDAN-R and DrAttack:

```bash
ATTACKER=models/Qwen2.5-7B-Instruct
EMBEDDING=models/Qwen3-Embedding-4B

OPENRT_TARGET_GPUS=0 \
OPENRT_ATTACKER_GPUS=1 \
OPENRT_JUDGE_GPUS=2 \
MOUSETRAP_TARGET_GPU=0 \
MOUSETRAP_JUDGE_GPU=1 \
python evaluation/run.py openrt \
  --model "$MODEL" \
  --attacker-model "$ATTACKER" \
  --judge-model "$JUDGE" \
  --embedding-model "$EMBEDDING"
```

The runner generates one deterministic, hashed, nine-prompts-per-behavior
Mousetrap bank if it does not already exist. Reuse the same bank for every
model condition. The final `*_paper_summary.json` reports the macro mean over
all 27 attacks.

Pass `--dry-run` to any subcommand to inspect the exact subprocess commands
without loading a model. Use `python evaluation/run.py BENCHMARK --help` for
benchmark-specific options.

## Third-party code and data

MMLU-Pro, HaluEval, and Flames retain only the upstream components used by this
project. HarmBench likewise retains only the DirectRequest evaluator and its
text behavior data. OpenRT keeps a pinned vendored package because its attack
implementations are runtime dependencies. Source revisions and licenses are
recorded in the benchmark directories and third-party notices.
