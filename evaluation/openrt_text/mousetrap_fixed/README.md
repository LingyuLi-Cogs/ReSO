# Fixed Mousetrap evaluation

`generate_prompt_bank.py` deterministically creates nine prompts per HarmBench
behavior and records their SHA256 in a companion manifest. Generate the bank
once and reuse it for every training condition:

```bash
python evaluation/openrt_text/mousetrap_fixed/generate_prompt_bank.py \
  --output evaluation/outputs/openrt/mousetrap_seed42.jsonl \
  --profile standard --seed 42
```

`run_mousetrap.sh` launches loopback-only target and judge vLLM services on two
different GPUs, evaluates the fixed bank, and reports Any@9 ASR:

```bash
MOUSETRAP_TARGET_GPU=0 MOUSETRAP_JUDGE_GPU=1 \
bash evaluation/openrt_text/mousetrap_fixed/run_mousetrap.sh \
  --model_path checkpoints/target-model \
  --judge_model_path models/CompassJudger-2-32B-Instruct \
  --prompt_bank evaluation/outputs/openrt/mousetrap_seed42.jsonl
```

The unified `evaluation/run.py openrt` command performs both steps when needed.
GPT-OSS checkpoints are detected from `config.json`; the launcher enables the
vLLM reasoning parser and passes `reasoning_effort=low` so only the final answer
is evaluated. If the checkpoint needs an offline Harmony vocabulary, provide it
through the standard `TIKTOKEN_ENCODINGS_BASE` environment variable.
