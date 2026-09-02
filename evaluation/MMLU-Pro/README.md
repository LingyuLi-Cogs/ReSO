# MMLU-Pro adapter

This is the project-used, local-vLLM subset of
[TIGER-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro). It retains the
five-shot CoT prompt, saved test/validation dataset, and one evaluator. API
clients, alternative prompt experiments, notebooks, and historical results are
not part of this reproduction harness.

Run from the repository root:

```bash
python evaluation/run.py mmlu-pro --model checkpoints/target-model
```

Defaults match the project run: all 14 subjects, five-shot CoT, context length
20,000, up to 1,024 new tokens, and tensor parallel size 8. Results, per-subject
answers, and the aggregate accuracy are written below
`evaluation/outputs/mmlu-pro/` unless `--output-dir` is supplied.

Upstream code and data are distributed under the license in `LICENSE`.
