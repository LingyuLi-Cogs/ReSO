# HaluEval adapter

This directory is the project-used evaluation subset of
[RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval). It contains the
QA, dialogue, and summarization data, the three official evaluation
instructions, and the local-vLLM evaluator. Dataset generation, API evaluation,
analysis code, IDE files, and historical outputs were intentionally omitted.

Run from the repository root:

```bash
python evaluation/run.py halueval --model checkpoints/target-model
```

The project setting runs all three tasks, samples the right or hallucinated
candidate with the evaluator's fixed seed, uses greedy decoding and the chat
template, and allows a 20,000-token context plus 4,096 generated tokens. The
headline score is accuracy over Yes/No hallucination decisions.

The upstream dataset and code are covered by `LICENSE`.
