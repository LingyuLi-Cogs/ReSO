# HarmBench DirectRequest adapter

This directory contains only the HarmBench evaluation used by the project:
DirectRequest generation on the 300 standard/contextual behaviors in
`harmbench_behaviors_text_all.csv`, followed by classification with a local
`cais/HarmBench-Llama-2-13b-cls` checkpoint. Copyright/hash-check behaviors,
multimodal code, adversarial training, and other red-team baselines are outside
this harness.

```bash
python evaluation/run.py harmbench \
  --model checkpoints/target-model \
  --classifier-model models/HarmBench-Llama-2-13b-cls
```

Generation and classification run as separate processes so GPU memory is
released between stages. The headline metric is ASR; lower is safer. Raw
generations, classifier labels, skipped-length records, and an aggregate CSV
are written to the selected output directory.

The retained upstream data and classifier prompt are covered by `LICENSE`.
