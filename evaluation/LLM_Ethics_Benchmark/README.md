# LLM Ethics Benchmark adapter

The project evaluator runs the three instruments used in
[LLM Ethics Benchmark](https://github.com/The-Responsible-AI-Initiative/LLM_Ethics_Benchmark):
MFQ-30, World Values Survey, and moral dilemmas. It generates locally and uses
the benchmark's deterministic scoring heuristics; no API provider is required.

```bash
python evaluation/run.py ethics --model checkpoints/target-model
```

Only the evaluator and instrument JSON files are retained. The unused upstream
API clients, images, tests, and debug programs are omitted. See `LICENSE` for
the upstream license.
