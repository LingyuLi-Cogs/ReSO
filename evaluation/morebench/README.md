# MoReBench adapter

The local evaluator reproduces MoReBench's response generation, rubric judging,
and aggregation flow. The project reports the 500-item main benchmark and uses
CompassJudger-2-32B-Instruct as the fixed local judge.

```bash
python evaluation/run.py morebench \
  --model checkpoints/target-model \
  --judge-model models/CompassJudger-2-32B-Instruct
```

The main and theory CSVs are retained for provenance, but the unified project
launcher selects `--dataset main`. Dataset source and license are documented in
`data/README.md`.
