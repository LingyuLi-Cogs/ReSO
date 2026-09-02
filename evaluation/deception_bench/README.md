# DeceptionBench adapter

The project setting evaluates all 150 scenarios across five domains and the six
single-turn L1/L2 conditions. It reports thought- and response-channel deception
rates using one fixed CompassJudger-2-32B-Instruct checkpoint.

```bash
python evaluation/run.py deception \
  --model checkpoints/target-model \
  --judge-model models/CompassJudger-2-32B-Instruct
```

The optional multi-turn L3 path remains implemented for analysis but is not
enabled by the project launcher. Dataset source, research-use restriction, and
license are documented in `data/README.md`.
