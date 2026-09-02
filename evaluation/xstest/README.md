# XSTest adapter

The project evaluates all 250 safe and 200 unsafe XSTest prompts with the
official three-class compliance/refusal rubric and reports balanced accuracy.
CompassJudger-2-32B-Instruct is kept fixed across model conditions.

```bash
python evaluation/run.py xstest \
  --model checkpoints/target-model \
  --judge-model models/CompassJudger-2-32B-Instruct
```

The dataset revision and SHA256 are verified at runtime. Results retain raw
generations, judge outputs, parse coverage, safe/unsafe rates, and conservative
bounds. See `THIRD_PARTY_NOTICES.md` and `data/README.md` for provenance.
