# Flames adapter

This is the two-stage evaluation path used by the project for
[Flames](https://github.com/AIFlames/Flames):

1. `basemodel.py` generates one response for each prompt with the target model.
2. `infer.py` applies the official five-dimension Flames scorer and reports the
   harmless rate and harmless score.

The public repository does not redistribute the Flames 1K Chinese input file
or scorer weights. Supply local copies at runtime:

```bash
python evaluation/run.py flames \
  --model checkpoints/target-model \
  --data-path data/Flames_1k_Chinese.jsonl \
  --scorer-model models/flames-scorer
```

The InternLM configuration, tokenizer, and classifier implementation retained
here are required to load the scorer and carry their original Apache-2.0
notices in the source headers.
