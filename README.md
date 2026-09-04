# Representational alignment yields generalizable safety in language models

## Repository guide

- [`dataset/`](dataset/README.md): processed Social Chemistry splits, moral-category
  indices, and the replay-data schema.
- [`comparison/`](comparison/README.md): hidden-state extraction, linear probing,
  category-center comparison, and analysis scripts.
- [`training/`](training/README.md): ReSO, DPO, and shuffled-control trainers with
  portable launch configurations.
- [`evaluation/`](evaluation/README.md): the nine evaluation harnesses used in the
  manuscript, including the project-specific integrations for external benchmarks.

## Minimal reproduction

The replay corpus is intentionally not distributed. Before running ReSO,
generate `dataset/replay.jsonl` locally:

```bash
pip install datasets
python training/prepare_replay.py \
  --output dataset/replay.jsonl \
  --n_docs 50000
```

Keep the generated replay file fixed across experiments. Offline construction
and the expected schema are documented in [`dataset/README.md`](dataset/README.md).

Set `MODEL_PATH` to a Hugging Face model identifier or local model directory,
then launch an experiment:

```bash
MODEL_PATH=checkpoints/base-model bash training/configs/reso.sh
MODEL_PATH=checkpoints/base-model bash training/configs/dpo.sh
MODEL_PATH=checkpoints/base-model bash training/configs/shuffle.sh
```

Detailed training, representation-comparison, and evaluation commands are kept
in the corresponding subdirectory READMEs so that this page remains a concise
entry point for review.
