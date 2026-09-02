# Training

## Entrypoints

| Script | Purpose |
| --- | --- |
| `reso_train.py` | Full-parameter ReSO for standard model sizes |
| `dpo_train.py` | Matched full-parameter DPO baseline |
| `reso_train_xl.py` | Memory-aware ReSO for 32B+ models |
| `dpo_train_xl.py` | Memory-aware DPO for 32B+ models |
| `fsdp_xl.py` | Shared XL/FSDP loading and sharding helpers |
| `prepare_replay.py` | Generate the replay corpus |

## Required replay corpus

Generate the replay corpus before launching ReSO:

```bash
pip install datasets
python training/prepare_replay.py \
  --output dataset/replay.jsonl \
  --n_docs 50000
```

## Portable configurations

```bash
MODEL_PATH=/path/to/model bash training/configs/dpo.sh
MODEL_PATH=/path/to/model bash training/configs/reso.sh
MODEL_PATH=/path/to/model SHUFFLE_SEED=42 \
  bash training/configs/shuffle.sh
```
