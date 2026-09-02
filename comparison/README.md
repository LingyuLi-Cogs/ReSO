# Representation comparison

This directory implements a three-stage, model-agnostic pipeline for comparing
human moral annotations with internal language-model representations.

```text
dataset/comparison/*.jsonl
          |
          v
extraction/        hidden states, mean pooling, last-token pooling
          |
          v
comparison/        linear probes, category centers, cosine similarities
          |
          v
analysis/          human/model correlations and cross-model plots
```

All generated tensors, probe weights, tables, and plots are written below
`comparison/artifacts/`. These run artifacts should not be committed. The
scripts contain no machine-specific model roster or checkpoint path.

## Layout

| Path | Purpose |
| --- | --- |
| `extraction/prepare_samples.py` | Rebuild the stratified comparison sample |
| `extraction/extract_activations.py` | Extract selected decoder-layer states |
| `comparison/linear_probe.py` | Predict ten-dimensional human vectors from states |
| `comparison/category_centers.py` | Build anisotropy-corrected category centers |
| `analysis/correlation_analysis.py` | Correlate model similarity with human scores |
| `analysis/plot_correlations.py` | Compare peak category correlations across models |
| `analysis/plot_category_centers.py` | Analyze layer-wise category geometry |
| `configs/pipeline.sh` | Run the complete pipeline for one model |

## Complete pipeline

From the repository root:

```bash
MODEL_PATH=/path/to/model \
MODEL_NAME=my-model \
  bash comparison/configs/pipeline.sh
```

The pipeline extracts every decoder layer, merges the extraction batches, then
runs both comparison methods and their analyses. Useful overrides include
`DATA_PATH`, `ARTIFACTS_DIR`, `BATCH_SIZE`, `DTYPE`, `PYTHON_BIN`, and
`ALLOW_NETWORK=1`.

## Run stages separately

Prepare a new stratified sample from the supplied human-vector table:

```bash
python comparison/extraction/prepare_samples.py
```

The default output is under `comparison/artifacts/preparation`; the checked-in
sample used by extraction remains
`dataset/comparison/sampled-human-representation-vectors.jsonl`.

Extract model representations:

```bash
python comparison/extraction/extract_activations.py \
  --model_path /path/to/model \
  --model_name my-model \
  --all_layers --merge_batches
```

Run the two comparison branches and the analyses:

```bash
python comparison/comparison/linear_probe.py --model my-model=my-model
python comparison/comparison/category_centers.py --models my-model
python comparison/analysis/correlation_analysis.py --models my-model
python comparison/analysis/plot_correlations.py --models my-model
python comparison/analysis/plot_category_centers.py --models my-model
```

When model arguments are omitted, the scripts discover available input folders
from `comparison/artifacts` automatically.

## Data semantics

One action may belong to multiple moral dimensions. Sample preparation
intentionally flattens such actions into one record per `sampled_dimension`,
while retaining the same source ID, action, scores, and moral vector. This is
the same intentional flattening used by the training dataset.
