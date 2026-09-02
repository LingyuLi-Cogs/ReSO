# Dataset

This directory contains the training and representation-comparison data.

## Social Chemistry splits

| File | Expanded rows | Unique annotation IDs |
| --- | ---: | ---: |
| `social_chem_train_expanded.csv` | 201,023 | 162,744 |
| `social_chem_val_expanded.csv` | 25,170 | 20,343 |
| `social_chem_test_expanded.csv` | 25,141 | 20,344 |

### Split rationale

The Social Chemistry train, validation, and test sets are partitioned by the
source action/annotation `id`. This is an annotation-level split: all rows
derived from one source ID are assigned to one split before multi-foundation
annotations are flattened. Consequently, the several `target_dimension` rows
produced from one annotation can never be divided across training and
evaluation. A source action tagged with multiple moral foundations therefore
appears once per `target_dimension` while retaining the same `id`, action text,
scores, and 10-dimensional `moral_vector`.

This split unit matches the supervision used by the method, because the moral
foundation, polarity, typicality, and moral vector are properties of a source
annotation rather than of a globally unique surface string. Distinct source
annotations may use the same action text while recording different contexts,
rules of thumb, foundation assignments, or judgments. They remain distinct
labeled observations under the annotation-level estimand and may therefore
occur in different splits. The split should accordingly be interpreted as
measuring generalization to held-out annotations, rather than as a guarantee
that every surface-form action string is disjoint across splits.

### Independence of external benchmarks

The nine out-of-distribution benchmarks are separate evaluation corpora; they
are not constructed by repartitioning or reusing the Social Chemistry records.
No benchmark example contributes to the ReSO, DPO, or shuffled-control training
loss. Thus, overlap among independently annotated Social Chemistry records does
not create train--test duplication with these external benchmark suites. This
statement concerns dataset provenance rather than guaranteed absence of
incidental lexical similarity between independently collected corpora. In the
optional instrumented runs, HarmBench is logged as a passive step-wise monitor;
it does not enter the training loss, checkpoint selection, or early-stopping
rule. See [`../evaluation/`](../evaluation/README.md) and
[`../training/step_monitoring/`](../training/step_monitoring/README.md) for the
corresponding harness and monitoring details.

CSV columns:

- `row_id`: contiguous row index within a split.
- `id`: source annotation identifier used for train/validation/test splitting.
- `action`: the raw action text supplied to the representation encoder.
- `rot-moral-foundations`: pipe-separated source foundation labels.
- `moral_vector`: 10-dimensional virtue/vice membership vector.
- `m_virtue`, `m_vice`: annotation strength for the two poles.
- `target_dimension`: the flattened moral dimension for this row.
- `split`: `train`, `val`, or `test`.

The matching `*_buckets.json` files index each CSV by moral dimension, pole,
and annotation strength.

## Representation-comparison data

`comparison/` contains the inputs used by the representation pipeline:

| File | Role |
| --- | --- |
| `social-chem-101.v1.0.tsv.zip` | Original Social Chemistry source table |
| `social-chem-cleaned.tsv.zip` | Filtered source rows |
| `human-representation-vectors.tsv.zip` | Ten-dimensional human moral vectors |
| `sampled-human-representation-vectors.jsonl` | Stratified extraction sample |

The JSONL sample records `sampled_dimension` and `sample_type`. Multi-foundation
actions are intentionally flattened once per sampled dimension, preserving the
source action ID and annotation values.

## Replay corpus

`replay.jsonl` is not included. Generate it before running ReSO:

```bash
pip install datasets
python training/prepare_replay.py \
  --output dataset/replay.jsonl \
  --n_docs 50000
```

Each line of the generated file is either:

```json
{"text": "..."}
```

or:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
