#!/usr/bin/env python3
"""Prepare stratified moral-representation samples from Social Chemistry.

The checked-in comparison sample is stored under ``dataset/comparison``. This
script reproduces it from either raw Social Chemistry rows, cleaned rows, or
precomputed human moral vectors. ZIP inputs are read directly.
"""

import argparse
import ast
import json
import random
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DIMENSIONS = (
    "care-harm",
    "fairness-cheating",
    "loyalty-betrayal",
    "authority-subversion",
    "sanctity-degradation",
)
DIMENSION_INDICES = {dimension: (2 * i, 2 * i + 1)
                     for i, dimension in enumerate(DIMENSIONS)}
NON_TYPICAL_SCORES = (0.125, 0.25, 0.375, 0.5, 0.75)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "dataset" / "comparison"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "comparison" / "artifacts" / "preparation"


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV or a ZIP containing one non-metadata table."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, **kwargs)

    with ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if not name.endswith("/") and not name.startswith("__MACOSX/")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected one data file in {path}, found {len(members)}: {members}"
            )
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream, **kwargs)


def clean_social_chemistry(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the original filtering and Moral Foundations encoding."""
    required = (
        "action",
        "rot-moral-foundations",
        "action-moral-judgment",
        "action-agree",
    )
    cleaned = df[(df["rot-bad"] == 0) & (df["m"] == 1)].copy()
    cleaned = cleaned.dropna(subset=list(required))

    def foundation_vector(value):
        vector = [0] * 10
        if isinstance(value, str):
            for foundation in (part.strip() for part in value.split("|")):
                if foundation in DIMENSION_INDICES:
                    virtue_idx, vice_idx = DIMENSION_INDICES[foundation]
                    vector[virtue_idx] = 1
                    vector[vice_idx] = 1
        return vector

    cleaned["mft-vector"] = cleaned["rot-moral-foundations"].map(
        foundation_vector
    )
    return cleaned


def compute_human_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Convert worker judgments into ten-dimensional moral vectors."""
    vectors = df.copy()
    if "id" not in vectors:
        vectors["id"] = [f"socialchem_{i:06d}" for i in range(len(vectors))]

    def parse_vector(value):
        return ast.literal_eval(value) if isinstance(value, str) else list(value)

    vectors["mft-vector"] = vectors["mft-vector"].map(parse_vector)
    signed_score = vectors["action-moral-judgment"].astype(float) / 2.0
    confidence = vectors["action-agree"].astype(float) / 4.0
    vectors["m_virtue"] = signed_score.clip(lower=0) * confidence
    vectors["m_vice"] = (-signed_score).clip(lower=0) * confidence

    def apply_scores(row):
        result = []
        for index, present in enumerate(row["mft-vector"]):
            if not present:
                result.append(0.0)
            elif index % 2 == 0:
                result.append(float(row["m_virtue"]))
            else:
                result.append(float(row["m_vice"]))
        return result

    vectors["moral_vector"] = vectors.apply(apply_scores, axis=1)
    return vectors[[
        "id",
        "action",
        "rot-moral-foundations",
        "moral_vector",
        "m_virtue",
        "m_vice",
    ]]


def sample_stratified(
    vectors: pd.DataFrame,
    *,
    seed: int,
    per_score: int,
    typical: int,
    neutral: int,
) -> list[dict]:
    """Flatten multi-foundation actions, then sample within each dimension."""
    vectors = vectors.copy()
    vectors["moral_vector"] = vectors["moral_vector"].map(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else list(value)
    )

    expanded = []
    for row in vectors.to_dict(orient="records"):
        foundations = row.get("rot-moral-foundations")
        if not isinstance(foundations, str):
            continue
        for foundation in (part.strip() for part in foundations.split("|")):
            if foundation in DIMENSION_INDICES:
                item = dict(row)
                item["sampled_dimension"] = foundation
                expanded.append(item)

    rng = random.Random(seed)
    sampled = []

    def take(records, count, sample_type, score):
        chosen = rng.sample(records, min(count, len(records)))
        for record in chosen:
            item = dict(record)
            item["sample_type"] = sample_type
            item["score_stratum"] = score
            sampled.append(item)

    for dimension in DIMENSIONS:
        virtue_idx, vice_idx = DIMENSION_INDICES[dimension]
        rows = [row for row in expanded if row["sampled_dimension"] == dimension]
        virtue_rows = [row for row in rows if row["moral_vector"][virtue_idx] > 0]
        vice_rows = [row for row in rows if row["moral_vector"][vice_idx] > 0]
        neutral_rows = [
            row for row in rows
            if row["moral_vector"][virtue_idx] == 0
            and row["moral_vector"][vice_idx] == 0
        ]

        for score in NON_TYPICAL_SCORES:
            take(
                [row for row in virtue_rows
                 if abs(row["moral_vector"][virtue_idx] - score) < 1e-6],
                per_score,
                "virtue",
                score,
            )

        take(
            [row for row in virtue_rows
             if abs(row["moral_vector"][virtue_idx] - 1.0) < 1e-6],
            typical,
            "virtue_typical",
            1.0,
        )

        for score in NON_TYPICAL_SCORES:
            take(
                [row for row in vice_rows
                 if abs(row["moral_vector"][vice_idx] - score) < 1e-6],
                per_score,
                "vice",
                score,
            )

        take(
            [row for row in vice_rows
             if abs(row["moral_vector"][vice_idx] - 1.0) < 1e-6],
            typical,
            "vice_typical",
            1.0,
        )
        take(neutral_rows, neutral, "neutral", 0.0)

    fields = (
        "id",
        "action",
        "rot-moral-foundations",
        "moral_vector",
        "m_virtue",
        "m_vice",
        "sampled_dimension",
        "sample_type",
        "score_stratum",
    )
    return [{field: row[field] for field in fields} for row in sampled]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stratified moral-representation extraction samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "human-representation-vectors.tsv.zip",
        help="Raw, cleaned, or vectorized Social Chemistry table",
    )
    parser.add_argument(
        "--input-stage",
        choices=("raw", "clean", "vectors"),
        default="vectors",
        help="Processing stage represented by --input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sampled-human-representation-vectors.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-score", type=int, default=300)
    parser.add_argument("--typical", type=int, default=150)
    parser.add_argument("--neutral", type=int, default=300)
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Write cleaned.csv and human-representation-vectors.csv beside --output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    read_kwargs = {"sep": "\t"} if args.input_stage == "raw" else {}
    table = read_table(args.input, **read_kwargs)

    cleaned = clean_social_chemistry(table) if args.input_stage == "raw" else table
    vectors = (
        compute_human_vectors(cleaned)
        if args.input_stage in {"raw", "clean"}
        else cleaned
    )
    samples = sample_stratified(
        vectors,
        seed=args.seed,
        per_score=args.per_score,
        typical=args.typical,
        neutral=args.neutral,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if args.save_intermediates:
        cleaned.to_csv(args.output.parent / "cleaned.csv", index=False)
        vectors.to_csv(
            args.output.parent / "human-representation-vectors.csv", index=False
        )

    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
