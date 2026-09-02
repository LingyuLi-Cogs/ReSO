#!/usr/bin/env python3
"""Audit the OLD splits without loading a model or changing any dataset.

The synthetic RSA construction is an adversarial example, NOT an estimated
effect of leakage and NOT a solved maximum. It deliberately uses held-out
human labels to demonstrate that aggregate RSA need not track clean-only RSA.
Requires numpy and pandas, already listed in the repository requirements.
"""

import ast
import hashlib
import json
import math
from pathlib import Path
import unicodedata
import zlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"


def training_data_functions():
    """Execute only the existing data/metric definitions, avoiding torch imports."""
    path = ROOT / "training/reso_train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        "MoralItemBank", "_stratified_select", "sample_batch", "mine_triplets",
        "build_rsa_fixture", "build_val_triplet_fixture", "build_judgment_fixture",
        "_strength", "_avg_rank", "spearman",
    }
    constants = {"DIMENSIONS", "POLE_CODES", "PROMPT_TEMPLATES", "RESPONSE_TEMPLATES"}
    nodes = [node for node in tree.body if
             (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in constants
                 for target in node.targets))]
    namespace = {"np": np, "pd": pd, "json": json, "math": math, "zlib": zlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def synthetic_example(fn, fixture, texts, pair_overlap, targets):
    """Construct valid cosine Gram matrices; change only overlap-touching edges.

    Identical texts share one vector, even across foundation segments. Repeated
    text-pair human targets are averaged before setting their shared cosine.
    All off-diagonal cosines are tiny, so strict diagonal dominance certifies
    positive definiteness and hence a realizable unit-vector Gram matrix.
    The construction is a deliberately pessimistic mathematical example; it
    uses validation labels and is not a plausible-learning-probability model.
    """
    within = fixture["within"]
    human = fixture["sh"][within]
    unique, inverse = np.unique(texts, return_inverse=True)
    a = inverse[fixture["iu"][0][within]]
    b = inverse[fixture["iu"][1][within]]
    codes, pair_index = np.unique(
        np.minimum(a, b) * len(unique) + np.maximum(a, b), return_inverse=True)
    self_pair = codes // len(unique) == codes % len(unique)
    ranks = fn["_avg_rank"](human)
    target = np.bincount(pair_index, weights=ranks) / np.bincount(pair_index)
    target = (target - ranks.mean()) / ranks.std()
    touched = np.bincount(pair_index, weights=pair_overlap) > 0
    noise = np.random.default_rng(1234).normal(size=len(codes))
    epsilon = 1e-6

    def scores(alpha):
        values = noise.copy()
        values[touched] = (1 - alpha) * noise[touched] + alpha * target[touched]
        values *= epsilon
        values[self_pair] = 1.0
        return values, values[pair_index]

    states, clean_scores = [], []
    for goal in targets:
        low, high = 0.0, 1.0
        for _ in range(36):
            alpha = (low + high) / 2
            rho = fn["spearman"](scores(alpha)[1], human)
            if rho < goal:
                low = alpha
            else:
                high = alpha
        alpha = (low + high) / 2
        values, similarities = scores(alpha)
        clean_scores.append(similarities[~pair_overlap].copy())
        gram = np.eye(len(unique))
        g0, g1 = codes // len(unique), codes % len(unique)
        nonself = ~self_pair
        gram[g0[nonself], g1[nonself]] = values[nonself]
        gram[g1[nonself], g0[nonself]] = values[nonself]
        row_bound = float((np.abs(gram).sum(axis=1) - 1).max())
        assert row_bound < 1  # Positive definite by strict diagonal dominance.
        states.append({
            "target_full_rsa": goal, "alpha": alpha,
            "full_rsa": fn["spearman"](similarities, human),
            "clean_only_rsa": fn["spearman"](similarities[~pair_overlap], human[~pair_overlap]),
            "offdiagonal_absolute_row_sum_max": row_bound,
            "gram_min_eigenvalue_lower_bound": 1 - row_bound,
        })
    assert np.array_equal(clean_scores[0], clean_scores[1])
    return {
        "warning": "Synthetic, adversarial, calibrated to two reported RSA values using held-out human labels; NOT a model measurement, leakage estimate, probability, or exact maximum.",
        "clean_clean_cosines_identical": True,
        "unique_texts": len(unique), "noise_seed": 1234, "epsilon": epsilon,
        "states": states,
        "full_rsa_increase": states[1]["full_rsa"] - states[0]["full_rsa"],
    }


def main():
    fn = training_data_functions()
    frames = {split: pd.read_csv(DATA / f"social_chem_{split}_expanded.csv", keep_default_na=False)
              for split in ("train", "val", "test")}
    train_texts = set(frames["train"].action)
    normalize = lambda text: " ".join(unicodedata.normalize("NFC", text).split())
    normalized_train = {normalize(text) for text in train_texts}
    result = {"scope": {
        "dataset": "Original dataset/*.csv and *_buckets.json, not action_grouped",
        "fixture_seed": 42, "rsa_per_domain": 150, "judge_per_domain": 100,
        "actual_run_args_or_per_item_model_outputs_available": False,
        "overlap_definition": "Exact text occurs in training data bank; actual sampling exposure not established",
        "fixtures_conditional_on_current_code_defaults": True,
        "accuracy_bound_scope": "Rescoring the same checkpoint after excluding overlap, not the causal effect of retraining or checkpoint reselection",
    }, "full_splits": {}}
    for split in ("val", "test"):
        frame = frames[split]
        overlap = frame.action.isin(train_texts)
        result["full_splits"][split] = {
            "rows": len(frame), "overlap_rows": int(overlap.sum()),
            "fraction": float(overlap.mean()),
            "overlap_unique_actions": len(set(frame.action) & train_texts),
            "normalized_overlap_rows": sum(normalize(text) in normalized_train for text in frame.action),
        }

    bank = fn["MoralItemBank"](DATA / "social_chem_val_expanded.csv", DATA / "val_buckets.json")
    rng = np.random.default_rng([42, 9999])
    fixture = fn["build_rsa_fixture"](bank, 150, rng)
    texts = [bank.texts[row] for row in fixture["rows"]]
    overlap = np.array([text in train_texts for text in texts])
    i, j = fixture["iu"]
    within = fixture["within"]
    touched = overlap[i] | overlap[j]
    both = overlap[i] & overlap[j]
    result["rsa"] = {
        "items": len(texts), "overlap_items": int(overlap.sum()),
        "item_fraction": float(overlap.mean()),
        "valid_within_foundation_pairs": int(within.sum()),
        "pairs_touching_overlap": int((within & touched).sum()),
        "touched_pair_fraction": float(touched[within].mean()),
        "both_endpoints_overlap": int((within & both).sum()),
        "both_endpoints_fraction": float(both[within].mean()),
        "clean_pairs": int((within & ~touched).sum()),
        "per_domain": {},
    }
    for index, domain in enumerate(fn["DIMENSIONS"]):
        mask = within & (i >= index * 150) & (i < (index + 1) * 150)
        result["rsa"]["per_domain"][domain] = {
            "overlap_items": int(overlap[index * 150:(index + 1) * 150].sum()),
            "pairs": int(mask.sum()), "touched_pairs": int((mask & touched).sum()),
        }

    judge = fn["build_judgment_fixture"](bank, 100, np.random.default_rng([42, 5150]))
    n = len(judge)
    k = sum(item["text"] in train_texts for item in judge)
    q = k / n
    result["judgment_accuracy"] = {
        "items": n, "overlap_items": k, "clean_items": n - k,
        "overlap_fraction": q,
        "unconditional_full_vs_clean_accuracy_difference_bound": q,
        "formula": "max(0, (N*A-k)/(N-k)) <= A_clean <= min(1, N*A/(N-k))",
        "illustrative_not_measured": {
            str(accuracy): {"clean_accuracy_min": max(0, (accuracy - q) / (1 - q)),
                            "clean_accuracy_max": min(1, accuracy / (1 - q)),
                            "max_full_minus_clean": accuracy - max(0, (accuracy - q) / (1 - q))}
            for accuracy in (0.69, 0.70, 0.75, 0.78)
        },
    }
    trips = fn["build_val_triplet_fixture"](
        bank, 8, 260, dict(delta_h=0.2, cap_k=8, proto_anchor_cross=False, proto_threshold=0.75), rng)
    n_triplets, n_touched = 0, 0
    for batch in trips:
        seen = np.array([bank.texts[row] in train_texts for row in batch["rows"]])
        triplet = batch["trip"]
        mask = seen[triplet["i"]] | seen[triplet["j"]] | seen[triplet["k"]]
        n_triplets += len(mask)
        n_touched += int(mask.sum())
    result["triplet_accuracy"] = {"triplets": n_triplets, "touched_triplets": n_touched,
                                  "fraction": n_touched / n_triplets}

    curve_path = ROOT / "original data/rsa_asr_pooled_all.points.csv"
    curve = pd.read_csv(curve_path)
    curve = curve[curve.arm == "reso"]
    goals = [float(curve.loc[curve.step == 0, "rsa"].iloc[0]), float(curve.rsa.max())]
    result["synthetic_rsa_counterexample"] = synthetic_example(fn, fixture, texts, touched[within], goals)
    paths = [ROOT / "training/reso_train.py", curve_path]
    paths += [DATA / f"social_chem_{split}_expanded.csv" for split in ("train", "val", "test")]
    paths.append(DATA / "val_buckets.json")
    result["input_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths}
    target = OUT / "results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(target), "judgment": result["judgment_accuracy"],
                      "rsa": result["rsa"], "synthetic": result["synthetic_rsa_counterexample"]}, indent=2))


if __name__ == "__main__":
    main()
