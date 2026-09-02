#!/usr/bin/env python3
"""Combine the 26-method OpenRT core and fixed Mousetrap into the paper metric."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-summary", nargs="+", required=True)
    parser.add_argument("--mousetrap-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mouse = json.loads(Path(args.mousetrap_summary).read_text(encoding="utf-8"))
    per_attack = {}
    for summary_path in args.core_summary:
        core = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for name, row in core["per_attack"].items():
            if name in per_attack:
                raise SystemExit(f"duplicate core attack: {name}")
            per_attack[name] = row["asr"]
    if set(per_attack) & {"autodan_turbo", "rainbow_teaming", "mousetrap"}:
        raise SystemExit("core summary contains a method excluded from the project suite")
    if len(per_attack) != 26 or any(value is None for value in per_attack.values()):
        raise SystemExit("expected 26 complete core attack results")
    mouse_asr = mouse["headline"]["any_at_k"]["asr"]
    if mouse_asr is None:
        raise SystemExit("Mousetrap Any@9 ASR is incomplete")
    per_attack["mousetrap_any_at_9"] = mouse_asr
    result = {
        "metric": "macro ASR over the 27 project OpenRT attacks",
        "n_attacks": 27,
        "macro_asr": sum(per_attack.values()) / len(per_attack),
        "per_attack": per_attack,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
