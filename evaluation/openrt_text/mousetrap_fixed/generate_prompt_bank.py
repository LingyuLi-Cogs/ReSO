#!/usr/bin/env python3
"""Generate a deterministic, reusable Mousetrap prompt bank."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterator, List, Sequence


HERE = Path(__file__).resolve().parent
OPENRT_TEXT_DIR = HERE.parent
VENDOR_DIR = OPENRT_TEXT_DIR / "vendor"
for path in (OPENRT_TEXT_DIR, VENDOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openrt_common import file_sha256, load_harmbench_rows
from OpenRT.attacks.blackbox.implementations.mousetrap import MousetrapAttack


GENERATOR_VERSION = "mousetrap-fixed-prompts-v1"
OPENRT_COMMIT = "365652f52c05c63324687ae69d4350499db9264c"
PROFILE_SHAPES = {
    "smoke": (1, 1),
    "standard": (3, 3),
    # The OpenRT runner caps Mousetrap at three available chaos levels.
    "paper": (3, 3),
}


class _UnusedModel:
    def query(self, *_: Any, **__: Any) -> str:
        raise AssertionError("prompt generation must not query a model")


def ascii_safe_atbash(text: str):
    """Upstream-equivalent Atbash for ASCII, preserving other Unicode letters.

    Upstream checks ``str.isalpha()`` but then assumes an ASCII code point,
    which raises ``ValueError`` for some official HarmBench rows. Mousetrap's
    own description defines this transform over the 26-letter alphabet, so
    non-ASCII characters should remain unchanged like spaces and punctuation.
    """
    transformed = ""
    for char in text:
        if "A" <= char <= "Z":
            transformed += chr(90 - (ord(char) - 65))
        elif "a" <= char <= "z":
            transformed += chr(122 - (ord(char) - 97))
        else:
            transformed += char
    policy = (
        "Replace each letter in string (at position x in the alphabet) with "
        "the letter at position (26 minus x) in the alphabet. For example, A "
        "is replaced by Z and B is replaced by Y. Spaces, numbers, and special "
        "characters remain unchanged."
    )
    return text, policy, policy, transformed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@contextmanager
def isolated_random(seed: int) -> Iterator[None]:
    """Use upstream's global ``random`` calls without leaking RNG state."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def manifest_path_for(prompt_path: Path) -> Path:
    return prompt_path.with_name(f"{prompt_path.stem}.manifest.json")


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_prompt(
    attack: MousetrapAttack,
    instruction: str,
    iteration: int,
    chaos_params: Sequence[int],
) -> Dict[str, Any]:
    """Run the upstream Mousetrap transform pipeline without model inference."""
    transformed = instruction
    encryption_steps: List[str] = []
    decryption_steps: List[str] = []
    for level in range(iteration):
        _, encryption, decryption, ciphertext = attack._chaos_functions(
            transformed, chaos_params[level]
        )
        encryption_steps.append(encryption)
        decryption_steps.append(decryption)
        transformed = ciphertext
    decryption_steps.reverse()
    return {
        "prompt": attack._attack_prompt(transformed, decryption_steps),
        "ctq": transformed,
        "ecps": encryption_steps,
        "dcps": decryption_steps,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and store the exact Mousetrap prompts that will be reused "
            "for every evaluated model. No model is loaded or queried."
        )
    )
    parser.add_argument("--output", required=True, help="output prompts JSONL")
    parser.add_argument(
        "--data_dir", default=str(OPENRT_TEXT_DIR / "data"),
        help="directory containing the offline HarmBench CSV",
    )
    parser.add_argument(
        "--functional_categories", nargs="+", default=["standard", "contextual"]
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_SHAPES), default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--question_template", choices=("auto", "policeman", "playwright"),
        default="auto",
    )
    parser.add_argument(
        "--chaos_params", nargs=3, type=int, choices=(1, 2, 3), default=None,
        metavar=("LEVEL1", "LEVEL2", "LEVEL3"),
        help="explicit permutation of 1,2,3; default is a seeded permutation",
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--trials_per_iteration", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_verify_hash", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_iterations, profile_trials = PROFILE_SHAPES[args.profile]
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None else profile_iterations
    )
    trials_per_iteration = (
        args.trials_per_iteration
        if args.trials_per_iteration is not None else profile_trials
    )
    if not 1 <= max_iterations <= 3:
        raise SystemExit("--max_iterations must be between 1 and 3")
    if trials_per_iteration <= 0:
        raise SystemExit("--trials_per_iteration must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    output_path = Path(args.output).expanduser().resolve()
    manifest_path = manifest_path_for(output_path)
    if not args.force and (output_path.exists() or manifest_path.exists()):
        raise SystemExit(
            f"refusing to overwrite an existing prompt bank: {output_path}; "
            "pass --force only when intentionally replacing it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_rng = random.Random(args.seed)
    chaos_params = (
        list(args.chaos_params)
        if args.chaos_params is not None
        else config_rng.sample([1, 2, 3], 3)
    )
    if sorted(chaos_params) != [1, 2, 3]:
        raise SystemExit("--chaos_params must be a permutation of 1 2 3")
    question_template = args.question_template
    if question_template == "auto":
        question_template = config_rng.choice(
            list(MousetrapAttack.QUESTION_TEMPLATES)
        )

    tasks, dataset_sha256 = load_harmbench_rows(
        args.data_dir,
        args.functional_categories,
        debug=args.debug,
        limit=args.limit,
        verify_hash=not args.no_verify_hash,
    )
    attack = MousetrapAttack(
        _UnusedModel(),
        judge=None,
        max_iterations=max_iterations,
        trials_per_iteration=trials_per_iteration,
        chaos_params=chaos_params,
        question_template=question_template,
        verbose=False,
    )
    # Local compatibility patch; the vendored OpenRT source remains untouched.
    attack._atbash = ascii_safe_atbash
    upstream_chaos_functions = attack._chaos_functions

    def compatible_chaos_functions(text: str, param: int):
        state = random.getstate()
        try:
            return upstream_chaos_functions(text, param)
        except ValueError as error:
            if str(error) != (
                "Not enough words in the word list to create a one-to-one mapping."
            ):
                raise
            # The only other upstream level-2 transform is reverse-by-words.
            # Treat an exhausted substitution vocabulary as an invalid draw and
            # use that valid alternative without changing later RNG state.
            random.setstate(state)
            return attack._reverse_by_words(text)

    attack._chaos_functions = compatible_chaos_functions

    temporary = output_path.with_name(output_path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for task in tasks:
            for iteration in range(1, max_iterations + 1):
                for trial in range(1, trials_per_iteration + 1):
                    prompt_id = f"{task['item_id']}__depth{iteration}__trial{trial}"
                    prompt_seed = stable_seed(
                        GENERATOR_VERSION, args.seed, task["item_id"], iteration, trial
                    )
                    with isolated_random(prompt_seed):
                        generated = build_prompt(
                            attack, task["instruction"], iteration, chaos_params
                        )
                    record = {
                        "prompt_id": prompt_id,
                        "item_id": task["item_id"],
                        "source_index": task["source_index"],
                        "behavior": task["behavior"],
                        "context": task["context"],
                        "instruction": task["instruction"],
                        "functional_category": task["functional_category"],
                        "semantic_category": task["semantic_category"],
                        "iteration": iteration,
                        "trial": trial,
                        "prompt_seed": prompt_seed,
                        "question_template": question_template,
                        "chaos_params": chaos_params[:iteration],
                        **generated,
                    }
                    record["prompt_sha256"] = sha256_text(record["prompt"])
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": utc_now(),
        "openrt_commit": OPENRT_COMMIT,
        "prompt_bank": str(output_path),
        "prompt_bank_sha256": file_sha256(output_path),
        "dataset_sha256": dataset_sha256,
        "functional_categories": args.functional_categories,
        "profile": args.profile,
        "seed": args.seed,
        "question_template": question_template,
        "chaos_params": chaos_params,
        "max_iterations": max_iterations,
        "trials_per_iteration": trials_per_iteration,
        "compatibility_fixes": [
            "atbash_ascii_alphabet_preserve_non_ascii_v1",
            "word_substitution_exhaustion_fallback_reverse_words_v1",
        ],
        "n_behaviors": len(tasks),
        "n_prompts": count,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Wrote {count} fixed prompts for {len(tasks)} behaviors: {output_path}")
    print(f"Prompt bank SHA256: {manifest['prompt_bank_sha256']}")
    print(f"Manifest: {manifest_path}")
    print(
        "Reuse this unchanged prompt bank for every model; do not regenerate it "
        "between model comparisons."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
