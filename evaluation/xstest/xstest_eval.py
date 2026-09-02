#!/usr/bin/env python3
"""One-command offline XSTest evaluation with local target and judge models.

Examples:
  python xstest/xstest_eval.py \
      --model_path checkpoints/target --judge_model_path models/local-judge

  torchrun --standalone --nproc_per_node=8 xstest/xstest_eval.py \
      --model_path checkpoints/target --judge_model_path models/local-judge
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set these before importing Transformers or Hugging Face Hub code.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.common import (barrier, default_model_name, dist_is_on,  # noqa: E402
                               load_hf_model, render_prompt,
                               setup_distributed)

from xstest_common import (JUDGE_INVALID, JUDGE_PROMPT_VERSION,  # noqa: E402
                           LABEL_FULL_COMPLIANCE, LABEL_FULL_REFUSAL,
                           PARSER_VERSION, XSTEST_COMMIT,
                           build_judge_prompt, load_xstest_rows,
                           parse_judge_label, split_thinking,
                           summarize_judgements)


COMPAT_KEYS = (
    "dataset",
    "dataset_sha256",
    "judge_model_path",
    "debug",
    "limit",
    "max_new_tokens",
    "judge_max_new_tokens",
    "max_input_tokens",
    "seed",
    "do_sample",
    "temperature",
    "gen_batch",
    "judge_batch",
    "attn_impl",
    "world_size",
    "judge_prompt_version",
    "parser_version",
    "xstest_commit",
)


def clean_name(name):
    return "".join(char if (char.isalnum() or char in "-_.") else "_"
                   for char in name)


def load_chat_model(path, args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise SystemExit(f"Tokenizer at {path} has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_hf_model(path, args.attn_impl, args.local_files_only, dtype)
    model.config.use_cache = True
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, tokenizer


def model_input_cap(model, requested, max_new_tokens):
    context = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(context, int) or context <= 0:
        return requested
    return min(requested, max(1, context - max_new_tokens))


def render_with_system(tokenizer, user_prompt, system_prompt):
    """Render the official classifier system/user messages."""
    if tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            merged = f"{system_prompt}\n\n{user_prompt}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


@torch.no_grad()
def generate_local(model, tokenizer, prompts, device, max_new_tokens,
                   batch_size, max_input_tokens, seed, do_sample=False,
                   temperature=0.0, desc="", system_prompt=None):
    outputs = []
    input_cap = model_input_cap(model, max_input_tokens, max_new_tokens)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        if system_prompt:
            rendered = [render_with_system(tokenizer, prompt, system_prompt)
                        for prompt in chunk]
        else:
            rendered = [render_prompt(tokenizer, prompt) for prompt in chunk]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_cap,
            add_special_tokens=False,
        ).to(device)
        input_tokens = encoded["attention_mask"].sum(dim=1).tolist()
        torch.manual_seed(seed * 100003 + start)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_args.update(temperature=temperature, top_p=0.95)
        generated = model.generate(**encoded, **generation_args)
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        for index, token_row in enumerate(new_tokens):
            raw = tokenizer.decode(token_row, skip_special_tokens=True).strip()
            output_tokens = len(
                tokenizer(raw, add_special_tokens=False)["input_ids"]
            )
            outputs.append({
                "raw": raw,
                "input_tokens": int(input_tokens[index]),
                "output_tokens": int(output_tokens),
                "input_truncated_to": input_cap,
            })
        if desc:
            done = min(start + batch_size, len(prompts))
            print(f"    {desc}: {done}/{len(prompts)}", flush=True)
    return outputs


def gather_to_main(records, rank, world, total_items, chunk_size=256):
    """Gather bounded Python-object chunks to rank 0 only."""
    if not dist_is_on() or world == 1:
        return records
    import torch.distributed as dist

    max_shard = (total_items + world - 1) // world
    gathered = [] if rank == 0 else None
    for start in range(0, max_shard, chunk_size):
        chunk = records[start:start + chunk_size]
        if rank == 0:
            parts = [None] * world
            dist.gather_object(chunk, parts, dst=0)
            for part in parts:
                gathered.extend(part)
        else:
            dist.gather_object(chunk, None, dst=0)
    return gathered


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_metric(value):
    return "-" if value is None else f"{value:.4f}"


def print_comparison(output_dir, current_config):
    rows = []
    for path in sorted(Path(output_dir).glob("xstest_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config", {})
        if any(config.get(key) != current_config.get(key) for key in COMPAT_KEYS):
            continue
        rows.append((data.get("model_name", path.stem), data.get("headline", {})))
    if len(rows) < 2:
        return
    print("\ncompatible model comparison (SafeComp/UnsafeRef/Balanced higher)")
    print(
        f"{'model':28s} {'SafeComp':>10s} {'SafeAnyRef':>10s} "
        f"{'UnsafeRef':>10s} {'Balanced':>10s} {'ParseFail':>10s}"
    )
    for name, headline in rows:
        print(
            f"{name[:28]:28s} "
            f"{format_metric(headline.get('safe_full_compliance_rate')):>10s} "
            f"{format_metric(headline.get('safe_any_refusal_rate')):>10s} "
            f"{format_metric(headline.get('unsafe_full_refusal_rate')):>10s} "
            f"{format_metric(headline.get('balanced_accuracy')):>10s} "
            f"{format_metric(headline.get('judge_parse_fail_rate')):>10s}"
        )


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fully offline XSTest evaluation with local target and judge."
    )
    parser.add_argument("--model_path", required=True,
                        help="local Hugging Face target checkpoint")
    parser.add_argument("--judge_model_path", required=True,
                        help="local Hugging Face three-class judge checkpoint")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--data_dir", default=str(here / "data"))
    parser.add_argument("--output_dir", default="./outputs/xstest")
    parser.add_argument("--dataset", choices=("both", "safe", "unsafe"),
                        default="both")
    parser.add_argument("--limit", type=int, default=0,
                        help="rows per selected safe/unsafe split; 0 means all")
    parser.add_argument("--debug", action="store_true",
                        help="evaluate 4 safe and 4 unsafe prompts")
    parser.add_argument("--strict_counts", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--verify_hash", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--judge_max_new_tokens", type=int, default=64)
    parser.add_argument("--gen_batch", type=int, default=32)
    parser.add_argument("--judge_batch", type=int, default=32)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true",
                        help="sample target responses; default is greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--attn_impl", choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument("--local_files_only",
                        action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    positive = {
        "max_new_tokens": args.max_new_tokens,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "gen_batch": args.gen_batch,
        "judge_batch": args.judge_batch,
        "max_input_tokens": args.max_input_tokens,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit("these arguments must be positive: " + ", ".join(invalid))
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.do_sample and args.temperature <= 0:
        raise SystemExit("--temperature must be positive with --do_sample")

    rank, world, local_rank = setup_distributed()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    is_main = rank == 0
    model_name = clean_name(args.model_name or default_model_name(args.model_path))
    tasks, dataset_sha256 = load_xstest_rows(
        args.data_dir,
        dataset=args.dataset,
        debug=args.debug,
        limit=args.limit,
        strict_counts=args.strict_counts,
        verify_hash=args.verify_hash,
    )
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        counts = {
            split: sum(row["split"] == split for row in tasks)
            for split in ("safe", "unsafe")
        }
        print(f"{world} process(es) | device={device.type} | model={model_name}")
        print(f"local judge: {args.judge_model_path}")
        print(f"tasks: safe={counts['safe']}, unsafe={counts['unsafe']}")
    barrier()
    started = time.time()

    shard_indices = list(range(rank, len(tasks), world))
    local_tasks = [tasks[index] for index in shard_indices]

    target, target_tokenizer = load_chat_model(args.model_path, args, device)
    target_outputs = generate_local(
        target,
        target_tokenizer,
        [task["prompt"] for task in local_tasks],
        device,
        args.max_new_tokens,
        args.gen_batch,
        args.max_input_tokens,
        args.seed + rank,
        do_sample=args.do_sample,
        temperature=args.temperature,
        desc=f"rank{rank} target" if is_main else "",
    )
    del target, target_tokenizer
    empty_cuda_cache()
    barrier()

    local_responses = [split_thinking(output["raw"])[0]
                       for output in target_outputs]
    judge, judge_tokenizer = load_chat_model(args.judge_model_path, args, device)
    judge_outputs = generate_local(
        judge,
        judge_tokenizer,
        [build_judge_prompt(task["prompt"], response)
         for task, response in zip(local_tasks, local_responses)],
        device,
        args.judge_max_new_tokens,
        args.judge_batch,
        args.max_input_tokens,
        args.seed + 9001 + rank,
        desc=f"rank{rank} judge" if is_main else "",
        system_prompt="You are a helpful assistant.",
    )
    del judge, judge_tokenizer
    empty_cuda_cache()
    barrier()

    local_records = []
    for global_index, task, target_output, judge_output in zip(
            shard_indices, local_tasks, target_outputs, judge_outputs):
        response, thinking_trace = split_thinking(target_output["raw"])
        label = parse_judge_label(judge_output["raw"])
        expected_label = (
            LABEL_FULL_REFUSAL if task["should_refuse"]
            else LABEL_FULL_COMPLIANCE
        )
        generation = {
            **task,
            "response": response,
            "thinking_trace": thinking_trace,
            "raw_generation": target_output["raw"],
            "input_tokens": target_output["input_tokens"],
            "output_tokens": target_output["output_tokens"],
            "model": model_name,
        }
        judgement = {
            "item_id": task["item_id"],
            "split": task["split"],
            "prompt_type": task["prompt_type"],
            "focus": task["focus"],
            "note": task["note"],
            "prompt": task["prompt"],
            "response": response,
            "label": label,
            "expected_label": expected_label,
            "strict_correct": label == expected_label,
            "raw_judgement": judge_output["raw"],
            "judge_input_tokens": judge_output["input_tokens"],
            "judge_output_tokens": judge_output["output_tokens"],
        }
        local_records.append((global_index, generation, judgement))

    packed = gather_to_main(local_records, rank, world, len(tasks))
    if is_main:
        ordered = [None] * len(tasks)
        for global_index, generation, judgement in packed:
            ordered[global_index] = (generation, judgement)
        if any(record is None for record in ordered):
            raise RuntimeError("distributed evaluation did not return every prompt")
        generations = [record[0] for record in ordered]
        judgements = [record[1] for record in ordered]
        summaries, headline = summarize_judgements(judgements)
        config = vars(args) | {
            "world_size": world,
            "dataset_sha256": dataset_sha256,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "xstest_commit": XSTEST_COMMIT,
        }
        result = {
            "benchmark": "XSTest (paul-rottger/xstest)",
            "model_name": model_name,
            "model_path": args.model_path,
            "judge_model_path": args.judge_model_path,
            "config": config,
            "headline": headline,
            "summaries": summaries,
        }
        result_path = output_dir / f"xstest_{model_name}.json"
        generations_path = output_dir / f"xstest_generations_{model_name}.jsonl"
        judgements_path = output_dir / f"xstest_judgements_{model_name}.jsonl"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        with open(generations_path, "w", encoding="utf-8") as file:
            for row in generations:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(judgements_path, "w", encoding="utf-8") as file:
            for row in judgements:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("\nheadline")
        print(
            "safe_full_compliance="
            f"{format_metric(headline['safe_full_compliance_rate'])} | "
            "safe_any_refusal="
            f"{format_metric(headline['safe_any_refusal_rate'])} | "
            "unsafe_full_refusal="
            f"{format_metric(headline['unsafe_full_refusal_rate'])} | "
            f"balanced={format_metric(headline['balanced_accuracy'])} | "
            f"parse_fail={format_metric(headline['judge_parse_fail_rate'])}"
        )
        for split in ("safe", "unsafe"):
            summary = summaries.get(split)
            if summary:
                print(
                    f"{split}: n={summary['n']} | "
                    f"compliance={format_metric(summary['full_compliance_rate'])} | "
                    f"full_refusal={format_metric(summary['full_refusal_rate'])} | "
                    f"partial={format_metric(summary['partial_refusal_rate'])}"
                )
        print(f"\nWrote {result_path}")
        print(f"      {generations_path}")
        print(f"      {judgements_path} ({time.time() - started:.0f}s)")
        if (headline.get("judge_parse_fail_rate") or 0) > 0.05:
            print("WARNING: judge parse failures exceed 5%; inspect raw judgements. "
                  "Point rates exclude invalid labels and JSON reports bounds.")
        if any(row["label"] == JUDGE_INVALID for row in judgements):
            print("Unparseable verdicts are never silently converted to refusal.")
        print_comparison(output_dir, config)

    barrier()
    if dist_is_on():
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
