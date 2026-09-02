#!/usr/bin/env python3
"""Unified launcher for the nine AdaptSafety evaluation benchmarks."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path("evaluation")
DEFAULT_OUTPUT = EVAL_ROOT / "outputs"


def model_name(path):
    parts = [part for part in Path(path.rstrip("/")).parts if part not in ("/", ".")]
    if parts and parts[-1] == "model":
        parts.pop()
    name = parts[-1] if parts else "model"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def checkpoint_model_type(path):
    """Read a local checkpoint type without importing ML libraries."""
    candidate = Path(path).expanduser()
    config = candidate / "config.json"
    if not config.is_file():
        snapshots = candidate / "snapshots"
        if snapshots.is_dir():
            choices = sorted(p / "config.json" for p in snapshots.iterdir())
            choices = [p for p in choices if p.is_file()]
            if len(choices) == 1:
                config = choices[0]
    if not config.is_file():
        return None
    try:
        return json.loads(config.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return None


def add_target(parser):
    parser.add_argument("--model", required=True, help="local target checkpoint")
    parser.add_argument("--model-name", default=None, help="stable output label")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print commands only")


def parser_for_cli():
    parser = argparse.ArgumentParser(
        description="Run one of the nine evaluation benchmarks with project settings."
    )
    sub = parser.add_subparsers(dest="benchmark", required=True)

    mmlu = sub.add_parser("mmlu-pro")
    add_target(mmlu)
    mmlu.add_argument("--data-path", default=str(EVAL_ROOT / "MMLU-Pro/dataset"))
    mmlu.add_argument("--tp", type=int, default=8)
    mmlu.add_argument("--max-model-len", type=int, default=20000)
    mmlu.add_argument("--max-new-tokens", type=int, default=1024)

    halu = sub.add_parser("halueval")
    add_target(halu)
    halu.add_argument("--task", default="all")
    halu.add_argument("--tp", type=int, default=0, help="0 uses all visible GPUs")
    halu.add_argument("--max-model-len", type=int, default=20000)
    halu.add_argument("--max-new-tokens", type=int, default=4096)

    flames = sub.add_parser("flames")
    add_target(flames)
    flames.add_argument("--data-path", required=True, help="Flames_1k_Chinese JSONL")
    flames.add_argument("--scorer-model", required=True, help="local Flames scorer")
    flames.add_argument("--tp", type=int, default=4)
    flames.add_argument("--max-model-len", type=int, default=20000)
    flames.add_argument("--max-new-tokens", type=int, default=1024)
    flames.add_argument("--temperature", type=float, default=0.7)

    ethics = sub.add_parser("ethics")
    add_target(ethics)

    more = sub.add_parser("morebench")
    add_target(more)
    more.add_argument("--judge-model", required=True)

    xstest = sub.add_parser("xstest")
    add_target(xstest)
    xstest.add_argument("--judge-model", required=True)

    deception = sub.add_parser("deception")
    add_target(deception)
    deception.add_argument("--judge-model", required=True)

    harm = sub.add_parser("harmbench")
    add_target(harm)
    harm.add_argument("--classifier-model", required=True)
    harm.add_argument("--tp", type=int, default=0)

    openrt = sub.add_parser("openrt")
    add_target(openrt)
    openrt.add_argument("--attacker-model", required=True)
    openrt.add_argument("--judge-model", required=True)
    openrt.add_argument("--embedding-model", required=True)
    openrt.add_argument(
        "--prompt-bank",
        default=None,
        help="shared fixed Mousetrap JSONL; generated deterministically if absent",
    )
    return parser


def add_model_name(command, args, flag="--model_name"):
    if args.model_name:
        command.extend([flag, args.model_name])


def commands_for(args):
    py = sys.executable
    output = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT / args.benchmark
    name = args.model_name or model_name(args.model)
    commands = []

    if args.benchmark == "mmlu-pro":
        command = [
            py, str(EVAL_ROOT / "MMLU-Pro/formomi.py"),
            "--model", args.model, "--data_path", args.data_path,
            "--save_dir", str(output), "--selected_subjects", "all",
            "--tp", str(args.tp), "--max_model_len", str(args.max_model_len),
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "halueval":
        tp = args.tp or 0
        command = [
            py, str(EVAL_ROOT / "HaluEval/evaluation/eval.py"),
            "--task", args.task, "--model", args.model,
            "--data_dir", str(EVAL_ROOT / "HaluEval/data"),
            "--instruction_dir", str(EVAL_ROOT / "HaluEval/evaluation"),
            "--save_dir", str(output), "--max_model_len", str(args.max_model_len),
            "--max_new_tokens", str(args.max_new_tokens), "--temperature", "0",
            "--use_chat_template",
        ]
        if tp:
            command.extend(["--tensor_parallel_size", str(tp)])
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "flames":
        responses = output / f"{name}.jsonl"
        commands.extend([
            [
                py, str(EVAL_ROOT / "flames/basemodel.py"),
                "--input_file", args.data_path, "--output_file", str(responses),
                "--model_path", args.model, "--tensor_parallel_size", str(args.tp),
                "--max_model_len", str(args.max_model_len),
                "--max_new_tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
            ],
            [
                py, str(EVAL_ROOT / "flames/infer.py"),
                "--model_path", args.scorer_model, "--data_path", str(responses),
            ],
        ])
    elif args.benchmark == "ethics":
        command = [
            py, str(EVAL_ROOT / "LLM_Ethics_Benchmark/ethics_benchmark_eval.py"),
            "--model_path", args.model,
            "--benchmark_dir", str(EVAL_ROOT / "LLM_Ethics_Benchmark/data/instruments"),
            "--output_dir", str(output),
        ]
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "morebench":
        command = [
            py, str(EVAL_ROOT / "morebench/morebench_eval.py"),
            "--model_path", args.model, "--judge_model_path", args.judge_model,
            "--data_dir", str(EVAL_ROOT / "morebench/data"),
            "--output_dir", str(output), "--dataset", "main",
        ]
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "xstest":
        command = [
            py, str(EVAL_ROOT / "xstest/xstest_eval.py"),
            "--model_path", args.model, "--judge_model_path", args.judge_model,
            "--data_dir", str(EVAL_ROOT / "xstest/data"),
            "--output_dir", str(output), "--dataset", "both",
        ]
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "deception":
        command = [
            py, str(EVAL_ROOT / "deception_bench/deception_bench_eval.py"),
            "--model_path", args.model, "--judge_model_path", args.judge_model,
            "--data_dir", str(EVAL_ROOT / "deception_bench/data"),
            "--output_dir", str(output),
        ]
        add_model_name(command, args)
        commands.append(command)
    elif args.benchmark == "harmbench":
        command = [
            "bash", str(EVAL_ROOT / "HarmBench/eval.sh"),
            args.model, args.classifier_model, str(output), name,
        ]
        if args.tp:
            command = ["env", f"TP={args.tp}", *command]
        commands.append(command)
    elif args.benchmark == "openrt":
        bank = Path(args.prompt_bank) if args.prompt_bank else output / "mousetrap_seed42.jsonl"
        gcg_name = f"{name}_gcg"
        blackbox_name = f"{name}_blackbox"
        mouse_name = f"{name}_mousetrap"
        if not bank.exists():
            commands.append([
                py, str(EVAL_ROOT / "openrt_text/mousetrap_fixed/generate_prompt_bank.py"),
                "--output", str(bank), "--profile", "standard", "--seed", "42",
            ])
        gcg_command = [
                "bash", str(EVAL_ROOT / "openrt_text/run_openrt32.sh"),
                "--model_path", args.model,
                "--attacker_model_path", args.attacker_model,
                "--judge_model_path", args.judge_model,
                "--embedding_model_path", args.embedding_model,
                "--attacks", "gcg", "--profile", "standard",
                "--output_dir", str(output), "--run_name", gcg_name,
        ]
        if checkpoint_model_type(args.model) == "gpt_oss":
            gcg_command.extend(["--gpt_oss_gcg", "--dequantize_mxfp4_for_gcg"])
        commands.extend([
            gcg_command,
            [
                "bash", str(EVAL_ROOT / "openrt_text/run_openrt32.sh"),
                "--model_path", args.model,
                "--attacker_model_path", args.attacker_model,
                "--judge_model_path", args.judge_model,
                "--embedding_model_path", args.embedding_model,
                "--attacks", "blackbox", "--profile", "standard",
                "--output_dir", str(output), "--run_name", blackbox_name,
            ],
            [
                "bash", str(EVAL_ROOT / "openrt_text/mousetrap_fixed/run_mousetrap.sh"),
                "--model_path", args.model, "--judge_model_path", args.judge_model,
                "--prompt_bank", str(bank), "--output_dir", str(output),
                "--run_name", mouse_name,
            ],
            [
                py, str(EVAL_ROOT / "openrt_text/summarize_paper.py"),
                "--core-summary", str(output / gcg_name / "summary.json"),
                str(output / blackbox_name / "summary.json"),
                "--mousetrap-summary", str(output / mouse_name / "summary.json"),
                "--output", str(output / f"{name}_paper_summary.json"),
            ],
        ])
    return commands


def main():
    args = parser_for_cli().parse_args()
    commands = commands_for(args)
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
