#!/usr/bin/env python3
"""
Offline MMLU-Pro CoT evaluation for one local model / checkpoint
================================================================

This script evaluates a locally-present model on a locally-present MMLU-Pro
dataset using chain-of-thought few-shot prompts.

Offline behavior:
- HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE are forced on.
- Model/tokenizer must be local.
- Dataset must be local, preferably prepared by datasets.save_to_disk().
- Initial CoT prompt file must be local.

To prepare MMLU-Pro on a networked machine:

    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    ds.save_to_disk("data/MMLU-Pro")

Then copy that directory to the offline GPU box and run this script with:

    --data_path data/MMLU-Pro

Requires:
- vllm
- transformers
- datasets
- torch
- tqdm

Optional:
- local LoRA adapter path via --lora_path
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

# Force offline before importing HF / vLLM related libraries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


CHOICES = list("ABCDEFGHIJKLMNOP")
DEFAULT_MAX_MODEL_LENGTH = 20000
DEFAULT_MAX_NEW_TOKENS = 1024
RANDOM_SEED = 12345

random.seed(RANDOM_SEED)

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = HERE / "dataset"
DEFAULT_INITIAL_PROMPT = HERE / "cot_prompt_lib/initial_prompt.txt"


# ============================================================================
# Utility
# ============================================================================

def sanitize(name: str) -> str:
    """Make a string safe for file / directory names."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def default_model_name(path: str) -> str:
    """Derive a compact model name from a local model path."""
    parts = [q for q in Path(str(path).rstrip("/")).parts if q not in ("/", ".")]
    if parts and parts[-1] == "model":
        parts = parts[:-1]
    if len(parts) >= 2 and (
        parts[-1] in ("best", "final") or parts[-1].startswith("step_")
    ):
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else "model"


def resolve_tp(args) -> int:
    """vLLM tensor parallel size. 0 means all visible GPUs."""
    if args.tp and args.tp > 0:
        return args.tp
    return max(1, torch.cuda.device_count())


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = sanitize(input_args.model_name or default_model_name(input_args.model))
    subjects = sanitize(input_args.selected_subjects.replace(",", "-").replace(" ", "_"))
    return [model_name, scoring_method, subjects]


# ============================================================================
# Dataset loading / preprocessing
# ============================================================================

def preprocess(dataset_split):
    """Drop N/A options and convert HF dataset split into a mutable list."""
    res_df = []
    for row in dataset_split:
        each = dict(row)
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def load_mmlu_pro(data_path: str):
    """Load MMLU-Pro from a local datasets.save_to_disk directory."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset path not found: {data_path}\n\n"
            "Prepare it on a networked machine with:\n"
            '  from datasets import load_dataset\n'
            '  ds = load_dataset("TIGER-Lab/MMLU-Pro")\n'
            '  ds.save_to_disk("data/MMLU-Pro")\n\n'
            "Then copy that folder to this machine and pass:\n"
            "  --data_path data/MMLU-Pro"
        )

    dataset = load_from_disk(str(data_path))

    if "test" not in dataset or "validation" not in dataset:
        raise SystemExit(
            f"Local dataset at {data_path} must contain 'test' and 'validation' splits."
        )

    test_df = preprocess(dataset["test"])
    val_df = preprocess(dataset["validation"])
    return test_df, val_df


def select_by_category(df, subject):
    return [each for each in df if each["category"] == subject]


# ============================================================================
# Prompt construction
# ============================================================================

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]

    prompt += question + "\n"
    prompt += "Options:\n"

    for i, opt in enumerate(options):
        prompt += f"{CHOICES[i]}. {opt}\n"

    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."

    return prompt


def generate_cot_prompt(val_df, curr, k, initial_prompt_path):
    initial_prompt_path = Path(initial_prompt_path)
    if not initial_prompt_path.exists():
        raise SystemExit(f"Initial CoT prompt file not found: {initial_prompt_path}")

    prompt = initial_prompt_path.read_text(encoding="utf-8")

    subject = curr["category"]
    subject_val_df = select_by_category(val_df, subject)
    subject_val_df = subject_val_df[:k]

    prompt = prompt.replace("{$}", subject) + "\n"

    for example in subject_val_df:
        prompt += format_cot_example(example, including_answer=True)

    prompt += format_cot_example(curr, including_answer=False)

    return prompt


# ============================================================================
# Model loading
# ============================================================================

def load_model(args):
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model path not found: {model_path}\n"
            "This offline script expects --model to be a local model directory."
        )

    if args.lora_path and not Path(args.lora_path).exists():
        raise SystemExit(f"LoRA path not found: {args.lora_path}")

    tp = resolve_tp(args)

    logging.info(f"Loading local model: {model_path}")
    logging.info(f"tensor_parallel_size={tp}, offline=True")

    llm = LLM(
        model=str(model_path),
        gpu_memory_utilization=float(args.gpu_util),
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        enable_lora=True if args.lora_path else False,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["Question:"],
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(
            lora_name="lora",
            lora_path=str(Path(args.lora_path)),
            lora_int_id=1,
        )

    return (llm, sampling_params, lora_request), tokenizer


# ============================================================================
# Answer extraction
# ============================================================================

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    logging.debug("1st answer extract failed:\n%s", text)
    return extract_again(text)


def extract_again(text):
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1).upper()

    return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).upper()

    return None


# ============================================================================
# Inference / evaluation
# ============================================================================

def batch_inference(llm, sampling_params, lora_request, inference_batch):
    start = time.time()

    outputs = llm.generate(
        inference_batch,
        sampling_params,
        lora_request=lora_request,
    )

    logging.info(
        "%d-size batch cost time: %.2f sec",
        len(inference_batch),
        time.time() - start,
    )

    response_batch = []
    pred_batch = []

    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred_batch.append(extract_answer(generated_text))

    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(res, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1

    if corr + wrong == 0:
        return 0.0, 0.0, 0.0

    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, args):
    llm, sampling_params, lora_request = model

    logging.info("evaluating %s", subject)

    inference_batches = []

    for i in tqdm(range(len(test_df)), desc=f"building prompts: {subject}"):
        k = args.ntrain
        curr = test_df[i]

        prompt_length_ok = False
        prompt = None

        while not prompt_length_ok:
            if k < 0:
                raise RuntimeError(
                    f"Prompt is still too long even with k=0 for subject={subject}, "
                    f"question index={i}. Try lowering --max_new_tokens or raising "
                    f"--max_model_len if the model supports it."
                )

            prompt = generate_cot_prompt(
                val_df=val_df,
                curr=curr,
                k=k,
                initial_prompt_path=args.initial_prompt,
            )

            # Tokenize on CPU only. No need to move prompt tokens to GPU.
            inputs = tokenizer(prompt, return_tensors="pt")
            length = len(inputs["input_ids"][0])

            if length < args.max_model_len - args.max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1

        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(
        llm,
        sampling_params,
        lora_request,
        inference_batches,
    )

    res = []
    for j, curr in enumerate(test_df):
        item = dict(curr)
        item["pred"] = pred_batch[j]
        item["model_outputs"] = response_batch[j]
        res.append(item)

    accu, corr, wrong = save_res(res, output_path)

    logging.info(
        "subject=%s accu=%.4f corr=%s wrong=%s",
        subject,
        accu,
        corr,
        wrong,
    )

    return accu, corr, wrong


# ============================================================================
# Main
# ============================================================================

def main(args):
    model, tokenizer = load_model(args)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_test_df, full_val_df = load_mmlu_pro(args.data_path)

    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])

    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")

        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)

    selected_subjects = sorted(set(selected_subjects))

    if not selected_subjects:
        raise SystemExit(
            f"No subjects selected from --selected_subjects={args.selected_subjects!r}."
        )

    logging.info("selected subjects:\n%s", "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))

    sta_dict = {}

    with open(args.summary_path, "a", encoding="utf-8") as f:
        f.write("\n------category level sta------\n")

    for subject in selected_subjects:
        sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}

        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)

        output_path = Path(args.save_result_dir) / f"{sanitize(subject)}.json"

        acc, corr_count, wrong_count = eval_cot(
            subject=subject,
            model=model,
            tokenizer=tokenizer,
            val_df=val_df,
            test_df=test_df,
            output_path=output_path,
            args=args,
        )

        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc

        with open(args.summary_path, "a", encoding="utf-8") as f:
            f.write(f"Average accuracy {sta_dict[subject]['accu']:.4f} - {subject}\n")

    total_corr, total_wrong = 0.0, 0.0

    for _, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]

    total_accu = total_corr / (total_corr + total_wrong + 1e-6)

    sta_dict["total"] = {
        "corr": total_corr,
        "wrong": total_wrong,
        "accu": total_accu,
    }

    with open(args.summary_path, "a", encoding="utf-8") as f:
        f.write("\n------average acc sta------\n")
        f.write(f"Average accuracy: {total_accu:.4f}\n")

    # Save structured summary too.
    summary_json_path = Path(args.summary_path).with_suffix(".json")
    summary_json_path.write_text(
        json.dumps(sta_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Append global record.
    global_record_file = Path(args.global_record_file)
    global_record_file.parent.mkdir(parents=True, exist_ok=True)

    with open(global_record_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [args.time_str, total_accu]
        writer.writerow(record)

    logging.info("wrote summary: %s", args.summary_path)
    logging.info("wrote summary json: %s", summary_json_path)
    logging.info("wrote global record: %s", global_record_file)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline MMLU-Pro CoT evaluation for one local model."
    )

    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")

    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--global_record_file",
        "-grf",
        type=str,
        default=None,
        help="Optional aggregate CSV. Defaults to <save_dir>/runs.csv.",
    )

    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Local target model directory. No HF repo download is attempted.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional label for output files. Default is derived from --model.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Local MMLU-Pro dataset directory saved by datasets.save_to_disk().",
    )

    parser.add_argument(
        "--initial_prompt",
        type=str,
        default=str(DEFAULT_INITIAL_PROMPT),
        help="Local path to cot_prompt_lib/initial_prompt.txt.",
    )

    parser.add_argument("--lora_path", "-lp", type=str, default=None)

    parser.add_argument(
        "--tp",
        type=int,
        default=0,
        help="vLLM tensor parallel size. 0 = all visible GPUs.",
    )

    parser.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Allow local custom model code. Default: True.",
    )

    parser.add_argument(
        "--no_trust_remote_code",
        dest="trust_remote_code",
        action="store_false",
    )

    args = parser.parse_args()

    # Validate local paths early.
    if not Path(args.model).exists():
        raise SystemExit(f"Model path not found: {args.model}")

    if not Path(args.data_path).exists():
        raise SystemExit(f"Dataset path not found: {args.data_path}")

    if not Path(args.initial_prompt).exists():
        raise SystemExit(f"Initial prompt file not found: {args.initial_prompt}")

    if args.lora_path and not Path(args.lora_path).exists():
        raise SystemExit(f"LoRA path not found: {args.lora_path}")

    # Build output paths.
    os.makedirs(args.save_dir, exist_ok=True)
    if args.global_record_file is None:
        args.global_record_file = str(Path(args.save_dir) / "runs.csv")

    path_parts = args_generate_path(args)
    args.save_result_dir = os.path.join(args.save_dir, *path_parts)

    file_prefix = "-".join(path_parts)
    timestamp = time.time()
    args.time_str = time.strftime("%m-%d_%H-%M", time.localtime(timestamp))

    file_name = f"{file_prefix}_{args.time_str}_summary.txt"

    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    args.summary_path = os.path.join(summary_dir, file_name)

    os.makedirs(args.save_result_dir, exist_ok=True)

    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)

    log_file = os.path.join(
        save_log_dir,
        file_name.replace("_summary.txt", "_logfile.log"),
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("HF_HUB_OFFLINE=%s", os.environ.get("HF_HUB_OFFLINE"))
    logging.info("TRANSFORMERS_OFFLINE=%s", os.environ.get("TRANSFORMERS_OFFLINE"))
    logging.info("HF_DATASETS_OFFLINE=%s", os.environ.get("HF_DATASETS_OFFLINE"))
    logging.info("model=%s", args.model)
    logging.info("data_path=%s", args.data_path)
    logging.info("initial_prompt=%s", args.initial_prompt)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
