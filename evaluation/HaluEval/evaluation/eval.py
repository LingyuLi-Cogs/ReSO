import os

# ============================================================
# 必须放在 transformers / vllm 导入之前
# 强制关闭 HuggingFace / transformers 联网
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# 关闭 vLLM telemetry / usage stats
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import logging
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ============================================================
# 任务配置
#
# HaluEval 判别任务：
# 给定 context + answer/response/summary，模型输出 Yes/No。
#
# Yes = 含幻觉
# No  = 不含幻觉
# ============================================================

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "qa": {
        "data_file": "qa_data.json",
        "instruction_file": "qa/qa_evaluation_instruction.txt",
        "context_label": "#Question#",
        "answer_label": "#Answer#",
        "context_key": "question",
        "right_key": "right_answer",
        "hallu_key": "hallucinated_answer",
        "extra_keys": ["knowledge"],
    },
    "dialogue": {
        "data_file": "dialogue_data.json",
        "instruction_file": "dialogue/dialogue_evaluation_instruction.txt",
        "context_label": "#Dialogue History#",
        "answer_label": "#Response#",
        "context_key": "dialogue_history",
        "right_key": "right_response",
        "hallu_key": "hallucinated_response",
        "extra_keys": ["knowledge"],
    },
    "summarization": {
        "data_file": "summarization_data.json",
        "instruction_file": "summarization/summarization_evaluation_instruction.txt",
        "context_label": "#Document#",
        "answer_label": "#Summary#",
        "context_key": "document",
        "right_key": "right_summary",
        "hallu_key": "hallucinated_summary",
        "extra_keys": [],
    },
}


# ============================================================
# 基础工具
# ============================================================

def require_local_dir(path: str, name: str) -> str:
    """强制要求本地目录，避免误传 HuggingFace repo id 导致联网。"""
    if not path:
        raise ValueError(f"{name} 必须指定本地目录。")

    local_path = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.isdir(local_path):
        raise FileNotFoundError(
            f"{name} 必须是本地目录，不能是 HuggingFace repo id 或 URL。\n"
            f"当前传入：{path}\n"
            f"解析路径：{local_path}"
        )

    return local_path


def require_local_file(path: str, name: str) -> str:
    """强制要求本地文件。"""
    if not path:
        raise ValueError(f"{name} 必须指定本地文件。")

    local_path = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"{name} 不是本地文件：{local_path}")

    return local_path


def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def sanitize_path_component(x: Any) -> str:
    s = str(x).strip()
    s = re.sub(r"[\\/:\*\?\"<>\|\s]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._")
    return s or "unknown"


def model_dir_name(model_path: str) -> str:
    return sanitize_path_component(os.path.basename(os.path.normpath(model_path)))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_jsonl(data: Dict[str, Any], output_path: str, append: bool = True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_text(path: str, text: str, append: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(text)


class RunningStats:
    """
    与官方代码保持接近：
    - correct / wrong：能抽取到 Yes/No 的样本
    - failed：无法抽取 Yes/No，或同时出现 Yes 和 No
    - accuracy_all：correct / 总样本数
    - accuracy_scored：correct / 可解析样本数
    """

    def __init__(self):
        self.correct = 0
        self.wrong = 0
        self.failed = 0

    def update(self, correct: Optional[bool]):
        if correct is None:
            self.failed += 1
        elif correct:
            self.correct += 1
        else:
            self.wrong += 1

    def merge(self, other: "RunningStats"):
        self.correct += other.correct
        self.wrong += other.wrong
        self.failed += other.failed

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.failed

    @property
    def scored(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy_all(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def accuracy_scored(self) -> float:
        if self.scored == 0:
            return 0.0
        return self.correct / self.scored

    def format(self) -> str:
        return (
            f"correct={self.correct}, "
            f"wrong={self.wrong}, "
            f"failed={self.failed}, "
            f"total={self.total}, "
            f"accuracy_all={self.accuracy_all:.4f}, "
            f"accuracy_scored={self.accuracy_scored:.4f}"
        )


# ============================================================
# Yes / No 抽取
# ============================================================

def extract_yes_no_official_style(text: str) -> Optional[str]:
    """
    尽量贴近官方代码的判断方式：

    官方逻辑：
    - 同时包含 Yes 和 No：failed
    - 两者都不包含：failed
    - 只包含 Yes：Yes
    - 只包含 No：No

    这里做了轻微增强：
    - 去掉句号
    - 大小写不敏感
    - 使用单词边界，避免 yesterday 里的 yes 误判为 Yes
    """
    if not text:
        return None

    text = text.replace(".", "")
    yes = re.search(r"\byes\b", text, flags=re.IGNORECASE) is not None
    no = re.search(r"\bno\b", text, flags=re.IGNORECASE) is not None

    if yes and no:
        return None
    if not yes and not no:
        return None
    if yes:
        return "Yes"
    if no:
        return "No"

    return None


# ============================================================
# instruction / prompt
# ============================================================

def read_instruction(instruction_dir: str, task: str) -> str:
    cfg = TASK_CONFIG[task]
    path = require_local_file(
        os.path.join(instruction_dir, cfg["instruction_file"]),
        f"{task} instruction",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_user_prompt(task: str, instruction: str, context: str, answer_text: str) -> str:
    cfg = TASK_CONFIG[task]

    return (
        f"{instruction}\n\n"
        f"{cfg['context_label']}: {context}\n"
        f"{cfg['answer_label']}: {answer_text}\n"
        f"#Your Judgement#:"
    )


def render_model_prompt(user_prompt: str, tokenizer, use_chat_template: bool) -> str:
    """
    本地模型输入格式。

    - 默认不用 system prompt；
    - 如果模型是 chat model，可以加 --use_chat_template。
    """
    if not use_chat_template:
        return user_prompt

    messages = [{"role": "user", "content": user_prompt}]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        raise RuntimeError(
            "使用 chat_template 失败。请确认本地 tokenizer_config.json 中包含 "
            "chat_template，或关闭 --use_chat_template。"
        ) from e


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_prompt(
    task: str,
    instruction: str,
    context: str,
    answer_text: str,
    tokenizer,
    args,
) -> Tuple[str, str, int, bool]:
    """
    构造 prompt，并在超长时截断 context。

    保留：
    - instruction
    - answer_text / summary / response
    - #Your Judgement#:

    只截断 context/document/dialogue/question。
    """
    max_prompt_tokens = args.max_model_len - args.max_new_tokens

    if max_prompt_tokens <= 0:
        raise ValueError(
            f"max_model_len={args.max_model_len} 必须大于 "
            f"max_new_tokens={args.max_new_tokens}"
        )

    ctx = context
    truncated = False

    while True:
        user_prompt = build_user_prompt(task, instruction, ctx, answer_text)
        model_prompt = render_model_prompt(
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
        )
        tok_len = count_tokens(tokenizer, model_prompt)

        if tok_len <= max_prompt_tokens:
            return user_prompt, model_prompt, tok_len, truncated

        words = ctx.split()

        if len(words) <= 16:
            raise ValueError(
                "prompt 过长，且无法通过截断 context 解决。\n"
                f"task={task}\n"
                f"token_length={tok_len}\n"
                f"max_prompt_tokens={max_prompt_tokens}\n"
            )

        drop = max(16, len(words) // 10)
        ctx = " ".join(words[:-drop])
        truncated = True


# ============================================================
# 数据构造
# ============================================================

def build_samples(
    task: str,
    raw_items: List[Dict[str, Any]],
    rng: random.Random,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """
    保持官方评测逻辑：
    每条数据随机选择 right 或 hallucinated 一个作为待评估答案。
    """
    cfg = TASK_CONFIG[task]

    if max_samples is not None and max_samples > 0:
        raw_items = raw_items[:max_samples]

    samples = []

    for idx, item in enumerate(raw_items):
        context = str(item.get(cfg["context_key"], "")).strip()
        right = str(item.get(cfg["right_key"], "")).strip()
        hallu = str(item.get(cfg["hallu_key"], "")).strip()

        if rng.random() > 0.5:
            answer_text = hallu
            ground_truth = "Yes"
            is_hallucinated = True
        else:
            answer_text = right
            ground_truth = "No"
            is_hallucinated = False

        sample = {
            "idx": idx,
            "context": context,
            "answer_text": answer_text,
            "ground_truth": ground_truth,
            "is_hallucinated": is_hallucinated,
            "raw": item,
        }

        for k in cfg.get("extra_keys", []):
            if k in item:
                sample[k] = item[k]

        samples.append(sample)

    return samples


# ============================================================
# 本地模型加载与生成
# ============================================================

def load_vllm_model(model_path: str, args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    tensor_parallel_size = args.tensor_parallel_size

    if tensor_parallel_size is None or tensor_parallel_size <= 0:
        num_gpus = torch.cuda.device_count()
        if num_gpus <= 0:
            raise RuntimeError("vLLM 需要至少一张 CUDA GPU。")
        tensor_parallel_size = num_gpus

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        gpu_memory_utilization=float(args.gpu_util),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    return llm, tokenizer, sampling_params


def cleanup_llm(llm):
    if llm is None:
        return

    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception:
        pass

    try:
        del llm
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass


def batch_generate(
    llm,
    sampling_params,
    prompts: List[str],
    batch_size: int,
    desc: str,
) -> List[str]:
    if not prompts:
        return []

    if batch_size <= 0:
        batch_size = len(prompts)

    responses = []
    start_time = time.time()

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch = prompts[start:start + batch_size]
        outputs = llm.generate(batch, sampling_params)

        for output in outputs:
            responses.append(output.outputs[0].text.strip())

    elapsed = time.time() - start_time

    logging.info(
        f"{desc}: generated {len(prompts)} prompts, "
        f"batch_size={batch_size}, time={elapsed:.2f}s"
    )

    return responses


# ============================================================
# 单轮评测
# ============================================================

def evaluate_task(
    task: str,
    llm,
    tokenizer,
    sampling_params,
    args,
    rng: random.Random,
    run_dir: str,
) -> RunningStats:
    cfg = TASK_CONFIG[task]

    instruction = read_instruction(args.instruction_dir, task)

    data_path = require_local_file(
        os.path.join(args.data_dir, cfg["data_file"]),
        f"{task} data",
    )

    raw_items = load_jsonl(data_path)
    samples = build_samples(task, raw_items, rng, args.max_samples)

    logging.info(f"[{task}] loaded samples: {len(samples)}")

    prompts = []
    metas = []
    trunc_count = 0

    for sample in tqdm(samples, desc=f"build prompts: {task}"):
        user_prompt, model_prompt, tok_len, truncated = fit_prompt(
            task=task,
            instruction=instruction,
            context=sample["context"],
            answer_text=sample["answer_text"],
            tokenizer=tokenizer,
            args=args,
        )

        if truncated:
            trunc_count += 1

        prompts.append(model_prompt)

        metas.append(
            {
                "sample": sample,
                "user_prompt": user_prompt,
                "prompt_token_length": tok_len,
                "prompt_truncated": truncated,
            }
        )

    responses = batch_generate(
        llm=llm,
        sampling_params=sampling_params,
        prompts=prompts,
        batch_size=args.batch_size,
        desc=f"generation: {task}",
    )

    if len(responses) != len(metas):
        raise RuntimeError(
            f"{task} responses 数量不一致：{len(responses)} vs {len(metas)}"
        )

    output_path = os.path.join(run_dir, f"{task}_results.jsonl")

    if os.path.exists(output_path):
        os.remove(output_path)

    stats = RunningStats()

    for meta, output in zip(metas, responses):
        sample = meta["sample"]
        gold = sample["ground_truth"]

        pred = extract_yes_no_official_style(output)

        if pred is None:
            correct = None
            judgement = "failed!"
        else:
            correct = pred == gold
            judgement = pred

        stats.update(correct)

        out_item = {
            "task": task,
            "idx": sample["idx"],
            "ground_truth": gold,
            "judgement": judgement,
            "correct": correct,
            "is_hallucinated": sample["is_hallucinated"],
            "model_output": output,
            "prompt": meta["user_prompt"],
            "prompt_token_length": meta["prompt_token_length"],
            "prompt_truncated": meta["prompt_truncated"],
            "raw": sample["raw"],
        }

        if task == "qa":
            out_item.update(
                {
                    "knowledge": sample.get("knowledge"),
                    "question": sample["context"],
                    "answer": sample["answer_text"],
                }
            )
        elif task == "dialogue":
            out_item.update(
                {
                    "knowledge": sample.get("knowledge"),
                    "dialogue_history": sample["context"],
                    "response": sample["answer_text"],
                }
            )
        elif task == "summarization":
            out_item.update(
                {
                    "document": sample["context"],
                    "summary": sample["answer_text"],
                }
            )

        dump_jsonl(out_item, output_path, append=True)

    logging.info(
        f"[{task}] {stats.format()}, "
        f"truncated={trunc_count}/{len(samples)}, "
        f"output_path={output_path}"
    )

    return stats


def run_evaluation(args):
    model_path = require_local_dir(args.model, "--model")
    data_dir = require_local_dir(args.data_dir, "--data_dir")
    instruction_dir = require_local_dir(args.instruction_dir, "--instruction_dir")

    args.model = model_path
    args.data_dir = data_dir
    args.instruction_dir = instruction_dir

    tasks = resolve_tasks(args.task)

    time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())

    model_name = (
        sanitize_path_component(args.model_name)
        if args.model_name
        else model_dir_name(model_path)
    )

    run_name = args.run_name or f"answer_{model_name}_{time_str}"
    run_dir = os.path.join(args.save_dir, run_name)

    os.makedirs(run_dir, exist_ok=True)

    setup_logging(os.path.join(run_dir, "run.log"))

    summary_path = os.path.join(run_dir, "summary.txt")

    write_text(
        summary_path,
        (
            "HaluEval local answer-mode summary\n"
            f"mode: answer only\n"
            f"model: {model_path}\n"
            f"model_name: {model_name}\n"
            f"data_dir: {data_dir}\n"
            f"instruction_dir: {instruction_dir}\n"
            f"tasks: {tasks}\n"
            f"save_dir: {args.save_dir}\n"
            f"run_dir: {run_dir}\n"
            f"seed: {args.seed}\n"
            f"max_samples: {args.max_samples}\n"
            f"max_model_len: {args.max_model_len}\n"
            f"max_new_tokens: {args.max_new_tokens}\n"
            f"temperature: {args.temperature}\n"
            f"top_p: {args.top_p}\n"
            f"use_chat_template: {args.use_chat_template}\n"
            f"batch_size: {args.batch_size}\n"
            f"gpu_util: {args.gpu_util}\n"
            f"tensor_parallel_size: {args.tensor_parallel_size}\n\n"
        ),
        append=False,
    )

    logging.info(f"run_dir={run_dir}")
    logging.info(f"model={model_path}")
    logging.info(f"model_name={model_name}")
    logging.info(f"tasks={tasks}")

    rng = random.Random(args.seed)

    llm = None

    try:
        logging.info("Loading local model...")
        llm, tokenizer, sampling_params = load_vllm_model(model_path, args)
        logging.info("Model loaded.")

        total_stats = RunningStats()

        for task in tasks:
            task_stats = evaluate_task(
                task=task,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                args=args,
                rng=rng,
                run_dir=run_dir,
            )

            total_stats.merge(task_stats)

            write_text(
                summary_path,
                (
                    f"[{task}]\n"
                    f"{task_stats.format()}\n"
                    f"output_path: {os.path.join(run_dir, f'{task}_results.jsonl')}\n\n"
                ),
                append=True,
            )

        write_text(
            summary_path,
            (
                "------ overall ------\n"
                f"{total_stats.format()}\n"
            ),
            append=True,
        )

        logging.info("------ overall ------")
        logging.info(total_stats.format())

        print(f"\nResults saved to: {run_dir}")

    finally:
        cleanup_llm(llm)


# ============================================================
# 参数解析
# ============================================================

def resolve_tasks(task_arg: str) -> List[str]:
    task_arg = task_arg.strip().lower()

    if task_arg == "all":
        return ["qa", "dialogue", "summarization"]

    tasks = []

    for t in task_arg.split(","):
        t = t.strip().lower()
        if not t:
            continue

        if t not in TASK_CONFIG:
            raise ValueError(
                f"未知 task: {t}，可选：qa, dialogue, summarization, all"
            )

        if t not in tasks:
            tasks.append(t)

    if not tasks:
        raise ValueError("没有选中任何 task。")

    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(repo_root, "data")
    default_instruction_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="HaluEval 本地离线评测：只做一轮 Yes/No hallucination 判断。"
    )

    parser.add_argument(
        "--task",
        "-t",
        default="qa",
        help="qa, dialogue, summarization, all，或逗号分隔，如 qa,dialogue",
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="本地模型目录，不能是 HuggingFace repo id。",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="自定义用于保存结果目录的模型名。不影响实际加载的 --model 路径。",
    )

    parser.add_argument(
        "--data_dir",
        default=default_data_dir,
        help="HaluEval 数据目录，包含 qa_data.json / dialogue_data.json / summarization_data.json。",
    )

    parser.add_argument(
        "--instruction_dir",
        default=default_instruction_dir,
        help="instruction 根目录，包含 qa/ dialogue/ summarization/ 子目录。",
    )

    parser.add_argument(
        "--save_dir",
        "-s",
        default="halueval_results",
        help="结果输出根目录。",
    )

    parser.add_argument(
        "--run_name",
        default=None,
        help="自定义运行目录名。优先级高于 --model_name。",
    )

    parser.add_argument(
        "--gpu_util",
        "-gu",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization。",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        "-tp",
        type=int,
        default=None,
        help="张量并行 GPU 数。默认使用全部 CUDA GPU。",
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="vLLM max_model_len。",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="最大生成 token 数。Yes/No 任务建议设置较小，如 8、16、32。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="外层 batch size。<=0 表示每个任务一次性送入 vLLM。",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度。",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p。",
    )

    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="是否使用 tokenizer.apply_chat_template。Chat 模型建议开启。",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="每个任务最多评测多少条。<=0 表示全部。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="随机种子，用于随机选择 right / hallucinated。",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    run_evaluation(args)


if __name__ == "__main__":
    main()