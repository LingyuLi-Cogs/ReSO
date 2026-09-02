import os

# Force offline before importing HF / vLLM related libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def process_jsonl_vllm(input_file, output_file, model_path, tensor_parallel_size,
                       max_model_len, max_new_tokens, temperature):
    """
    使用 vLLM 进行 8 卡并行推理
    """

    model_path = Path(model_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if not model_path.is_dir():
        raise NotADirectoryError(f"模型路径不是目录: {model_path}")

    print(f"正在加载分词器: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"正在读取输入文件: {input_file}")
    original_data = []
    prompts = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            original_data.append(data)

            messages = [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": data.get("prompt", "")},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

    print("正在加载模型并初始化 vLLM (8卡并行)...")

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="auto",
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"开始推理，总计 {len(prompts)} 条数据...")
    outputs = llm.generate(prompts, sampling_params)

    print(f"推理完成，正在保存至: {output_file}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            result_data = original_data[i]
            result_data["response"] = generated_text

            f_out.write(json.dumps(result_data, ensure_ascii=False) + "\n")

    print("全部处理完成！")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 vLLM 对 JSONL 文件进行批量推理"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 JSONL 文件路径",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出 JSONL 文件路径",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径或 HuggingFace 模型名称",
    )

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=20000)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_jsonl_vllm(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
