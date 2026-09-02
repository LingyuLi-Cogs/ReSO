"""Small, dependency-light helpers shared by the local evaluators.

The evaluation scripts intentionally do not import the training programs.  This
keeps evaluation usable in a clean inference environment and avoids depending
on private or historical modules that are not part of the public repository.
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM


def dist_is_on():
    return dist.is_available() and dist.is_initialized()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(
            os.environ.get(
                "LOCAL_RANK", rank % max(1, torch.cuda.device_count())
            )
        )
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def barrier():
    if dist_is_on():
        dist.barrier()


def gather_all(value, world=None):
    """Gather one Python value from every rank and return the list on all ranks."""
    if not dist_is_on():
        return [value]
    size = dist.get_world_size() if world is None else world
    gathered = [None] * size
    dist.all_gather_object(gathered, value)
    return gathered


def default_model_name(path):
    parts = [part for part in Path(str(path).rstrip("/")).parts if part not in ("/", ".")]
    if parts and parts[-1] == "model":
        parts = parts[:-1]
    if len(parts) >= 2 and (
        parts[-1] in ("best", "final") or parts[-1].startswith("step_")
    ):
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1] if parts else "model"


def load_hf_model(model_path, attn_impl, local_files_only, dtype):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if attn_impl != "auto":
        return AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation=attn_impl, **kwargs
        )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="sdpa", **kwargs
        )


def render_prompt(tokenizer, user_text):
    """Render the deployment chat format, with a plain-QA fallback."""
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": user_text}]
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
    return f"Question: {user_text}\nAnswer: "
