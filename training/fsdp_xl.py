#!/usr/bin/env python3
"""
Shared FSDP plumbing for the XL (32B+) variants of the ReSO / DPO arms
=======================================================================

Four changes versus the 8B-class setup in reso_train.py / dpo_train.py; the
experimental semantics (losses, data, fixtures, selection) are untouched:

  1. Low-host-RAM loading. Rank 0 materializes real weights on CPU; every
     other rank builds the module on the meta device and receives weights via
     FSDP's sync_module_states broadcast during per-layer sharding. Host RAM
     holds ONE model copy total instead of world_size copies (Qwen3-32B fp32:
     ~131GB once, not ~1TB). GPU peak at init is the local shard plus one
     transient unsharded decoder layer.

  2. Sharded frozen reference. The reference (DPO anchoring / ReSO L_pres and
     KL monitors) is FSDP FULL_SHARD in bf16 (~2*P/world bytes per GPU, ~8GB
     at 32B) instead of a per-GPU replica (~66GB at 32B — does not fit next
     to the sharded training state). Its forwards thereby become collectives;
     this is safe because training and validation already run the same code
     with the same shapes on every rank.

  3. --master_dtype bf16 for 70B-class models. fp32 sharded masters + AdamW
     state cost 16 bytes/param (1.15TB at 72B > 8x141GB). bf16 masters cost
     8 bytes/param (moments follow the param dtype) at some optimizer
     precision; fp32 remains the default and the 32B recommendation.

  4. --cpu_offload escape hatch: FSDP CPUOffload keeps params/grads/optimizer
     state in host RAM (fp32 masters at 70B+ become possible, at a large
     step-time cost).

Also: a long process-group timeout (full-state-dict gathers of a 32B+ model
can exceed the NCCL default) and a rank-uniform resolution of --attn_impl
auto (meta-device ranks cannot use a try-on-load fallback, and all ranks must
agree on the kernel).

Recommended: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the launch
environment to curb allocator fragmentation at these sizes.
"""

import functools
import os
from datetime import timedelta

# At 32B+ the peak sits within ~1 decoder layer (~1GB) of the 141GB budget;
# expandable segments reclaim the reserved-but-unallocated fragmentation that
# otherwise tips it over. Set before the first CUDA allocation; an explicit
# user setting wins.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (CPUOffload,
                                    FullyShardedDataParallel as FSDP,
                                    MixedPrecision, ShardingStrategy)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from transformers import AutoConfig, AutoModelForCausalLM

from reso_train import decoder_layer_cls


def mem_report(tag, enabled=True):
    """One-line CUDA memory snapshot. At 141GB budgets, print these at phase
    boundaries so an OOM localizes itself: allocated is live tensors, peak is
    the high-water mark since the last reset_peak_memory_stats()."""
    if not (enabled and torch.cuda.is_available()):
        return
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    p = torch.cuda.max_memory_allocated() / 2**30
    print(f'[mem] {tag}: alloc {a:.1f} GiB | reserved {r:.1f} GiB | peak {p:.1f} GiB')


def setup_distributed_xl(timeout_minutes=180):
    """Same contract as reso_train.setup_distributed, plus a long collective
    timeout: gathering a full 32B+ state dict to rank 0 (and rank-0-only
    safetensors writing while other ranks wait) can exceed the default."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl',
                                timeout=timedelta(minutes=timeout_minutes))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get('LOCAL_RANK',
                                   rank % max(1, torch.cuda.device_count())))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def resolve_attn_impl(attn_impl, dtype):
    """Resolve 'auto' once, identically on every rank (meta-device ranks
    cannot use a try-on-load fallback, and all ranks must agree on the
    kernel). FSDP mixed precision feeds the kernels bf16 at runtime, so
    flash_attention_2 is sound even with fp32 master weights — but some
    transformers versions hard-error on FA2 + fp32 load dtype instead of
    warning. Probe with a tiny meta model: deterministic, so every rank
    resolves to the same implementation."""
    if attn_impl != 'auto':
        return attn_impl
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return 'sdpa'
    try:
        from transformers import LlamaConfig
        cfg = LlamaConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2)
        with torch.device('meta'):
            AutoModelForCausalLM.from_config(
                cfg, torch_dtype=dtype, attn_implementation='flash_attention_2')
        return 'flash_attention_2'
    except Exception:
        return 'sdpa'


def load_model_low_host_mem(model_path, attn_impl, local_files_only, dtype, rank):
    """Rank 0 loads real weights on CPU; other ranks build a meta-device
    skeleton from the config. Must be paired with wrap_fsdp_xl, whose
    sync_module_states broadcast materializes the meta ranks."""
    if rank == 0:
        kwargs = dict(torch_dtype=dtype, attn_implementation=attn_impl,
                      trust_remote_code=True, local_files_only=local_files_only)
        try:
            import accelerate  # noqa: F401  (low_cpu_mem_usage needs it)
            kwargs['low_cpu_mem_usage'] = True
        except ImportError:
            pass
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                     local_files_only=local_files_only)
    with torch.device('meta'):
        return AutoModelForCausalLM.from_config(
            cfg, torch_dtype=dtype, attn_implementation=attn_impl,
            trust_remote_code=True)


def wrap_fsdp_xl(model, rank, local_rank, layer_cls,
                 activation_checkpointing=True, cpu_offload=False):
    """FULL_SHARD with rank-0 weight broadcast (sync_module_states) so it
    composes with load_model_low_host_mem: nonzero ranks materialize each
    unit via to_empty and receive rank 0's values before sharding. Compute
    is bf16, grad reduce fp32, per-decoder-layer non-reentrant activation
    checkpointing — identical numerics to reso_train.wrap_fsdp."""
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={layer_cls})
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32)
    kwargs = dict(auto_wrap_policy=policy, mixed_precision=mp,
                  sharding_strategy=ShardingStrategy.FULL_SHARD,
                  device_id=local_rank, use_orig_params=True,
                  limit_all_gathers=True, sync_module_states=True)
    if rank != 0:
        kwargs['param_init_fn'] = lambda m: m.to_empty(
            device=torch.device('cuda', local_rank), recurse=False)
    if cpu_offload:
        kwargs['cpu_offload'] = CPUOffload(offload_params=True)
    model = FSDP(model, **kwargs)
    if activation_checkpointing:
        wrapper = functools.partial(checkpoint_wrapper,
                                    checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper,
                                       check_fn=lambda m: isinstance(m, layer_cls))
    return model


def load_sharded_reference(model_path, attn_impl, local_files_only, rank,
                           local_rank):
    """Frozen bf16 reference as FSDP FULL_SHARD (~2*P/world bytes per GPU).
    Reference forwards are collectives from here on: every rank must reach
    every reference forward together — already guaranteed, since training
    steps and validation run the same code on all ranks. Takes the RAW
    --attn_impl value and resolves it for bf16, so the reference can keep
    flash_attention_2 even when a fp32 policy had to fall back to sdpa."""
    impl = resolve_attn_impl(attn_impl, torch.bfloat16)
    ref = load_model_low_host_mem(model_path, impl, local_files_only,
                                  torch.bfloat16, rank)
    ref.config.use_cache = False
    ref.requires_grad_(False)
    layer_cls = decoder_layer_cls(ref)
    ref = wrap_fsdp_xl(ref, rank, local_rank, layer_cls,
                       activation_checkpointing=False)
    ref.eval()
    return ref
