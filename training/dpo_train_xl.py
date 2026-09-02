#!/usr/bin/env python3
"""
DPO XL: behavioral alignment arm for 32B+ models (8x H200)
==========================================================

Same experiment as dpo_train.py — identical pair construction, DPO loss,
reference anchoring to the *initial* weights, passive RSA and 3-way judgment
monitors, fixtures, selection and checkpoint format — re-plumbed for
Qwen3-32B-class and 70B-class models. See fsdp_xl.py for the shared changes
(low-host-RAM rank-0 loading + broadcast, sharded frozen reference,
--master_dtype, --cpu_offload). Runs are not resumable: training always starts
from --model_path at step 0, which is also the frozen DPO reference.

One DPO-specific addition: gradient accumulation. At 8B the 64-pair step runs
as one forward of 128 sequences; at 32B the two full-vocabulary logit tensors
plus the fp32 log-softmax kept in the autograd graph (~2 bytes and ~4 bytes
per position x 151k vocab respectively) make that a >40GB activation bill.
--micro_pairs splits the step into micro-batches whose losses are backwarded
with weights len(chunk)/len(pairs), so the optimizer step is exactly the mean
over the same pairs_per_step pairs — the effective batch, the stratified
sampler, and the loss are unchanged versus the 8B arm. FSDP reduces grads per
micro-backward into the sharded fp32 accumulators (no no_sync, no unsharded
grad residency). Every rank draws the same pair count, so micro chunking
stays collective-safe.

Memory budget, Qwen3-32B on 8x H200 (141GB), fp32 masters, micro_pairs=16:

  sharded fp32 params + grads + AdamW moments      ~66 GB/GPU
  sharded bf16 reference                            ~8 GB/GPU
  activations + policy/ref logits + fp32 logprob
  chunks for 32 sequences (~192 tok)               ~20-30 GB peak
  ------------------------------------------------ ~100-105 GB: fits

Host RAM: rank 0 holds one full copy at load (~131GB fp32 / ~66GB bf16 at
32B) and one full bf16 copy (~66GB) per best/final checkpoint gather.

70B-class: --master_dtype bf16 (AdamW moments then run in bf16 — watch the
val curves) or --cpu_offload with fp32 masters (slow, needs ~1.2TB host RAM);
also drop --micro_pairs to 8 and --eval_batch_items to 128. Full-parameter
DPO LR transfer is not size-free: stay at the low end (5e-7) or below.

Launch (8x H200):

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=8 dpo_train_xl.py \
      --model_path /path/to/Qwen3-32B \
      --output_dir ./outputs/dpo32b_full

Requires: torch >= 2.1, transformers >= 4.40, flash-attn (recommended),
accelerate, pandas, numpy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dpo_train import (batch_pairs_from_bank, build_val_pairs, collate_pairs,
                       completion_logprobs, dpo_loss_and_stats, run_validation)
from fsdp_xl import (load_model_low_host_mem, load_sharded_reference,
                     mem_report, resolve_attn_impl, setup_distributed_xl,
                     wrap_fsdp_xl)
from reso_train import (MoralItemBank, all_reduce_, barrier, broadcast_,
                        build_judgment_fixture, build_rsa_fixture,
                        decoder_layer_cls, dist_is_on, encode_actions,
                        freeze_io_embeddings, load_replay_val_batches,
                        pooled_forward, sample_batch, save_full_checkpoint)


# ============================================================================
# CLI (superset of dpo_train.py; XL additions at the bottom)
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='DPO XL: behavioral alignment arm '
                                            'for 32B+ models (full-parameter).')
    here = Path(__file__).resolve().parent
    data_dir = here.parent / 'dataset'
    # paths
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--train_csv', type=str,
                   default=str(data_dir / 'social_chem_train_expanded.csv'))
    p.add_argument('--train_buckets', type=str, default=str(data_dir / 'train_buckets.json'))
    p.add_argument('--val_csv', type=str,
                   default=str(data_dir / 'social_chem_val_expanded.csv'))
    p.add_argument('--val_buckets', type=str, default=str(data_dir / 'val_buckets.json'))
    p.add_argument('--replay_data', type=str, default=None,
                   help='optional JSONL for the replay-perplexity monitor (no gradient)')
    p.add_argument('--output_dir', type=str, default='./outputs/dpo_xl')
    # DPO loss / pairs
    p.add_argument('--beta_dpo', type=float, default=0.1, help='DPO temperature on log-ratios')
    p.add_argument('--neutral_rejected_frac', type=float, default=0.25,
                   help='fraction of pole-item pairs whose rejected is a false neutral')
    p.add_argument('--chosen_nll_coef', type=float, default=0.0,
                   help='optional NLL regularizer on chosen completions (off = canonical DPO)')
    p.add_argument('--pairs_per_step', type=int, default=64,
                   help='preference pairs per rank per optimizer step (stratified '
                        'item draw; the effective batch, matched to the 8B arm)')
    p.add_argument('--max_prompt_tokens', type=int, default=160)
    p.add_argument('--max_resp_tokens', type=int, default=32)
    # optimization
    p.add_argument('--num_steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=5e-7,
                   help='full-parameter DPO peak LR (canonical range 5e-7..1e-6; '
                        'stay at or below the low end for 32B+)')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_embeddings', action='store_true',
                   help='also train input embeddings and unembedding (frozen by default)')
    # eval / logging / selection
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--save_interval', type=int, default=0,
                   help='periodic full checkpoints every N steps; 0 disables '
                        '(each is a ~2 bytes/param model dir — ~66GB at 32B; '
                        'best/ and final/ are always written)')
    p.add_argument('--val_pairs', type=int, default=1024)
    p.add_argument('--rsa_per_domain', type=int, default=150)
    p.add_argument('--judge_per_domain', type=int, default=100,
                   help='held-out items per domain for the 3-way judgment '
                        'monitor (0 disables it); keep matched to the ReSO arm '
                        'so both arms score the same fixture')
    p.add_argument('--judge_batch_seqs', type=int, default=48,
                   help='candidate sequences per forward in the judgment monitor '
                        '(halved vs the 8B script; halve again for 70B-class)')
    p.add_argument('--eval_batch_items', type=int, default=256,
                   help='items per no-grad embedding forward in the passive RSA '
                        'monitor (halved vs the 8B script; halve again for 70B)')
    p.add_argument('--max_action_tokens', type=int, default=64,
                   help='for the passive RSA monitor readout')
    p.add_argument('--ema_momentum', type=float, default=0.99,
                   help='EMA momentum for the RSA-monitor centering mean '
                        '(mirrors the ReSO arm); updated once per validation '
                        'call since this arm has no training-time '
                        'representation pass')
    p.add_argument('--ema_init_batches', type=int, default=8,
                   help='train-bank batches used to initialize the EMA mean '
                        'before training starts')
    p.add_argument('--replay_val_docs', type=int, default=64)
    p.add_argument('--replay_seqs', type=int, default=2)
    p.add_argument('--replay_len', type=int, default=1024)
    p.add_argument('--patience', type=int, default=10,
                   help='evals without val preference-accuracy gain before early stop')
    p.add_argument('--min_delta', type=float, default=2e-3)
    # system
    p.add_argument('--attn_impl', choices=['auto', 'flash_attention_2', 'sdpa', 'eager'],
                   default='auto')
    p.add_argument('--no_activation_checkpointing', action='store_true')
    p.add_argument('--local_files_only', action='store_true')
    # XL additions
    p.add_argument('--micro_pairs', type=int, default=16,
                   help='pairs per forward/backward; grad-accumulated up to '
                        'pairs_per_step (0 = single pass, the 8B behavior). '
                        'Use 8 for 70B-class')
    p.add_argument('--master_dtype', choices=['fp32', 'bf16'], default='fp32',
                   help='sharded master-weight dtype. fp32 (default) fits up to '
                        '~32B on 8x141GB; bf16 halves param+grad memory and runs '
                        'AdamW moments in bf16 (needed for 70B-class)')
    p.add_argument('--cpu_offload', action='store_true',
                   help='FSDP CPUOffload: params/grads/optimizer state in host '
                        'RAM — fits fp32 masters at 70B+, several times slower')
    return p.parse_args()


# ============================================================================
# Main (mirrors dpo_train.main; model plumbing + micro-batching differ)
# ============================================================================

@record  # surfaces per-rank tracebacks in the torchrun error summary
def main():
    args = parse_args()
    rank, world, local = setup_distributed_xl()
    if not dist_is_on():
        raise SystemExit('FSDP training must be launched with torchrun, e.g.\n'
                         '  torchrun --standalone --nproc_per_node=8 dpo_train_xl.py ...')
    device = torch.device(f'cuda:{local}')
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng([args.seed, rank])

    master_dtype = torch.float32 if args.master_dtype == 'fp32' else torch.bfloat16
    attn_impl = resolve_attn_impl(args.attn_impl, master_dtype)
    micro = args.micro_pairs if args.micro_pairs > 0 else args.pairs_per_step

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'args.json', 'w') as f:
            json.dump(vars(args) | {'world_size': world,
                                    'attn_impl_resolved': attn_impl}, f, indent=2)
        metrics_f = open(out_dir / 'metrics.jsonl', 'a')

    def log(event):
        if is_main:
            metrics_f.write(json.dumps(event) + '\n')
            metrics_f.flush()

    # ------------------------------------------------------------ model
    if is_main:
        print(f'Loading {args.model_path} on {world} GPU(s) '
              f'({args.master_dtype} masters, attn={attn_impl}, '
              f'rank-0 load + FSDP broadcast)...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # rank 0: real weights on CPU; other ranks: meta skeleton (see fsdp_xl)
    model = load_model_low_host_mem(args.model_path, attn_impl,
                                    args.local_files_only, master_dtype, rank)
    model.config.use_cache = False
    args.n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    if not args.train_embeddings:
        freeze_io_embeddings(model)
    n_tr = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)
    if is_main:
        print(f'trainable params: {n_tr / 1e9:.2f}B (full-parameter, matched to '
              f'the ReSO arm; embeddings '
              f'{"trained" if args.train_embeddings else "frozen"})')

    layer_cls = decoder_layer_cls(model)
    model = wrap_fsdp_xl(model, rank, local, layer_cls,
                         activation_checkpointing=not args.no_activation_checkpointing,
                         cpu_offload=args.cpu_offload)

    barrier()
    mem_report('policy wrapped (expect ~4P/world GiB fp32 shards)', is_main)

    # DPO reference: frozen bf16 replica of the *initial* weights, FSDP-sharded
    # (a per-GPU replica does not fit at 32B+)
    ref_model = load_sharded_reference(args.model_path, args.attn_impl,
                                       args.local_files_only, rank, local)
    barrier()
    mem_report('reference wrapped (expect +~2P/world GiB)', is_main)
    torch.cuda.reset_peak_memory_stats()

    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.num_steps * args.warmup_ratio), args.num_steps)

    # ------------------------------------------------------------ data
    train_bank = MoralItemBank(args.train_csv, args.train_buckets)

    # Validation runs on all ranks, so fixtures are built identically everywhere.
    val_bank = MoralItemBank(args.val_csv, args.val_buckets)
    fx_rng = np.random.default_rng([args.seed, 4242])
    val_pairs = build_val_pairs(val_bank, args.val_pairs, fx_rng,
                                args.neutral_rejected_frac)
    # identical fixture seed to the ReSO arm -> same RSA items across arms
    rsa_fx = build_rsa_fixture(val_bank, args.rsa_per_domain,
                               np.random.default_rng([args.seed, 9999]))
    # likewise for the judgment fixture: same seed derivation, same items
    judge_fx = (build_judgment_fixture(val_bank, args.judge_per_domain,
                                       np.random.default_rng([args.seed, 5150]))
                if args.judge_per_domain > 0 else None)
    replay_val = None
    if args.replay_data:
        replay_val = load_replay_val_batches(args.replay_data, tokenizer,
                                             args.replay_val_docs,
                                             args.replay_seqs, args.replay_len)
    if is_main:
        print(f'val fixtures: {len(val_pairs)} pairs, {len(rsa_fx["rows"])} RSA '
              f'items, {len(judge_fx) if judge_fx else 0} judgment items')

    # dpo_train.run_validation chunks val pairs by .pairs_per_step; at XL the
    # per-forward budget is micro_pairs, so hand it an adjusted view of args.
    val_args = argparse.Namespace(**{**vars(args), 'pairs_per_step': micro})

    mu = torch.zeros(args.n_layers, d_model, device=device)

    def init_ema_mean():
        model.eval()
        with torch.no_grad():
            acc_sum = torch.zeros_like(mu)
            acc_cnt = torch.zeros((), device=device)
            for _ in range(args.ema_init_batches):
                rows, _ = sample_batch(train_bank, args.eval_batch_items, data_rng)
                enc = encode_actions(tokenizer, [train_bank.texts[r] for r in rows],
                                     args.max_action_tokens, device)
                P = pooled_forward(model, enc, args.n_layers)
                acc_sum += P.sum(1)
                acc_cnt += P.shape[1]
            all_reduce_(acc_sum)
            all_reduce_(acc_cnt)
            mu.copy_(acc_sum / acc_cnt)
        if is_main:
            print(f'EMA mean initialized from {int(acc_cnt)} items')
        model.train()

    best_acc = -float('inf')
    init_ema_mean()

    def trainer_state(step):
        return dict(step=step, best_metric=best_acc, mu=mu.detach().cpu())

    ref_ppl_cache = {}
    evals_since_best = 0
    flags = torch.zeros(2, device=device)  # [stop, save_best]

    # Baseline before any optimizer step: the reference point every later eval is
    # read against. Excluded from best-checkpoint selection and early stopping.
    # Cheapest eval of the run — AdamW allocates its moments lazily on the first
    # .step(), so the sharded optimizer state does not exist yet.
    vm0 = run_validation(model, ref_model, tokenizer, val_pairs, val_bank,
                         rsa_fx, replay_val, val_args, device, ref_ppl_cache, mu,
                         judge_fx)
    mem_report('eval step 0', is_main)
    if is_main:
        vm0.update({'event': 'val', 'step': 0, 'val/is_best': False})
        log(vm0)
        print(f"  eval 0 (baseline): pref_acc {vm0['val/pref_acc']:.4f} "
              f"| dpo_loss {vm0['val/dpo_loss']:.4f} "
              f"| judge {vm0.get('val/judge_acc', float('nan')):.4f} "
              f"| RSA(passive) {vm0['val/rsa']:.4f} "
              f"| cross {vm0['val/rsa_cross_domain']:.4f}"
              + (f" | dppl {vm0['val/replay_ppl_delta_pct']:+.2f}%"
                 if 'val/replay_ppl_delta_pct' in vm0 else ''))

    model.train()
    t_last = time.time()

    # ------------------------------------------------------------ training loop
    for step in range(1, args.num_steps + 1):
        pairs = batch_pairs_from_bank(train_bank, args.pairs_per_step, data_rng,
                                      args.neutral_rejected_frac)
        # micro-batched forward/backwards; losses weighted by chunk fraction so
        # the accumulated grad equals the full-batch mean over `pairs`. The
        # chunk count is identical on every rank (the stratified draw size
        # depends only on pairs_per_step), keeping FSDP collectives aligned.
        loss_step, stats = 0.0, {}
        for s in range(0, len(pairs), micro):
            chunk = pairs[s:s + micro]
            enc = collate_pairs(tokenizer, chunk, device,
                                args.max_prompt_tokens, args.max_resp_tokens)
            pol_logits = model(input_ids=enc['input_ids'],
                               attention_mask=enc['attention_mask'],
                               use_cache=False).logits
            with torch.no_grad():
                ref_logits = ref_model(input_ids=enc['input_ids'],
                                       attention_mask=enc['attention_mask'],
                                       use_cache=False).logits

            pol_lp = completion_logprobs(pol_logits, enc['input_ids'],
                                         enc['completion_mask'])
            with torch.no_grad():
                ref_lp = completion_logprobs(ref_logits, enc['input_ids'],
                                             enc['completion_mask'])

            loss, cstats = dpo_loss_and_stats(pol_lp, ref_lp, args.beta_dpo)
            if args.chosen_nll_coef > 0:
                b = len(chunk)
                n_tok = enc['completion_mask'][:b, 1:].sum(-1).clamp(min=1).float()
                loss = loss + args.chosen_nll_coef * (-pol_lp[:b] / n_tok).mean()

            w = len(chunk) / len(pairs)
            (loss * w).backward()
            loss_step += float(loss) * w
            for k, v in cstats.items():
                stats[k] = stats.get(k, 0.0) + v * w

        del enc, pol_logits, ref_logits, pol_lp, ref_lp, loss
        grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        # free the sharded fp32 grads immediately (instead of at the top of
        # the next step) so they don't sit through validation / saving
        optimizer.zero_grad(set_to_none=True)
        mem_report(f'step {step} after optimizer step', is_main and step == 1)

        # ------------------------------------------------------ logging
        if is_main and (step % args.log_interval == 0 or step == 1):
            poles = np.array([p['pole'] for p in pairs])
            ev = {'event': 'train', 'step': step,
                  'loss': round(loss_step, 5),
                  'lr': scheduler.get_last_lr()[0],
                  'grad_norm': round(float(grad_norm), 4),
                  'pairs': len(pairs),
                  'micro_pairs': micro,
                  'n_virtue': int((poles == 1).sum()),
                  'n_vice': int((poles == -1).sum()),
                  'n_neutral': int((poles == 0).sum()),
                  'sec_per_step': round((time.time() - t_last) / args.log_interval, 3)}
            ev.update({k: round(v, 5) for k, v in stats.items()})
            log(ev)
            print(f"step {step:5d} | loss {ev['loss']:.4f} | acc {ev['acc']:.3f} "
                  f"| margin {ev['reward_margin']:.3f} | {ev['sec_per_step']:.2f}s/step")
            t_last = time.time()

        # ------------------------------------------------------ validation
        if step % args.eval_interval == 0 or step == args.num_steps:
            flags.zero_()
            vm = run_validation(model, ref_model, tokenizer, val_pairs, val_bank,
                                rsa_fx, replay_val, val_args, device,
                                ref_ppl_cache, mu, judge_fx)
            mem_report(f'eval step {step}', is_main)
            if is_main:
                vm.update({'event': 'val', 'step': step})
                # selection stays on val pref_acc: judgment accuracy is logged
                # as a monitor and never enters the decision.
                improved = vm['val/pref_acc'] > best_acc + args.min_delta
                if improved:
                    best_acc = vm['val/pref_acc']
                    evals_since_best = 0
                    flags[1] = 1.0
                else:
                    evals_since_best += 1
                vm['val/best_pref_acc'] = best_acc
                vm['val/is_best'] = improved
                log(vm)
                print(f"  eval {step}: pref_acc {vm['val/pref_acc']:.4f} "
                      f"(best {best_acc:.4f}) | dpo_loss {vm['val/dpo_loss']:.4f} "
                      f"| judge {vm.get('val/judge_acc', float('nan')):.4f} "
                      f"| RSA(passive) {vm['val/rsa']:.4f} "
                      f"| cross {vm['val/rsa_cross_domain']:.4f}"
                      + (f" | dppl {vm['val/replay_ppl_delta_pct']:+.2f}%"
                         if 'val/replay_ppl_delta_pct' in vm else ''))
                if evals_since_best >= args.patience:
                    print(f'Early stop: no val pref-acc gain in {args.patience} evals')
                    flags[0] = 1.0
            broadcast_(flags)
            if flags[1] > 0:  # collective save on all ranks
                save_full_checkpoint(out_dir / 'best', model, tokenizer,
                                     trainer_state(step), is_main)
            if flags[0] > 0:
                break

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_full_checkpoint(out_dir / f'step_{step:06d}', model, tokenizer,
                                 trainer_state(step), is_main)

    # ------------------------------------------------------------ final save
    save_full_checkpoint(out_dir / 'final', model, tokenizer,
                         trainer_state(step), is_main)
    if is_main:
        log({'event': 'done', 'step': step, 'best_pref_acc': best_acc})
        metrics_f.close()
        print(f'Done. Best val preference accuracy {best_acc:.4f}. Outputs in {out_dir}')
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
