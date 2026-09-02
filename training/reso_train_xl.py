#!/usr/bin/env python3
"""
ReSO XL: representational similarity optimization for 32B+ models (8x H200)
============================================================================

Same experiment as reso_train.py — L = L_struct + beta * L_pres, identical
data pipeline, triplet mining, EMA centering, fixtures, selection and
checkpoint format — re-plumbed for Qwen3-32B-class and 70B-class models.
See fsdp_xl.py for what changed (low-host-RAM rank-0 loading + broadcast,
sharded frozen reference, --master_dtype, --cpu_offload); everything the
paper cares about is bit-identical in intent to the 8B script.

Memory budget, Qwen3-32B (32.8B params, 64 layers, d=5120) on 8x H200
(141GB), fp32 masters (default):

  sharded fp32 params            16.4 GB/GPU
  sharded fp32 grads             16.4 GB/GPU
  sharded AdamW moments (fp32)   32.8 GB/GPU
  sharded bf16 reference          8.2 GB/GPU
  retained graph + transients (batch_items=260, 64 tok; replay 2x1024
  full-vocab KL; bmm structure loss — the gather-based one adds ~11GB)
                                                   ~40-50 GB peak
  ------------------------------------------------ ~115-125 GB: fits,
  with little slack — a 32B run peaked at ~134GB with the gather-based
  loss and OOM'd on a single ~1GB layer all-gather. If it still OOMs on
  your stack: --batch_items 192 first (~-7GB), then --replay_seqs 1.

Host RAM: rank 0 transiently holds one full copy at load (~131GB fp32 /
~66GB bf16 for 32B) and one full bf16 copy (~66GB) at each best/final
checkpoint gather. Other ranks stay near zero.

70B-class (e.g. Qwen2.5-72B): fp32 masters + AdamW do not fit (16 B/param =
1.15TB > 8x141GB). Either
  --master_dtype bf16      8 B/param sharded state (~72GB/GPU at 72B) + 18GB
                           reference; AdamW moments run in bf16 — a real
                           optimizer-precision tradeoff at lr ~1e-5..5e-7,
                           watch grad_norm and val curves; also reduce
                           --batch_items (e.g. 128-192) and
                           --eval_batch_items (e.g. 128), or
  --cpu_offload            keeps fp32 masters in host RAM (needs ~1.2TB host
                           RAM and is several times slower per step).

Launch (8x H200):

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=8 reso_train_xl.py \
      --model_path /path/to/Qwen3-32B \
      --replay_data /path/to/replay.jsonl \
      --output_dir ./outputs/reso32b_beta0.1 --beta 0.1

All public ReSO variants (--shuffle_labels, --learned_layer_weights,
--proto_anchor_cross_pole, --hard_negative_frac, beta sweep)
work unchanged. Checkpoints are plain bf16 HF model dirs (~66GB at 32B),
directly loadable by the Part 1 diagnostics. Validation logs the same passive
3-way judgment monitor as the 8B script, built from the same fixture seed, so
the numbers stay comparable across arms and scales, and the same step-0
baseline before any optimizer step (excluded from best-checkpoint selection and
early stopping). Runs are not resumable: training always starts from
--model_path at step 0. Note that full-parameter LR
transfer is not size-free: expect the useful LR for 32B+ to sit at or below
the low end of the 8B range.

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
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from fsdp_xl import (load_model_low_host_mem, load_sharded_reference,
                     mem_report, resolve_attn_impl, setup_distributed_xl,
                     wrap_fsdp_xl)
from reso_train import (MoralItemBank, ReplayStream, all_reduce_, barrier,
                        broadcast_, build_judgment_fixture, build_rsa_fixture,
                        build_val_triplet_fixture, decoder_layer_cls,
                        dist_is_on, encode_actions, family_stats,
                        freeze_io_embeddings, load_replay_val_batches,
                        mine_triplets, pooled_forward, run_validation,
                        sample_batch, save_full_checkpoint, token_kl)


def structure_loss_bmm(Zn, trip, margin, tau, layer_w, device):
    """Same loss as reso_train.structure_loss (identical dot products,
    Bradley-Terry, layer weighting), restructured for memory: the original
    gathers four per-triplet fp32 copies of shape [L, T, d] into the autograd
    graph (~11GB at B=260, K=8, d=5120, 64 layers); here the full [L, B, B]
    cosine matrix is one batched matmul (~17MB at B=260) whose backward flows
    through Zn directly, and triplet similarities are indexed out of it."""
    ti = torch.from_numpy(trip['i']).to(device)
    tj = torch.from_numpy(trip['j']).to(device)
    tk = torch.from_numpy(trip['k']).to(device)
    S = torch.bmm(Zn, Zn.transpose(1, 2))   # [L_inc, B, B]
    s_ij = S[:, ti, tj]                      # [L_inc, T]
    s_ik = S[:, ti, tk]
    logits = (s_ij - s_ik - margin) / tau
    per_layer = F.softplus(-logits).mean(dim=1)  # -log sigmoid
    loss = (layer_w * per_layer).sum()
    with torch.no_grad():
        acc = (s_ij > s_ik).float().mean(0).cpu().numpy()  # uniform over layers
    return loss, acc


# ============================================================================
# CLI (superset of reso_train.py; XL additions at the bottom)
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='ReSO XL: representational similarity '
                                            'optimization for 32B+ models.')
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
    p.add_argument('--replay_data', type=str, default=str(data_dir / 'replay.jsonl'),
                   help='JSONL replay corpus ({"text"} or {"messages"})')
    p.add_argument('--output_dir', type=str, default='./outputs/reso_xl')
    # loss
    p.add_argument('--beta', type=float, default=0.1, help='the single tradeoff knob')
    p.add_argument('--margin', type=float, default=0.05, help='delta: similarity margin')
    p.add_argument('--tau', type=float, default=0.1, help='Bradley-Terry temperature')
    p.add_argument('--delta_h', type=float, default=0.2, help='min human gap for a triplet')
    p.add_argument('--cap_k', type=int, default=8, help='max triplets per anchor')
    p.add_argument('--learned_layer_weights', action='store_true',
                   help='ablation: softmax layer weights instead of uniform 1/L')
    p.add_argument('--exclude_layers', type=str, default='',
                   help='comma-separated decoder layers excluded from L_struct, e.g. "0,63"')
    p.add_argument('--proto_anchor_cross_pole', action='store_true',
                   help='conservative ablation: cross-pole triplets need prototype anchors')
    p.add_argument('--proto_threshold', type=float, default=0.75)
    p.add_argument('--hard_negative_frac', type=float, default=0.0,
                   help='efficiency variant: fraction of K mined by smallest model margin')
    # control arm
    p.add_argument('--shuffle_labels', action='store_true',
                   help='shuffled control: permute (pole, typicality, h) within each domain')
    p.add_argument('--shuffle_seed', type=int, default=0)
    # batches
    p.add_argument('--batch_items', type=int, default=260,
                   help='N: stratified action texts per rank per step (>=256 for '
                        'triplet density; fits at 32B — drop to 128-192 for 70B-class)')
    p.add_argument('--max_action_tokens', type=int, default=64)
    p.add_argument('--replay_seqs', type=int, default=2, help='M: replay sequences per rank per step')
    p.add_argument('--replay_len', type=int, default=1024)
    p.add_argument('--ema_momentum', type=float, default=0.99)
    p.add_argument('--ema_init_batches', type=int, default=8)
    # optimization
    p.add_argument('--num_steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=1e-5,
                   help='full-parameter AdamW peak LR (larger models usually want '
                        'the low end of the 8B sweep or below)')
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
    p.add_argument('--eval_batch_items', type=int, default=256,
                   help='items per no-grad embedding forward at eval (halved vs '
                        'the 8B script; halve again for 70B-class)')
    p.add_argument('--rsa_per_domain', type=int, default=150)
    p.add_argument('--val_triplet_batches', type=int, default=8)
    p.add_argument('--judge_per_domain', type=int, default=100,
                   help='held-out items per domain for the passive 3-way '
                        'judgment monitor (0 disables it); keep matched to the '
                        'DPO arm so both arms score the same fixture')
    p.add_argument('--judge_batch_seqs', type=int, default=48,
                   help='candidate sequences per forward in the judgment monitor '
                        '(halved vs the 8B script; halve again for 70B-class)')
    p.add_argument('--max_prompt_tokens', type=int, default=160,
                   help='judgment-monitor prompt truncation (matched to the DPO arm)')
    p.add_argument('--max_resp_tokens', type=int, default=32,
                   help='judgment-monitor completion truncation (matched to the DPO arm)')
    p.add_argument('--replay_val_docs', type=int, default=64)
    p.add_argument('--patience', type=int, default=10,
                   help='evals without val RSA improvement before early stop')
    p.add_argument('--min_delta', type=float, default=1e-3)
    # system
    p.add_argument('--attn_impl', choices=['auto', 'flash_attention_2', 'sdpa', 'eager'],
                   default='auto')
    p.add_argument('--no_activation_checkpointing', action='store_true')
    p.add_argument('--local_files_only', action='store_true')
    # XL additions
    p.add_argument('--master_dtype', choices=['fp32', 'bf16'], default='fp32',
                   help='sharded master-weight dtype. fp32 (default) fits up to '
                        '~32B on 8x141GB; bf16 halves param+grad memory and runs '
                        'AdamW moments in bf16 (needed for 70B-class)')
    p.add_argument('--cpu_offload', action='store_true',
                   help='FSDP CPUOffload: params/grads/optimizer state in host '
                        'RAM — fits fp32 masters at 70B+, several times slower')
    return p.parse_args()


# ============================================================================
# Main (mirrors reso_train.main; only the model/reference plumbing differs)
# ============================================================================

@record  # surfaces per-rank tracebacks in the torchrun error summary
def main():
    args = parse_args()
    rank, world, local = setup_distributed_xl()
    if not dist_is_on():
        raise SystemExit('FSDP training must be launched with torchrun, e.g.\n'
                         '  torchrun --standalone --nproc_per_node=8 reso_train_xl.py ...')
    device = torch.device(f'cuda:{local}')
    is_main = rank == 0

    if args.beta > 0 and args.replay_data is None:
        raise ValueError('--replay_data is required when beta > 0')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng([args.seed, rank])

    master_dtype = torch.float32 if args.master_dtype == 'fp32' else torch.bfloat16
    attn_impl = resolve_attn_impl(args.attn_impl, master_dtype)

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
        print(f'{args.n_layers} layers, d={d_model}; trainable params: {n_tr / 1e9:.2f}B '
              f'(full-parameter, embeddings '
              f'{"trained" if args.train_embeddings else "frozen"})')

    layer_cls = decoder_layer_cls(model)
    model = wrap_fsdp_xl(model, rank, local, layer_cls,
                         activation_checkpointing=not args.no_activation_checkpointing,
                         cpu_offload=args.cpu_offload)
    barrier()
    mem_report('policy wrapped (expect ~4P/world GiB fp32 shards)', is_main)

    # frozen reference: FSDP-sharded bf16 (per-GPU replicas don't fit at 32B+)
    ref_model = None
    if args.beta > 0 or args.replay_data:
        ref_model = load_sharded_reference(args.model_path, args.attn_impl,
                                           args.local_files_only, rank, local)
        barrier()
        mem_report('reference wrapped (expect +~2P/world GiB)', is_main)
    torch.cuda.reset_peak_memory_stats()

    excl = {int(x) for x in args.exclude_layers.split(',') if x.strip() != ''}
    inc_layers = [l for l in range(args.n_layers) if l not in excl]
    if not inc_layers:
        raise ValueError('all layers excluded from L_struct')
    inc_t = torch.tensor(inc_layers, dtype=torch.long, device=device)

    layer_logits = None
    if args.learned_layer_weights:
        layer_logits = torch.zeros(len(inc_layers), device=device, requires_grad=True)

    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    opt_groups = [{'params': trainable}]
    if layer_logits is not None:
        opt_groups.append({'params': [layer_logits]})
    optimizer = torch.optim.AdamW(opt_groups, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.num_steps * args.warmup_ratio), args.num_steps)

    # ------------------------------------------------------------ data
    train_bank = MoralItemBank(args.train_csv, args.train_buckets,
                               shuffle_labels=args.shuffle_labels,
                               shuffle_seed=args.shuffle_seed)
    mine_kwargs = dict(delta_h=args.delta_h, cap_k=args.cap_k,
                       proto_anchor_cross=args.proto_anchor_cross_pole,
                       proto_threshold=args.proto_threshold)

    # Validation runs on all ranks (FSDP forwards are collective), so fixtures
    # are built identically everywhere. Held-out fixtures never use shuffled
    # labels: the control arm is scored against the real human structure.
    val_bank = MoralItemBank(args.val_csv, args.val_buckets)
    fx_rng = np.random.default_rng([args.seed, 9999])
    rsa_fx = build_rsa_fixture(val_bank, args.rsa_per_domain, fx_rng)
    trip_fx = build_val_triplet_fixture(val_bank, args.val_triplet_batches,
                                        args.batch_items, mine_kwargs, fx_rng)
    # Dedicated stream (not fx_rng): the judgment fixture must not shift the RSA
    # or triplet draws. Same seed derivation as the 8B script and the DPO arm.
    judge_fx = (build_judgment_fixture(val_bank, args.judge_per_domain,
                                       np.random.default_rng([args.seed, 5150]))
                if args.judge_per_domain > 0 else None)
    replay_val = None
    if args.replay_data:
        replay_val = load_replay_val_batches(args.replay_data, tokenizer,
                                             args.replay_val_docs,
                                             args.replay_seqs, args.replay_len)
    if is_main:
        print(f'val fixtures: {len(rsa_fx["rows"])} RSA items, '
              f'{sum(len(b["trip"]["i"]) for b in trip_fx)} val triplets, '
              f'{len(judge_fx) if judge_fx else 0} judgment items')

    replay = None
    if args.beta > 0:
        replay = ReplayStream(args.replay_data, tokenizer, args.replay_len,
                              rank, world, skip_first=args.replay_val_docs)

    # ------------------------------------------------------------ EMA centering
    # mu_l: stop-gradient EMA of the per-layer mean (anisotropy correction),
    # initialized from a pre-pass, then updated online with all-reduced stats.
    mu = torch.zeros(args.n_layers, d_model, device=device)
    best_val_rsa = -float('inf')
    model.eval()
    with torch.no_grad():
        acc_sum = torch.zeros_like(mu)
        acc_cnt = torch.zeros((), device=device)
        for _ in range(args.ema_init_batches):
            rows, _ = sample_batch(train_bank, args.batch_items, data_rng)
            enc = encode_actions(tokenizer, [train_bank.texts[r] for r in rows],
                                 args.max_action_tokens, device)
            P = pooled_forward(model, enc, args.n_layers)
            acc_sum += P.sum(1)
            acc_cnt += P.shape[1]
        all_reduce_(acc_sum)
        all_reduce_(acc_cnt)
        mu.copy_(acc_sum / acc_cnt)
    if is_main:
        print(f'EMA means initialized from {int(acc_cnt)} items')
    mem_report('after EMA init pre-pass', is_main)
    model.train()

    def trainer_state(step):
        return dict(step=step, best_val_rsa=best_val_rsa, mu=mu.detach().cpu(),
                    layer_logits=(layer_logits.detach().cpu()
                                  if layer_logits is not None else None))

    ref_ppl_cache = {}
    evals_since_best = 0
    flags = torch.zeros(2, device=device)  # [stop, save_best]
    uniform_w = torch.full((len(inc_layers),), 1.0 / len(inc_layers), device=device)

    # Baseline before any optimizer step: the reference point every later eval is
    # read against. Excluded from best-checkpoint selection and early stopping.
    # Cheapest eval of the run — AdamW allocates its moments lazily on the first
    # .step(), so the sharded optimizer state does not exist yet.
    vm0 = run_validation(model, ref_model, tokenizer, val_bank, rsa_fx,
                         trip_fx, replay_val, mu, inc_t, inc_layers,
                         args, device, ref_ppl_cache, judge_fx)
    mem_report('eval step 0', is_main)
    if is_main:
        vm0.update({'event': 'val', 'step': 0, 'val/is_best': False})
        log(vm0)
        print(f"  eval 0 (baseline): RSA {vm0['val/rsa']:.4f} "
              f"| cross {vm0['val/rsa_cross_domain']:.4f} "
              f"| trip_acc {vm0.get('val/triplet_acc', float('nan')):.4f} "
              f"| judge {vm0.get('val/judge_acc', float('nan')):.4f} "
              f"| KL {vm0.get('val/kl_ref_policy', float('nan')):.5f} "
              f"| dppl {vm0.get('val/replay_ppl_delta_pct', float('nan')):+.2f}%")

    t_last = time.time()

    # ------------------------------------------------------------ training loop
    for step in range(1, args.num_steps + 1):
        first = step == 1
        # (a) representation pass. L_struct is backwarded immediately after
        # (before the preservation forward): the two graphs never coexist and
        # FSDP completes a canonical forward->backward gather/reshard cycle
        # per loss term instead of holding two live graphs over every layer.
        # Grads accumulate across the two backwards, so the update equals
        # backward(L_struct + beta * L_pres) exactly.
        rows, segments = sample_batch(train_bank, args.batch_items, data_rng)
        enc = encode_actions(tokenizer, [train_bank.texts[r] for r in rows],
                             args.max_action_tokens, device)
        P = pooled_forward(model, enc, args.n_layers)  # [L_dec, B, d] fp32, graph

        with torch.no_grad():
            bsum = P.detach().sum(1)
            bcnt = torch.tensor(float(P.shape[1]), device=device)
            all_reduce_(bsum)
            all_reduce_(bcnt)
            mu.mul_(args.ema_momentum).add_((bsum / bcnt) * (1.0 - args.ema_momentum))

        Zn = F.normalize(P - mu.unsqueeze(1), dim=-1)[inc_t]  # [L_inc, B, d]

        # (b) in-batch triplet mining
        seg_sims = None
        if args.hard_negative_frac > 0.0:
            mid = Zn[len(inc_layers) // 2]
            full = (mid @ mid.T).detach().cpu().numpy()
            seg_sims = [full[s['offset']:s['offset'] + len(s['rows']),
                             s['offset']:s['offset'] + len(s['rows'])] for s in segments]
        trip, n_pre_cap = mine_triplets(segments, rng=data_rng,
                                        hard_frac=args.hard_negative_frac,
                                        seg_sims=seg_sims, **mine_kwargs)

        layer_w = (F.softmax(layer_logits, dim=0) if layer_logits is not None
                   else uniform_w)
        if trip is not None:
            loss_struct, trip_acc = structure_loss_bmm(Zn, trip, args.margin,
                                                       args.tau, layer_w, device)
        else:
            # keep the graph so every rank backwards through the model (the
            # reduce-scatters are collectives and must fire on all ranks)
            loss_struct, trip_acc = P.sum() * 0.0, None
        loss_struct.backward()
        loss_struct_f = float(loss_struct)
        del P, Zn, enc, loss_struct  # release the representation graph roots
        mem_report(f'step {step} after L_struct backward', is_main and first)

        # (c) preservation pass: own forward + backward, grads accumulate
        loss_pres_f = 0.0
        if args.beta > 0:
            renc = replay.next_batch(args.replay_seqs, device)
            pol_logits = model(**renc, use_cache=False).logits
            with torch.no_grad():
                ref_logits = ref_model(**renc, use_cache=False).logits
            loss_pres = token_kl(pol_logits, ref_logits,
                                 renc['attention_mask'].bool())
            (args.beta * loss_pres).backward()
            loss_pres_f = float(loss_pres)
            del renc, pol_logits, ref_logits, loss_pres
            mem_report(f'step {step} after L_pres backward', is_main and first)

        loss_f = loss_struct_f + args.beta * loss_pres_f

        # (d) update: FSDP reduces sharded grads; layer_logits synced manually
        if layer_logits is not None:
            if layer_logits.grad is None:
                layer_logits.grad = torch.zeros_like(layer_logits)
            elif dist_is_on():
                all_reduce_(layer_logits.grad)
                layer_logits.grad.div_(world)
            torch.nn.utils.clip_grad_norm_([layer_logits], args.max_grad_norm)
        grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        # free the sharded fp32 grads immediately (instead of at the top of
        # the next step) so they don't sit through validation / saving
        optimizer.zero_grad(set_to_none=True)
        mem_report(f'step {step} after optimizer step', is_main and first)

        # ------------------------------------------------------ logging
        if is_main and (step % args.log_interval == 0 or step == 1):
            ev = {'event': 'train', 'step': step,
                  'loss': round(loss_f, 5),
                  'loss_struct': round(loss_struct_f, 5),
                  'loss_pres_kl': round(loss_pres_f, 6),
                  'lr': scheduler.get_last_lr()[0],
                  'grad_norm': round(float(grad_norm), 4),
                  'triplets': int(len(trip['i'])) if trip else 0,
                  'triplets_valid_pre_cap': n_pre_cap,
                  'sec_per_step': round((time.time() - t_last) / args.log_interval, 3)}
            if trip_acc is not None:
                ev.update(family_stats(trip_acc, trip['fam']))
            if layer_logits is not None:
                w = F.softmax(layer_logits, 0).detach().cpu().numpy()
                ev['layer_w'] = {int(l): round(float(v), 4)
                                 for l, v in zip(inc_layers, w)}
            log(ev)
            print(f"step {step:5d} | loss {ev['loss']:.4f} "
                  f"(struct {ev['loss_struct']:.4f}, kl {ev['loss_pres_kl']:.5f}) "
                  f"| trip {ev['triplets']} | {ev['sec_per_step']:.2f}s/step")
            t_last = time.time()

        # ------------------------------------------------------ validation
        if step % args.eval_interval == 0 or step == args.num_steps:
            flags.zero_()
            vm = run_validation(model, ref_model, tokenizer, val_bank, rsa_fx,
                                trip_fx, replay_val, mu, inc_t, inc_layers,
                                args, device, ref_ppl_cache, judge_fx)
            mem_report(f'eval step {step}', is_main)
            if is_main:
                vm.update({'event': 'val', 'step': step})
                # selection stays on val RSA: judgment accuracy is logged as a
                # passive monitor and never enters the decision.
                improved = vm['val/rsa'] > best_val_rsa + args.min_delta
                if improved:
                    best_val_rsa = vm['val/rsa']
                    evals_since_best = 0
                    flags[1] = 1.0
                else:
                    evals_since_best += 1
                vm['val/best_rsa'] = best_val_rsa
                vm['val/is_best'] = improved
                log(vm)
                print(f"  eval {step}: RSA {vm['val/rsa']:.4f} "
                      f"(best {best_val_rsa:.4f}) | cross {vm['val/rsa_cross_domain']:.4f} "
                      f"| trip_acc {vm.get('val/triplet_acc', float('nan')):.4f} "
                      f"| judge {vm.get('val/judge_acc', float('nan')):.4f} "
                      f"| KL {vm.get('val/kl_ref_policy', float('nan')):.5f} "
                      f"| dppl {vm.get('val/replay_ppl_delta_pct', float('nan')):+.2f}%")
                if evals_since_best >= args.patience:
                    print(f'Early stop: no val RSA gain in {args.patience} evals')
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
        log({'event': 'done', 'step': step, 'best_val_rsa': best_val_rsa})
        metrics_f.close()
        print(f'Done. Best val RSA {best_val_rsa:.4f}. Outputs in {out_dir}')
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
