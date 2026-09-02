#!/usr/bin/env python3
"""
DPO arm: behavioral alignment on the same annotations (multi-GPU, H200,
full-parameter)
========================================================================

The converse arm of the ReSO experiment. Here behavior is trained and geometry
is measured; in the ReSO arm geometry is trained and behavior is measured.
This script optimizes the conditional output distribution p_theta(y|x) toward
preferred moral-judgment completions with standard DPO; the model's internal
concept organization receives no direct training signal and is logged only as
a passive monitor (never used for selection or early stopping).

Preference pairs are constructed from the existing Social-Chem annotations —
no external preference data:

  prompt    a moral-judgment question about the raw action text, rendered in
            the deploy format (chat template, generation prompt) — behavior is
            trained in the format behavior is elicited in
  chosen    a judgment matching the annotation: pole (virtue/vice/neutral)
            verbalized at an intensity bucketed from the membership score m
  rejected  a misjudgment: the opposite pole at matched intensity (or, for a
            fraction of pole items, a false "neutral" — mirroring the
            pole-vs-neutral distinction the annotations define)

Experimental controls matched to the ReSO arm:
  - same base model; both arms train the identical full-parameter set (the
    whole decoder stack, embeddings/unembedding frozen by default in both)
  - same stratified sampler over the (domain x pole x typicality) buckets, so
    per-step item exposure is distributed identically across arms
  - reference model = a frozen bf16 replica of the initial weights (KL is
    implicit, inside the DPO logit via reference anchoring — no separate KL
    term, per the DPO column of the correspondence table)
  - prompt/response surface templates are shared between chosen and rejected,
    so polarity is the only systematic difference within a pair

Parallelism mirrors the ReSO arm: FSDP FULL_SHARD (fp32 sharded masters and
AdamW state, bf16 compute, per-layer activation checkpointing) for the policy,
plus the frozen bf16 reference replica per GPU. Validation runs on every rank
(FSDP forwards are collectives); rank 0 logs and decides. Each process
transiently needs ~50GB host RAM while loading policy + reference.

Monitors (metrics.jsonl): val DPO loss / preference accuracy / reward margins
(selection + early stopping), and passive per-layer RSA against the human RDM
on the same held-out fixture the ReSO arm uses. RSA is centered by an EMA of
the per-layer mean (mirroring the ReSO arm's training-time centering) rather
than a fresh per-eval sample mean; since this arm never runs a representation
pass during training, the EMA is initialized from a pre-training pass over the
train bank (--ema_init_batches) and thereafter updated once per validation
call (--ema_momentum). A baseline validation (RSA, pref_acc, DPO loss, replay
ppl) is logged at step 0 before any optimizer step, excluded from
best-checkpoint selection and early stopping. Also logged: 3-way judgment
accuracy (argmax over virtue/vice/neutral phrasings of a held-out item, scored
by teacher-forced mean per-token logprob) on the same fixture and with the same
scoring the ReSO arm now logs, so the arms are directly comparable — a cousin
of this arm's objective, so confirmatory here and evidential there; and optional
replay perplexity drift if --replay_data is given. Checkpoints (best/, final/) are
plain HF model directories, directly loadable by the Part 1 geometry
diagnostics — the endpoints — post hoc. Runs are not resumable: training always
starts from --model_path at step 0, which is also the frozen DPO reference.

Launch (8x H200):

  torchrun --standalone --nproc_per_node=8 dpo_train.py \
      --model_path /path/to/Qwen3-8B \
      --output_dir ./outputs/dpo_full

Requires: torch >= 2.1, transformers >= 4.40, pandas, numpy.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from reso_train import (MoralItemBank, PHRASES, PROMPT_TEMPLATES,
                        RESPONSE_TEMPLATES, _strength, all_reduce_, barrier,
                        broadcast_, build_judgment_fixture, build_rsa_fixture,
                        completion_logprobs, decoder_layer_cls, dist_is_on,
                        encode_actions, freeze_io_embeddings, judgment_monitor,
                        load_hf_model, load_replay_val_batches, pooled_forward,
                        render_prompt, sample_batch, save_full_checkpoint,
                        setup_distributed, shifted_nll, spearman, wrap_fsdp)

# ============================================================================
# Pair construction from the annotations
#
# The elicitation vocabulary (PROMPT_TEMPLATES / RESPONSE_TEMPLATES / PHRASES),
# render_prompt and completion_logprobs now live in reso_train so that arm can
# run the same judgment monitor without importing this one; they are re-exported
# above, so `from dpo_train import ...` keeps working for the eval scripts.
# ============================================================================

def build_pair(text, pole, m, rng, neutral_rejected_frac=0.25):
    """One (prompt, chosen, rejected) triple. Chosen and rejected share the
    same surface templates; only the judgment phrase differs."""
    prompt = PROMPT_TEMPLATES[int(rng.integers(len(PROMPT_TEMPLATES)))].format(
        action=text.strip())
    resp_t = RESPONSE_TEMPLATES[int(rng.integers(len(RESPONSE_TEMPLATES)))]
    pole = int(pole)
    if pole == 0:
        chosen_phrase = PHRASES[0]['mid']
        rejected_phrase = PHRASES[int(rng.choice([1, -1]))]['mid']
    else:
        s = _strength(m)
        chosen_phrase = PHRASES[pole][s]
        if rng.random() < neutral_rejected_frac:
            rejected_phrase = PHRASES[0]['mid']
        else:
            rejected_phrase = PHRASES[-pole][s]
    return dict(prompt=prompt,
                chosen=resp_t.format(phrase=chosen_phrase),
                rejected=resp_t.format(phrase=rejected_phrase),
                pole=pole, m=float(m))


def batch_pairs_from_bank(bank, n_pairs, rng, neutral_rejected_frac):
    """Stratified item draw (same sampler as the ReSO arm) -> DPO pairs."""
    _, segments = sample_batch(bank, n_pairs, rng)
    pairs = []
    for seg in segments:
        for li in range(len(seg['rows'])):
            pairs.append(build_pair(bank.texts[seg['rows'][li]],
                                    seg['pole'][li], seg['typ'][li],
                                    rng, neutral_rejected_frac))
    return pairs


def collate_pairs(tokenizer, pairs, device, max_prompt_tokens, max_resp_tokens):
    """[chosen_0..chosen_{B-1}, rejected_0..rejected_{B-1}] with a completion
    mask; log-probs are summed over completion tokens only."""
    texts = ([(p['prompt'], p['chosen']) for p in pairs] +
             [(p['prompt'], p['rejected']) for p in pairs])
    seqs, cmasks = [], []
    for prompt, resp in texts:
        pid = tokenizer(render_prompt(tokenizer, prompt),
                        add_special_tokens=False)['input_ids'][:max_prompt_tokens]
        rid = (tokenizer(resp, add_special_tokens=False)['input_ids'][:max_resp_tokens]
               + [tokenizer.eos_token_id])
        seqs.append(pid + rid)
        cmasks.append([0] * len(pid) + [1] * len(rid))
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    cm = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, (s, c) in enumerate(zip(seqs, cmasks)):
        ids[r, :len(s)] = torch.tensor(s)
        attn[r, :len(s)] = 1
        cm[r, :len(s)] = torch.tensor(c)
    return dict(input_ids=ids.to(device), attention_mask=attn.to(device),
                completion_mask=cm.to(device))


# ============================================================================
# DPO loss
# ============================================================================

def dpo_loss_and_stats(pol_logp, ref_logp, beta_dpo):
    b = pol_logp.shape[0] // 2
    pol_w, pol_l = pol_logp[:b], pol_logp[b:]
    ref_w, ref_l = ref_logp[:b], ref_logp[b:]
    rw = beta_dpo * (pol_w - ref_w)   # implicit rewards
    rl = beta_dpo * (pol_l - ref_l)
    logits = rw - rl
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        stats = dict(acc=float((logits > 0).float().mean()),
                     reward_margin=float((rw - rl).mean()),
                     reward_chosen=float(rw.mean()),
                     reward_rejected=float(rl.mean()),
                     logp_chosen=float(pol_w.mean()),
                     logp_rejected=float(pol_l.mean()))
    return loss, stats


# ============================================================================
# Validation (runs on ALL ranks — FSDP forwards are collective)
# ============================================================================

def build_val_pairs(bank, n_pairs, rng, neutral_rejected_frac):
    pairs = []
    while len(pairs) < n_pairs:
        pairs.extend(batch_pairs_from_bank(bank, min(260, n_pairs), rng,
                                           neutral_rejected_frac))
    return pairs[:n_pairs]


@torch.no_grad()
def rsa_monitor(model, tokenizer, texts_by_row, fx, n_layers, device,
                max_action_tokens, eval_batch_items, mu, ema_momentum):
    """Passive geometry monitor: per-layer RSA of raw-action cosines against
    the human RDM, centered by an EMA of the per-layer mean (mirrors the ReSO
    arm's training-time centering instead of a fresh per-eval sample mean).
    mu is updated in place from this call's batch mean before being used to
    center that same batch, matching the ReSO arm's update-then-use order.
    Measured, never optimized or selected on, in this arm."""
    chunks = []
    rows = fx['rows']
    for s in range(0, len(rows), eval_batch_items):
        enc = encode_actions(tokenizer, [texts_by_row[r] for r in rows[s:s + eval_batch_items]],
                             max_action_tokens, device)
        chunks.append(pooled_forward(model, enc, n_layers))
    P = torch.cat(chunks, dim=1)                          # [L_dec, n, d]
    mu.mul_(ema_momentum).add_(P.mean(1) * (1.0 - ema_momentum))
    Zn = F.normalize(P - mu.unsqueeze(1), dim=-1)
    i0 = torch.from_numpy(fx['iu'][0]).to(device)
    i1 = torch.from_numpy(fx['iu'][1]).to(device)
    within, cross = [], []
    for li in range(n_layers):
        pv = (Zn[li] @ Zn[li].T)[i0, i1].cpu().numpy()
        within.append(spearman(pv[fx['within']], fx['sh'][fx['within']]))
        cross.append(spearman(pv[fx['cross']], fx['sh'][fx['cross']]))
    return dict(rsa=float(np.mean(within)),
                rsa_per_layer={int(l): round(v, 4) for l, v in enumerate(within)},
                rsa_cross_domain=float(np.mean(cross)))


@torch.no_grad()
def run_validation(model, ref_model, tokenizer, val_pairs, val_bank, rsa_fx,
                   replay_val, args, device, ref_ppl_cache, mu, judge_fx=None):
    model.eval()
    m = {}

    # --- behavioral objective: val DPO loss / preference accuracy ---
    losses, accs, margins, n_seen = [], [], [], 0
    for s in range(0, len(val_pairs), args.pairs_per_step):
        chunk = val_pairs[s:s + args.pairs_per_step]
        enc = collate_pairs(tokenizer, chunk, device,
                            args.max_prompt_tokens, args.max_resp_tokens)
        pol = model(input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'], use_cache=False).logits
        ref = ref_model(input_ids=enc['input_ids'],
                        attention_mask=enc['attention_mask'], use_cache=False).logits
        pol_lp = completion_logprobs(pol, enc['input_ids'], enc['completion_mask'])
        ref_lp = completion_logprobs(ref, enc['input_ids'], enc['completion_mask'])
        loss, stats = dpo_loss_and_stats(pol_lp, ref_lp, args.beta_dpo)
        w = len(chunk)
        losses.append(float(loss) * w)
        accs.append(stats['acc'] * w)
        margins.append(stats['reward_margin'] * w)
        n_seen += w
    m['val/dpo_loss'] = sum(losses) / n_seen
    m['val/pref_acc'] = sum(accs) / n_seen
    m['val/reward_margin'] = sum(margins) / n_seen

    # --- passive geometry monitor (EMA-centered, mirrors the ReSO arm) ---
    geo = rsa_monitor(model, tokenizer, val_bank.texts, rsa_fx, args.n_layers,
                      device, args.max_action_tokens, args.eval_batch_items,
                      mu, args.ema_momentum)
    m.update({f'val/{k}': v for k, v in geo.items()})

    # --- 3-way judgment accuracy on the fixture the ReSO arm also scores ---
    # A cousin of this arm's objective (confirmatory here, the cross-arm cell
    # there), but on free-standing items rather than preference pairs and read
    # out by argmax over three candidates rather than against the reference.
    # Logged only: selection and early stopping stay on val/pref_acc.
    if judge_fx:
        jm = judgment_monitor(model, tokenizer, judge_fx, device,
                              args.max_prompt_tokens, args.max_resp_tokens,
                              args.judge_batch_seqs)
        m.update({f'val/{k}': v for k, v in jm.items()})

    # --- optional capability guardrail monitor ---
    if replay_val:
        nll_sum, nll_n, rnll_sum = 0.0, 0, 0.0
        for enc in replay_val:
            enc = {k: v.to(device) for k, v in enc.items()}
            pol = model(**enc, use_cache=False).logits
            nll, n = shifted_nll(pol, enc['input_ids'], enc['attention_mask'])
            nll_sum += float(nll)
            nll_n += n
            if ref_ppl_cache.get('ref_ppl') is None:
                ref = ref_model(**enc, use_cache=False).logits
                rnll, _ = shifted_nll(ref, enc['input_ids'], enc['attention_mask'])
                rnll_sum += float(rnll)
        m['val/replay_ppl'] = math.exp(nll_sum / max(nll_n, 1))
        if ref_ppl_cache.get('ref_ppl') is None:
            ref_ppl_cache['ref_ppl'] = math.exp(rnll_sum / max(nll_n, 1))
        m['val/replay_ppl_ref'] = ref_ppl_cache['ref_ppl']
        m['val/replay_ppl_delta_pct'] = 100.0 * (m['val/replay_ppl'] /
                                                 ref_ppl_cache['ref_ppl'] - 1.0)
    model.train()
    return m


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='DPO arm: behavioral alignment on '
                                            'the Social-Chem annotations (full-parameter).')
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
    p.add_argument('--output_dir', type=str, default='./outputs/dpo')
    # DPO loss / pairs
    p.add_argument('--beta_dpo', type=float, default=0.1, help='DPO temperature on log-ratios')
    p.add_argument('--neutral_rejected_frac', type=float, default=0.25,
                   help='fraction of pole-item pairs whose rejected is a false neutral')
    p.add_argument('--chosen_nll_coef', type=float, default=0.0,
                   help='optional NLL regularizer on chosen completions (off = canonical DPO)')
    p.add_argument('--pairs_per_step', type=int, default=64,
                   help='preference pairs per rank per step (stratified item draw)')
    p.add_argument('--max_prompt_tokens', type=int, default=160)
    p.add_argument('--max_resp_tokens', type=int, default=32)
    # optimization
    p.add_argument('--num_steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=5e-7,
                   help='full-parameter DPO peak LR (canonical range 5e-7..1e-6)')
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
                        '(each is a ~16GB model dir; best/ and final/ are always written)')
    p.add_argument('--val_pairs', type=int, default=1024)
    p.add_argument('--rsa_per_domain', type=int, default=150)
    p.add_argument('--judge_per_domain', type=int, default=100,
                   help='held-out items per domain for the 3-way judgment '
                        'monitor (0 disables it); keep matched to the ReSO arm '
                        'so both arms score the same fixture')
    p.add_argument('--judge_batch_seqs', type=int, default=96,
                   help='candidate sequences per forward in the judgment monitor')
    p.add_argument('--eval_batch_items', type=int, default=512)
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
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    rank, world, local = setup_distributed()
    if not dist_is_on():
        raise SystemExit('FSDP training must be launched with torchrun, e.g.\n'
                         '  torchrun --standalone --nproc_per_node=8 dpo_train.py ...')
    device = torch.device(f'cuda:{local}')
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng([args.seed, rank])

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'args.json', 'w') as f:
            json.dump(vars(args) | {'world_size': world}, f, indent=2)
        metrics_f = open(out_dir / 'metrics.jsonl', 'a')

    def log(event):
        if is_main:
            metrics_f.write(json.dumps(event) + '\n')
            metrics_f.flush()

    # ------------------------------------------------------------ model
    if is_main:
        print(f'Loading {args.model_path} on {world} GPU(s)...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                              local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # fp32 masters; FSDP mixed precision runs compute in bf16
    model = load_hf_model(args.model_path, args.attn_impl, args.local_files_only,
                          torch.float32)
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
    model = wrap_fsdp(model, local, layer_cls,
                      activation_checkpointing=not args.no_activation_checkpointing)

    # DPO reference: frozen bf16 replica of the *initial* weights
    ref_model = load_hf_model(args.model_path, args.attn_impl,
                              args.local_files_only, torch.bfloat16)
    ref_model.config.use_cache = False
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.to(device)

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

    vm0 = run_validation(model, ref_model, tokenizer, val_pairs, val_bank,
                         rsa_fx, replay_val, args, device, ref_ppl_cache, mu,
                         judge_fx)
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
        enc = collate_pairs(tokenizer, pairs, device,
                            args.max_prompt_tokens, args.max_resp_tokens)

        pol_logits = model(input_ids=enc['input_ids'],
                           attention_mask=enc['attention_mask'],
                           use_cache=False).logits
        with torch.no_grad():
            ref_logits = ref_model(input_ids=enc['input_ids'],
                                   attention_mask=enc['attention_mask'],
                                   use_cache=False).logits

        pol_lp = completion_logprobs(pol_logits, enc['input_ids'], enc['completion_mask'])
        with torch.no_grad():
            ref_lp = completion_logprobs(ref_logits, enc['input_ids'], enc['completion_mask'])

        loss, stats = dpo_loss_and_stats(pol_lp, ref_lp, args.beta_dpo)
        if args.chosen_nll_coef > 0:
            b = len(pairs)
            n_tok = enc['completion_mask'][:b, 1:].sum(-1).clamp(min=1).float()
            loss = loss + args.chosen_nll_coef * (-pol_lp[:b] / n_tok).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # ------------------------------------------------------ logging
        if is_main and (step % args.log_interval == 0 or step == 1):
            poles = np.array([p['pole'] for p in pairs])
            ev = {'event': 'train', 'step': step,
                  'loss': round(float(loss), 5),
                  'lr': scheduler.get_last_lr()[0],
                  'grad_norm': round(float(grad_norm), 4),
                  'pairs': len(pairs),
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
                                rsa_fx, replay_val, args, device, ref_ppl_cache,
                                mu, judge_fx)
            if is_main:
                vm.update({'event': 'val', 'step': step})
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
