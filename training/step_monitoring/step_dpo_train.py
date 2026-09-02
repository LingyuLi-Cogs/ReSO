#!/usr/bin/env python3
"""
DPO arm + in-training HarmBench: behavioral alignment with an external safety
monitor
=============================================================================

`dpo_train.py` with the same addition `step_reso_train.py` makes to the ReSO
arm: at each validation the current policy is checkpointed and handed to the
external HarmBench harness (`eval.sh MODEL_PATH CLS_PATH OUT_DIR MODEL_NAME`),
and the attack success rate is logged beside the existing
metrics. The loss, the sampler, the pair construction and the fixtures are
imported unchanged from `dpo_train` (which in turn shares its machinery with
`reso_train`), so this script trains the identical model and only adds
observation.

The whole HarmBench lifecycle — GPU reservation, submission, harvesting,
pruning, the arg surface — is imported from `step_reso_train`, so the two arms
are monitored by literally the same code and their ASR traces are comparable
the way the RSA and judgment fixtures already are.

HarmBench ASR is a *passive external monitor*: it never enters the loss, the
best-checkpoint selection, or the early-stopping rule. Selection stays on
val/pref_acc. Note this arm's expected direction — DPO on moral judgments moves
behavior directly, and the interesting question is what that does to OOD attack
success — which is exactly why the trace is worth having per step rather than
only at the endpoints.

GPUs
----
`eval.sh` runs vLLM at gpu_memory_utilization=0.9 over *all visible* GPUs, so it
cannot share a device with this FSDP job. Reserve GPUs for it and train on the
rest:

  torchrun --standalone --nproc_per_node=6 training/step_monitoring/step_dpo_train.py \
      --model_path /path/to/Qwen3-8B \
      --output_dir ./outputs/dpo_full \
      --harmbench_script ./evaluation/HarmBench/eval.sh \
      --harmbench_cls_path /path/to/HarmBench-Llama-2-13b-cls \
      --harmbench_gpus 6,7

See the `step_reso_train` docstring for the modes (async / sync / queue), the
cost knobs (--harmbench_every, `--harmbench_env MAX_BEHAVIORS=40`), the
checkpoint pruning policy, the absolute-path rules, and the metrics.jsonl event
schema — all identical here, since the whole HarmBench arg surface comes from
that module. Pass the harness entrypoint with --harmbench_script.

Requires: everything dpo_train requires, plus a working HarmBench install
(vllm) and a local copy of the classifier.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dpo_train import (batch_pairs_from_bank, build_val_pairs, collate_pairs,
                       dpo_loss_and_stats, run_validation)
from dpo_train import parse_args as base_parse_args
from reso_train import (MoralItemBank, all_reduce_, barrier, broadcast_,
                        build_judgment_fixture, build_rsa_fixture,
                        completion_logprobs, decoder_layer_cls, dist_is_on,
                        encode_actions, freeze_io_embeddings, load_hf_model,
                        load_replay_val_batches, pooled_forward, sample_batch,
                        save_full_checkpoint, wrap_fsdp)
from step_reso_train import (build_runner, harvest, parse_args_with_harmbench,
                             setup_distributed, submit_job)


def parse_args():
    return parse_args_with_harmbench(base_parse_args, 'DPO')


# ============================================================================
# Main — dpo_train.main with the HarmBench hooks; the training path is byte-for-
# byte the same computation.
# ============================================================================

def main():
    args = parse_args()
    rank, world, local = setup_distributed(args.dist_timeout_min)
    if not dist_is_on():
        raise SystemExit('FSDP training must be launched with torchrun, e.g.\n'
                         '  torchrun --standalone --nproc_per_node=6 step_dpo_train.py ...')
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

    # HarmBench lives on rank 0 only; every rank still participates in the
    # checkpoint saves it triggers, which are FSDP collectives.
    runner = build_runner(args, out_dir, log, world, is_main)
    # absolute: it is handed to an external process and matched against the
    # runner's own resolved ckpt_root by the prune guard
    hb_ckpt_root = (out_dir / 'harmbench_ckpt').resolve()

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
    n_evals = 0
    flags = torch.zeros(3, device=device)  # [stop, save_best, harmbench]

    vm0 = run_validation(model, ref_model, tokenizer, val_pairs, val_bank,
                         rsa_fx, replay_val, args, device, ref_ppl_cache, mu,
                         judge_fx)
    # Step-0 HarmBench scores --model_path directly: the untrained reference is
    # already on disk, so the baseline costs no checkpoint.
    if is_main and runner and args.harmbench_at_step0:
        submit_job(runner, log, 0, args.model_path)
    if is_main:
        vm0.update({'event': 'val', 'step': 0, 'val/is_best': False})
        if runner:
            vm0.update(runner.status_fields())
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
            n_evals += 1
            vm = run_validation(model, ref_model, tokenizer, val_pairs, val_bank,
                                rsa_fx, replay_val, args, device, ref_ppl_cache,
                                mu, judge_fx)
            # Collect anything HarmBench finished since the last eval before the
            # val record is written, so it reports the freshest ASR available.
            harvest(runner, log, is_main)
            if is_main:
                vm.update({'event': 'val', 'step': step})
                # selection stays on val/pref_acc: the passive RSA monitor and
                # HarmBench ASR never enter the decision.
                improved = vm['val/pref_acc'] > best_acc + args.min_delta
                if improved:
                    best_acc = vm['val/pref_acc']
                    evals_since_best = 0
                    flags[1] = 1.0
                else:
                    evals_since_best += 1
                vm['val/best_pref_acc'] = best_acc
                if runner:
                    due = args.harmbench_every > 0 and n_evals % args.harmbench_every == 0
                    if due and runner.can_submit():
                        flags[2] = 1.0
                    elif due:
                        log(dict(event='harmbench_skip', step=step,
                                 reason=f'{runner.n_running} job(s) still running '
                                        f'(--harmbench_max_inflight '
                                        f'{args.harmbench_max_inflight})'))
                        print(f'  [harmbench] step {step} skipped: '
                              f'{runner.n_running} job(s) still running')
                    vm.update(runner.status_fields())
                vm['val/is_best'] = improved
                log(vm)
                print(f"  eval {step}: pref_acc {vm['val/pref_acc']:.4f} "
                      f"(best {best_acc:.4f}) | dpo_loss {vm['val/dpo_loss']:.4f} "
                      f"| judge {vm.get('val/judge_acc', float('nan')):.4f} "
                      f"| RSA(passive) {vm['val/rsa']:.4f} "
                      f"| cross {vm['val/rsa_cross_domain']:.4f}"
                      + (f" | dppl {vm['val/replay_ppl_delta_pct']:+.2f}%"
                         if 'val/replay_ppl_delta_pct' in vm else '')
                      + (f" | {runner.summary_str()}" if runner else ''))
                if evals_since_best >= args.patience:
                    print(f'Early stop: no val pref-acc gain in {args.patience} evals')
                    flags[0] = 1.0
            broadcast_(flags)
            if flags[1] > 0:  # collective save on all ranks
                save_full_checkpoint(out_dir / 'best', model, tokenizer,
                                     trainer_state(step), is_main)
            if flags[2] > 0:  # HarmBench round: collective save, then rank-0 launch
                hb_ckpt = hb_ckpt_root / f'step_{step:06d}'
                save_full_checkpoint(hb_ckpt, model, tokenizer,
                                     trainer_state(step), is_main)
                if is_main:
                    submit_job(runner, log, step, hb_ckpt / 'model', ckpt_dir=hb_ckpt)
                # sync mode blocks rank 0 for the whole HarmBench run; hold the
                # others here so the job resumes in lockstep rather than piling
                # into the next collective.
                if args.harmbench_mode == 'sync':
                    barrier()
            if flags[0] > 0:
                break

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_full_checkpoint(out_dir / f'step_{step:06d}', model, tokenizer,
                                 trainer_state(step), is_main)

    # ------------------------------------------------------------ final save
    save_full_checkpoint(out_dir / 'final', model, tokenizer,
                         trainer_state(step), is_main)
    if is_main and runner and args.harmbench_at_end:
        # step+1 keeps the final run's label distinct from the last periodic one
        # if the last eval already scored this step.
        submit_job(runner, log, step + 1, out_dir / 'final' / 'model')
    if is_main and runner:
        harvest(runner, log, is_main)
        if runner.n_running:
            print(f'Waiting up to {args.harmbench_end_timeout}s for '
                  f'{runner.n_running} HarmBench job(s)...')
            for rec in runner.wait_all(args.harmbench_end_timeout):
                log(rec)
    if is_main:
        done = {'event': 'done', 'step': step, 'best_pref_acc': best_acc}
        if runner and runner.last:
            done['harmbench/asr_last'] = runner.last['harmbench/asr']
            done['harmbench/asr_last_step'] = runner.last['step']
        log(done)
        metrics_f.close()
        print(f'Done. Best val preference accuracy {best_acc:.4f}. Outputs in {out_dir}')
        if runner:
            print(f'HarmBench: {runner.summary_str()} — full trace in '
                  f'{out_dir / "metrics.jsonl"} (event=harmbench) and '
                  f'{runner.out_dir / "harmbench_asr.csv"}')
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
