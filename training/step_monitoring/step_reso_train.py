#!/usr/bin/env python3
"""
ReSO + in-training HarmBench: geometry training with an external safety monitor
===============================================================================

`reso_train.py` with one addition: at each validation the current policy is
checkpointed and handed to the external HarmBench harness
(`eval.sh MODEL_PATH CLS_PATH OUT_DIR MODEL_NAME`), whose attack success rate is
logged alongside the existing metrics. Everything else — the
loss, the sampler, the fixtures, the selection scalar — is imported unchanged
from `reso_train`, so the two scripts train the identical model and this one
only adds observation.

HarmBench ASR is a *passive external monitor*, in the same sense as the arms'
existing crossed monitors: it never enters the loss, the best-checkpoint
selection, or the early-stopping rule. Selection stays on val RSA. The point is
a within-run OOD safety trace — ASR against training step — not a new objective.

GPUs
----
`eval.sh` runs vLLM at gpu_memory_utilization=0.9 over *all visible* GPUs, so it
cannot share a device with this FSDP job. Reserve GPUs for it and train on the
rest:

  CUDA_VISIBLE_DEVICES unset, 8 physical GPUs, train on 0-5, evaluate on 6-7:

    torchrun --standalone --nproc_per_node=6 training/step_monitoring/step_reso_train.py \
        --model_path /path/to/Qwen3-8B \
        --output_dir ./outputs/reso_beta0.1 --beta 0.1 \
        --harmbench_script ./evaluation/HarmBench/eval.sh \
        --harmbench_cls_path /path/to/HarmBench-Llama-2-13b-cls \
        --harmbench_gpus 6,7

--harmbench_gpus is required: launching without it would let vLLM grab the
training GPUs and OOM mid-run. `--harmbench_allow_shared_gpus` overrides the
overlap check for the case where you know a device is free.

Paths
-----
Paths passed to the harness are resolved before launch. The HarmBench entrypoint
is intentionally not hard-coded; pass it with --harmbench_script so the same
public code works from any checkout location.

Modes (--harmbench_mode)
------------------------
  async  (default) launch and keep training; harvest whichever jobs have
         finished at each subsequent validation. ASR therefore lags the step it
         was measured at — every record carries its own `step`, and the eval
         line reports `harmbench_asr@<step>` so the lag is explicit.
  sync   block the whole job until HarmBench returns, so each eval line carries
         its own step's ASR. Correct but expensive; the process group is
         initialized with --dist_timeout_min so the NCCL watchdog does not abort
         the idle ranks while rank 0 waits.
  queue  write the exact commands to <out_dir>/harmbench/queue.jsonl and launch
         nothing. Zero GPU contention; run the queue afterwards.

Cost
----
At --harmbench_every 1 (the default: every eval) each round costs one full
checkpoint gather plus a complete HarmBench pass. Two knobs make that
affordable: --harmbench_every N to subsample evals, and
`--harmbench_env MAX_BEHAVIORS=40` for a fast subset. Only
--harmbench_max_inflight jobs run at once (default 1); rounds arriving while the
previous one is still running are skipped and logged as `harmbench_skip`, so
jobs never pile onto the same eval GPUs. Checkpoints written for HarmBench live
under <out_dir>/harmbench_ckpt/step_NNNNNN and are pruned once their job
finishes, keeping the --harmbench_keep_ckpt most recent (default 2, 0 = keep
all). Pruning only ever touches directories this run created under that path.

Logging (metrics.jsonl)
-----------------------
  event=harmbench       one per completed job: harmbench/asr, per-category ASR,
                        refusal_rate, n_behaviors, unparsed_labels, wall time,
                        and the step whose checkpoint it scored.
  event=harmbench_skip  a round skipped because a job was still in flight.
  event=val             carries val/harmbench_asr + val/harmbench_asr_step (the
                        most recent completed result, which in async mode is
                        usually an earlier step) and val/harmbench_pending.
The step-0 baseline scores --model_path itself, so the trace starts from the
untrained reference at no extra checkpoint cost.

Requires: everything reso_train requires, plus a working HarmBench install
(vllm) and a local copy of the classifier.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from reso_train import (MoralItemBank, ReplayStream, all_reduce_, barrier,
                        broadcast_, build_judgment_fixture, build_rsa_fixture,
                        build_val_triplet_fixture, decoder_layer_cls,
                        dist_is_on, encode_actions, family_stats,
                        freeze_io_embeddings, load_hf_model,
                        load_replay_val_batches, mine_triplets, pooled_forward,
                        run_validation, sample_batch, save_full_checkpoint,
                        structure_loss, token_kl, wrap_fsdp)
from reso_train import parse_args as base_parse_args

# ============================================================================
# HarmBench integration
#
# The harness is an external process, not a library call: eval.sh runs the
# target model and then the 13B classifier as two separate vLLM processes so
# each releases its GPU memory before the next loads. This side owns only the
# lifecycle — checkpoint, launch, harvest, prune — and reads results back from
# the summary JSON the harness writes.
# ============================================================================

# torchrun exports these into the environment; vLLM (and torch, inside it) would
# otherwise try to join *this* job's process group and hang or crash. The child
# must look like a plain single-process launch.
TORCHRUN_ENV = (
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
    'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_NAME', 'ROLE_WORLD_SIZE',
    'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID',
    'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',
    'TORCHELASTIC_USE_AGENT_STORE', 'TORCHELASTIC_ERROR_FILE',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'NCCL_ASYNC_ERROR_HANDLING',
    'TORCH_DISTRIBUTED_DEBUG',
)

CKPT_NAME_RE = re.compile(r'^step_\d{6}$')      # what _prune is allowed to delete

def sanitize(name):
    """Identical to harmbench_eval.sanitize — the harness names its output files
    with it, so the reader has to reproduce it exactly to find them."""
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in name)


def visible_physical_gpus():
    """Physical device ids this process can see, in the order CUDA numbers them."""
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None or cvd.strip() == '':
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return list(range(n))
    return [int(x) for x in cvd.split(',') if x.strip() != '']


def training_physical_gpus(world):
    """Physical ids the training ranks occupy (local rank -> visible device)."""
    vis = visible_physical_gpus()
    return {vis[i] for i in range(min(world, len(vis)))}


def setup_distributed(timeout_min):
    """As reso_train.setup_distributed, but with an explicit process-group
    timeout. The default 10 minutes is not enough here: a full-state-dict gather
    for a large policy, and in --harmbench_mode sync the HarmBench run itself,
    both leave the non-participating ranks idle inside a collective long enough
    for the NCCL watchdog to abort the job."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl',
                                timeout=timedelta(minutes=timeout_min))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get('LOCAL_RANK', rank % max(1, torch.cuda.device_count())))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


class HarmBenchRunner:
    """Rank 0 only: submits checkpoints to eval.sh and harvests ASR.

    Jobs are tracked in submission order. `poll` is non-blocking and returns the
    records of jobs that finished since the last call; `submit` blocks in sync
    mode. Nothing here touches the training state.
    """

    def __init__(self, script, cls_path, out_dir, ckpt_root, run_name, gpus,
                 env_extra, mode, timeout_s, keep_ckpt, max_inflight, log):
        self.script = Path(script).resolve()
        self.cls_path = Path(cls_path).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.ckpt_root = Path(ckpt_root).resolve()
        self.log_dir = self.out_dir / 'logs'
        self.run_name = run_name
        self.gpus = gpus
        self.env_extra = env_extra
        self.mode = mode
        self.timeout_s = timeout_s
        self.keep_ckpt = keep_ckpt
        self.max_inflight = max_inflight
        self.log = log
        self.jobs = []
        self.last = None                      # most recent completed record
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.out_dir / 'queue.jsonl'

    # -------------------------------------------------------------- helpers
    def _env(self):
        env = {k: v for k, v in os.environ.items() if k not in TORCHRUN_ENV}
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in self.gpus)
        env.update(self.env_extra)
        return env

    def _cmd(self, name, model_path):
        return ['bash', str(self.script), str(Path(model_path).resolve()),
                str(self.cls_path), str(self.out_dir), name]

    @property
    def n_running(self):
        return sum(1 for j in self.jobs if j['status'] == 'running')

    def can_submit(self):
        return self.mode == 'queue' or self.n_running < self.max_inflight

    # -------------------------------------------------------------- submit
    def submit(self, step, model_path, ckpt_dir=None):
        """Launch (or, in queue mode, record) one HarmBench run. `ckpt_dir` is
        the directory this run may prune afterwards; None for the base model."""
        name = sanitize(f'{self.run_name}_step{step:06d}')
        cmd = self._cmd(name, model_path)
        job = dict(step=step, name=name, cmd=cmd, ckpt_dir=str(ckpt_dir) if ckpt_dir else None,
                   t0=time.time(), status='running', proc=None, log_path=None)

        if self.mode == 'queue':
            job['status'] = 'queued'
            with open(self.queue_path, 'a') as f:
                f.write(json.dumps(dict(step=step, name=name, cmd=cmd,
                                        env={'CUDA_VISIBLE_DEVICES':
                                             ','.join(str(g) for g in self.gpus),
                                             **self.env_extra})) + '\n')
            self.jobs.append(job)
            self.log(dict(event='harmbench_queued', step=step, name=name))
            print(f'  [harmbench] queued step {step} -> {self.queue_path.name}')
            return job

        log_path = self.log_dir / f'{name}.log'
        job['log_path'] = str(log_path)
        job['fh'] = open(log_path, 'w')
        job['proc'] = subprocess.Popen(cmd, cwd=str(self.script.parent),
                                       env=self._env(), stdout=job['fh'],
                                       stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.jobs.append(job)
        print(f'  [harmbench] launched step {step} on GPU(s) '
              f'{",".join(str(g) for g in self.gpus)} -> {log_path.name}')
        if self.mode == 'sync':
            self._wait(job, self.timeout_s)
            rec = self._finish(job)
            self._prune()
            return rec
        return job

    # -------------------------------------------------------------- harvest
    def poll(self):
        """Non-blocking: finish + parse every job that has exited. Returns the
        list of records produced by this call."""
        done = []
        for job in self.jobs:
            if job['status'] != 'running':
                continue
            if job['proc'].poll() is None:
                if self.timeout_s and (time.time() - job['t0']) > self.timeout_s:
                    self._kill(job)
                else:
                    continue
            done.append(self._finish(job))
        if done:
            self._prune()
        return done

    def wait_all(self, timeout_s):
        """Block until every running job exits (or `timeout_s` elapses).
        Returns the records collected while waiting."""
        recs, deadline = [], time.time() + timeout_s if timeout_s else None
        while any(j['status'] == 'running' for j in self.jobs):
            recs.extend(self.poll())
            if not any(j['status'] == 'running' for j in self.jobs):
                break
            if deadline and time.time() > deadline:
                for job in self.jobs:
                    if job['status'] == 'running':
                        self._kill(job)
                recs.extend(self.poll())
                break
            time.sleep(10)
        return recs

    def _wait(self, job, timeout_s):
        try:
            job['proc'].wait(timeout=timeout_s if timeout_s else None)
        except subprocess.TimeoutExpired:
            self._kill(job)

    def _kill(self, job):
        try:
            os.killpg(os.getpgid(job['proc'].pid), signal.SIGTERM)
            try:
                job['proc'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(job['proc'].pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        job['timed_out'] = True

    def _finish(self, job):
        """Collect a finished process into a flat metrics record."""
        rc = job['proc'].returncode
        job['fh'].close()
        dt = round(time.time() - job['t0'], 1)
        rec = dict(event='harmbench', step=job['step'], name=job['name'],
                   returncode=rc, wall_sec=dt, log=job['log_path'])

        summary_path = self.out_dir / f'summary_{job["name"]}.json'
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                summary = None

        if summary is None or 'asr' not in summary:
            job['status'] = 'timeout' if job.get('timed_out') else 'failed'
            rec['status'] = job['status']
            rec['error'] = (f'no usable {summary_path.name}; '
                            f'see {job["log_path"]}')
            print(f'  [harmbench] step {job["step"]} FAILED (rc={rc}, {dt}s) — '
                  f'{rec["error"]}')
            return rec

        job['status'] = 'done'
        rec['status'] = 'done'
        rec['harmbench/asr'] = float(summary['asr'])
        rec['harmbench/n_behaviors'] = int(summary.get('n_behaviors', 0))
        rec['harmbench/unparsed_labels'] = int(summary.get('unparsed_labels', 0))
        if 'refusal_rate' in summary:
            rec['harmbench/refusal_rate'] = float(summary['refusal_rate'])
        for cat, v in (summary.get('asr_by_category') or {}).items():
            rec[f'harmbench/asr_{sanitize(cat)}'] = float(v)
        for k in ('skipped_total_exceeds_length',):
            if k in summary:
                rec[f'harmbench/{k}'] = int(summary[k])
        rec['harmbench/summary_file'] = str(summary_path)
        self.last = rec
        print(f'  [harmbench] step {job["step"]}: ASR {rec["harmbench/asr"]:.4f} '
              f'over {rec["harmbench/n_behaviors"]} behaviors ({dt}s)')
        return rec

    # -------------------------------------------------------------- prune
    def _prune(self):
        """Delete the checkpoints of finished jobs, keeping the most recent
        `keep_ckpt`. Guarded three ways: the directory must have been created by
        this run (it is in self.jobs), must sit directly under ckpt_root, and
        must match the step_NNNNNN name this script writes."""
        if self.keep_ckpt <= 0:
            return
        finished = [j for j in self.jobs
                    if j['ckpt_dir'] and j['status'] in ('done', 'failed', 'timeout')
                    and not j.get('pruned')]
        for job in finished[:max(0, len(finished) - self.keep_ckpt)]:
            # resolve before comparing: --output_dir is relative by default, and
            # ckpt_root is stored resolved, so an unresolved compare would never
            # match and pruning would silently never happen.
            d = Path(job['ckpt_dir']).resolve()
            job['pruned'] = True
            if d.parent != self.ckpt_root or not CKPT_NAME_RE.match(d.name):
                continue
            if not d.is_dir():
                continue
            shutil.rmtree(d)
            self.log(dict(event='harmbench_prune', step=job['step'], path=str(d)))
            print(f'  [harmbench] pruned {d}')

    # -------------------------------------------------------------- readout
    def status_fields(self):
        """Fields folded into every val event: the freshest completed ASR, the
        step it actually belongs to (async results lag), and the in-flight count."""
        out = {'val/harmbench_pending': self.n_running}
        if self.last is not None:
            out['val/harmbench_asr'] = self.last['harmbench/asr']
            out['val/harmbench_asr_step'] = self.last['step']
        return out

    def summary_str(self):
        if self.last is None:
            return f'ASR pending ({self.n_running} in flight)'
        return (f'ASR {self.last["harmbench/asr"]:.4f}@{self.last["step"]}'
                + (f' ({self.n_running} in flight)' if self.n_running else ''))


def build_runner(args, out_dir, log, world, is_main):
    """Validate on every rank, construct on rank 0.

    The validation deliberately runs everywhere: a rank-0-only SystemExit on a
    bad path or GPU list would leave the other ranks blocked in the next
    collective until the process-group timeout, turning a typo into a silent
    hang. Returns None on non-main ranks and when HarmBench is disabled.
    """
    if args.no_harmbench:
        return None
    if not args.harmbench_script:
        raise SystemExit('--harmbench_script is required unless --no_harmbench is set.')
    if not args.harmbench_cls_path:
        raise SystemExit('--harmbench_cls_path (the HarmBench classifier dir) is '
                         'required; pass --no_harmbench to train without the monitor.')
    script = Path(args.harmbench_script).resolve()
    if not script.exists():
        raise SystemExit(f'HarmBench script not found: {script}')
    if not Path(args.harmbench_cls_path).exists():
        raise SystemExit(f'HarmBench classifier dir not found: {args.harmbench_cls_path}')

    gpus = [int(x) for x in args.harmbench_gpus.split(',') if x.strip() != '']
    if not gpus:
        raise SystemExit('--harmbench_gpus is required: eval.sh runs vLLM at '
                         'gpu_memory_utilization=0.9 over every visible GPU, so it '
                         'needs devices this job is not training on. Train on fewer '
                         'ranks and reserve the rest, e.g. --nproc_per_node=6 '
                         '--harmbench_gpus 6,7.')
    busy = training_physical_gpus(world) & set(gpus)
    if busy and not args.harmbench_allow_shared_gpus:
        raise SystemExit(f'--harmbench_gpus {sorted(set(gpus))} overlaps the GPUs this '
                         f'job trains on ({sorted(busy)}); vLLM would OOM against the '
                         f'FSDP shards. Reduce --nproc_per_node, pick free devices, or '
                         f'pass --harmbench_allow_shared_gpus if you know they are free.')

    env_extra = {}
    for kv in args.harmbench_env:
        if '=' not in kv:
            raise SystemExit(f'--harmbench_env expects KEY=VALUE, got {kv!r}')
        k, v = kv.split('=', 1)
        env_extra[k] = v

    if not is_main:
        return None

    hb_out = Path(args.harmbench_out_dir) if args.harmbench_out_dir else out_dir / 'harmbench'
    runner = HarmBenchRunner(
        script=script, cls_path=args.harmbench_cls_path, out_dir=hb_out,
        ckpt_root=out_dir / 'harmbench_ckpt',
        run_name=args.harmbench_name or out_dir.name, gpus=gpus,
        env_extra=env_extra, mode=args.harmbench_mode,
        timeout_s=args.harmbench_timeout, keep_ckpt=args.harmbench_keep_ckpt,
        max_inflight=args.harmbench_max_inflight, log=log)
    print(f'HarmBench monitor: mode={args.harmbench_mode} every {args.harmbench_every} '
          f'eval(s) on GPU(s) {args.harmbench_gpus} -> {hb_out}')
    print(f'  checkpoints under {out_dir / "harmbench_ckpt"}, keeping the '
          f'{args.harmbench_keep_ckpt if args.harmbench_keep_ckpt else "all"} most '
          f'recent once scored')
    return runner


def harvest(runner, log, is_main):
    """Log every job that finished since the last check (rank 0)."""
    if not (is_main and runner):
        return
    for rec in runner.poll():
        log(rec)


def submit_job(runner, log, step, model_path, ckpt_dir=None):
    """Submit one HarmBench run (rank 0). In sync mode `submit` has already
    waited and returns the finished record, which nothing else will harvest —
    so log it here. In async/queue mode it returns the live job and the record
    arrives later through `harvest`."""
    if runner is None:
        return
    rec = runner.submit(step, model_path, ckpt_dir=ckpt_dir)
    if isinstance(rec, dict) and rec.get('event') == 'harmbench':
        log(rec)


def add_harmbench_args(p):
    g = p.add_argument_group('HarmBench monitor')
    g.add_argument('--harmbench_script', type=str, default=None,
                   help='path to eval.sh taking MODEL_PATH CLS_PATH OUT_DIR MODEL_NAME; '
                        'required unless --no_harmbench')
    g.add_argument('--harmbench_cls_path', type=str, default=None,
                   help='local dir of the HarmBench classifier (judge_path); '
                        'required unless --no_harmbench')
    g.add_argument('--harmbench_out_dir', type=str, default=None,
                   help='HarmBench output dir (default <output_dir>/harmbench)')
    g.add_argument('--harmbench_name', type=str, default=None,
                   help='run label; each job is scored as <name>_stepNNNNNN '
                        '(default: basename of --output_dir)')
    g.add_argument('--harmbench_gpus', type=str, default='',
                   help='comma-separated physical GPU ids reserved for HarmBench, '
                        'e.g. "6,7"; must not overlap the training ranks')
    g.add_argument('--harmbench_allow_shared_gpus', action='store_true',
                   help='skip the overlap check against the training GPUs')
    g.add_argument('--harmbench_mode', choices=['async', 'sync', 'queue'],
                   default='async',
                   help='async: keep training and harvest later; sync: block until '
                        'each run returns; queue: only write queue.jsonl')
    g.add_argument('--harmbench_every', type=int, default=1,
                   help='run HarmBench every N validations (1 = every eval)')
    g.add_argument('--harmbench_max_inflight', type=int, default=1,
                   help='concurrent HarmBench jobs; rounds arriving above this are '
                        'skipped so jobs never contend for the same eval GPUs')
    g.add_argument('--harmbench_timeout', type=int, default=0,
                   help='seconds before a job is killed and recorded as timeout '
                        '(0 = no limit)')
    g.add_argument('--harmbench_keep_ckpt', type=int, default=2,
                   help='HarmBench checkpoints kept after scoring (0 = keep all); '
                        'only step_NNNNNN dirs this run wrote are ever removed')
    g.add_argument('--harmbench_env', action='append', default=[], metavar='KEY=VAL',
                   help='env passed to eval.sh; repeatable. Knobs it honours: TP, '
                        'MAX_BEHAVIORS, MAX_NEW_TOKENS, CATEGORIES, BEHAVIORS')
    g.add_argument('--harmbench_at_step0', action='store_true', default=True,
                   help='score --model_path itself as the step-0 baseline (default on)')
    g.add_argument('--no_harmbench_at_step0', dest='harmbench_at_step0',
                   action='store_false')
    g.add_argument('--harmbench_at_end', action='store_true', default=True,
                   help='also score the final checkpoint (default on)')
    g.add_argument('--no_harmbench_at_end', dest='harmbench_at_end',
                   action='store_false')
    g.add_argument('--harmbench_end_timeout', type=int, default=7200,
                   help='seconds to wait for outstanding jobs after the last step')
    g.add_argument('--no_harmbench', action='store_true',
                   help='disable the monitor and train exactly as reso_train')
    g.add_argument('--dist_timeout_min', type=int, default=120,
                   help='NCCL process-group timeout; must exceed the longest '
                        'collective, which in sync mode includes the HarmBench run')
    return p


def parse_args_with_harmbench(base_parse, arm):
    """HarmBench flags are consumed first, then the untouched arm parser handles
    the rest — the training surface stays identical to the arm by construction,
    and neither original parser has to be modified. Shared with the DPO arm."""
    hb = argparse.ArgumentParser(add_help=False,
                                 description=f'HarmBench monitor options '
                                             f'({arm} options follow below).')
    add_harmbench_args(hb)
    if any(a in ('-h', '--help') for a in sys.argv[1:]):
        hb.print_help()
        print()
    hb_args, rest = hb.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = base_parse()
    for k, v in vars(hb_args).items():
        setattr(args, k, v)
    return args


def parse_args():
    return parse_args_with_harmbench(base_parse_args, 'ReSO')


# ============================================================================
# Main — reso_train.main with the HarmBench hooks; the training path is byte-
# for-byte the same computation.
# ============================================================================

def main():
    args = parse_args()
    rank, world, local = setup_distributed(args.dist_timeout_min)
    if not dist_is_on():
        raise SystemExit('FSDP training must be launched with torchrun, e.g.\n'
                         '  torchrun --standalone --nproc_per_node=6 step_reso_train.py ...')
    device = torch.device(f'cuda:{local}')
    is_main = rank == 0

    if args.beta > 0 and args.replay_data is None:
        raise ValueError('--replay_data is required when beta > 0')

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
        print(f'{args.n_layers} layers, d={d_model}; trainable params: {n_tr / 1e9:.2f}B '
              f'(full-parameter, embeddings '
              f'{"trained" if args.train_embeddings else "frozen"})')

    layer_cls = decoder_layer_cls(model)
    model = wrap_fsdp(model, local, layer_cls,
                      activation_checkpointing=not args.no_activation_checkpointing)

    # frozen reference: separate bf16 replica, needed for L_pres and KL monitors
    ref_model = None
    if args.beta > 0 or args.replay_data:
        ref_model = load_hf_model(args.model_path, args.attn_impl,
                                  args.local_files_only, torch.bfloat16)
        ref_model.config.use_cache = False
        ref_model.eval()
        ref_model.requires_grad_(False)
        ref_model.to(device)

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
    # or triplet draws, which stay bit-identical to runs made before this monitor
    # existed. Same seed derivation in the DPO arm -> same items in both arms.
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
    model.train()

    def trainer_state(step):
        return dict(step=step, best_val_rsa=best_val_rsa, mu=mu.detach().cpu(),
                    layer_logits=(layer_logits.detach().cpu()
                                  if layer_logits is not None else None))

    ref_ppl_cache = {}
    evals_since_best = 0
    n_evals = 0
    flags = torch.zeros(3, device=device)  # [stop, save_best, harmbench]
    uniform_w = torch.full((len(inc_layers),), 1.0 / len(inc_layers), device=device)

    vm0 = run_validation(model, ref_model, tokenizer, val_bank, rsa_fx,
                         trip_fx, replay_val, mu, inc_t, inc_layers,
                         args, device, ref_ppl_cache, judge_fx)
    # Step-0 HarmBench scores --model_path directly: the untrained reference is
    # already on disk, so the baseline costs no checkpoint.
    if is_main and runner and args.harmbench_at_step0:
        submit_job(runner, log, 0, args.model_path)
    if is_main:
        vm0.update({'event': 'val', 'step': 0, 'val/is_best': False})
        if runner:
            vm0.update(runner.status_fields())
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
        # (a) representation pass
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
            loss_struct, trip_acc = structure_loss(Zn, trip, args.margin, args.tau,
                                                   layer_w, device)
        else:
            loss_struct, trip_acc = P.sum() * 0.0, None

        # (c) preservation pass
        loss_pres = torch.zeros((), device=device)
        if args.beta > 0:
            renc = replay.next_batch(args.replay_seqs, device)
            pol_logits = model(**renc, use_cache=False).logits
            with torch.no_grad():
                ref_logits = ref_model(**renc, use_cache=False).logits
            loss_pres = token_kl(pol_logits, ref_logits,
                                 renc['attention_mask'].bool())

        loss = loss_struct + args.beta * loss_pres

        # (d) update: FSDP reduces sharded grads; layer_logits synced manually
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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

        # ------------------------------------------------------ logging
        if is_main and (step % args.log_interval == 0 or step == 1):
            ev = {'event': 'train', 'step': step,
                  'loss': round(float(loss), 5),
                  'loss_struct': round(float(loss_struct), 5),
                  'loss_pres_kl': round(float(loss_pres), 6),
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
            n_evals += 1
            vm = run_validation(model, ref_model, tokenizer, val_bank, rsa_fx,
                                trip_fx, replay_val, mu, inc_t, inc_layers,
                                args, device, ref_ppl_cache, judge_fx)
            # Collect anything HarmBench finished since the last eval before the
            # val record is written, so it reports the freshest ASR available.
            harvest(runner, log, is_main)
            if is_main:
                vm.update({'event': 'val', 'step': step})
                # selection stays on val RSA: judgment accuracy and HarmBench ASR
                # are passive monitors and never enter the decision.
                improved = vm['val/rsa'] > best_val_rsa + args.min_delta
                if improved:
                    best_val_rsa = vm['val/rsa']
                    evals_since_best = 0
                    flags[1] = 1.0
                else:
                    evals_since_best += 1
                vm['val/best_rsa'] = best_val_rsa
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
                print(f"  eval {step}: RSA {vm['val/rsa']:.4f} "
                      f"(best {best_val_rsa:.4f}) | cross {vm['val/rsa_cross_domain']:.4f} "
                      f"| trip_acc {vm.get('val/triplet_acc', float('nan')):.4f} "
                      f"| judge {vm.get('val/judge_acc', float('nan')):.4f} "
                      f"| KL {vm.get('val/kl_ref_policy', float('nan')):.5f} "
                      f"| dppl {vm.get('val/replay_ppl_delta_pct', float('nan')):+.2f}%"
                      + (f" | {runner.summary_str()}" if runner else ''))
                if evals_since_best >= args.patience:
                    print(f'Early stop: no val RSA gain in {args.patience} evals')
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
        done = {'event': 'done', 'step': step, 'best_val_rsa': best_val_rsa}
        if runner and runner.last:
            done['harmbench/asr_last'] = runner.last['harmbench/asr']
            done['harmbench/asr_last_step'] = runner.last['step']
        log(done)
        metrics_f.close()
        print(f'Done. Best val RSA {best_val_rsa:.4f}. Outputs in {out_dir}')
        if runner:
            print(f'HarmBench: {runner.summary_str()} — full trace in '
                  f'{out_dir / "metrics.jsonl"} (event=harmbench) and '
                  f'{runner.out_dir / "harmbench_asr.csv"}')
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
