#!/usr/bin/env python3
"""
Prepare replay.jsonl — the general replay corpus D_rep for the ReSO
preservation term (and the optional replay-perplexity monitor in the DPO arm).
=============================================================================

The preservation loss is a KL to the frozen reference's own next-token
predictions, so the corpus needs no labels; it only needs to *cover the
distribution where drift would hurt*: general web/encyclopedic text
(pretraining-like), chat-formatted instruction data (the behavioral evals run
in chat format), and both English and Chinese (the Flames behavioral eval is
Chinese). It should be disjoint from the Social-Chem items so L_pres and
L_struct never pull on the same inputs — general corpora satisfy this by
construction.

Output format (what reso_train.py / dpo_train.py consume):
  - JSONL, one doc per line: {"text": "..."} or {"messages": [{"role": ...,
    "content": ...}, ...]}
  - the file is globally shuffled; the first --replay_val_docs lines (64 by
    default in the trainers) become the held-out validation reserve
  - the trainers truncate each doc at --replay_len (1024) tokens, so docs
    beyond a few thousand characters only cost disk

Sizing: the ReSO defaults consume replay_seqs(2) x 8 ranks x num_steps(3000)
= 48k docs for a repetition-free run (the stream cycles if the file is
smaller, which is acceptable but reuses docs). Default here: 50k docs.

Two modes:

  # 1. stream public corpora from the HF hub (needs `datasets` + network)
  python prepare_replay.py --output replay.jsonl --n_docs 50000

  # 2. pack your own local files (offline clusters): .jsonl lines in the
  #    formats above, or .txt files (one doc per file)
  python prepare_replay.py --output replay.jsonl \
      --from_files 'corpus/*.jsonl' 'notes/*.txt'

Default hub mix (override with --mix, swap sources in SOURCES if one is
gated/renamed on the hub):
  web_en=0.50   HuggingFaceFW/fineweb (sample-10BT)      pretraining-like, en
  wiki_zh=0.15  wikimedia/wikipedia (20231101.zh)        pretraining-like, zh
  chat_en=0.25  HuggingFaceH4/ultrachat_200k (train_sft) instruction, en
  chat_zh=0.10  BelleGroup/train_1M_CN                   instruction, zh
"""

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

SOURCES = {
    'web_en': dict(path='HuggingFaceFW/fineweb', name='sample-10BT',
                   split='train', kind='text'),
    'wiki_zh': dict(path='wikimedia/wikipedia', name='20231101.zh',
                    split='train', kind='text'),
    'chat_en': dict(path='HuggingFaceH4/ultrachat_200k', name=None,
                    split='train_sft', kind='messages'),
    'chat_zh': dict(path='BelleGroup/train_1M_CN', name=None,
                    split='train', kind='belle'),
}


def parse_args():
    p = argparse.ArgumentParser(description='Build the replay corpus for '
                                            'reso_train.py / dpo_train.py.')
    p.add_argument('--output', type=str, default='replay.jsonl')
    p.add_argument('--n_docs', type=int, default=50000)
    p.add_argument('--mix', type=str,
                   default='web_en=0.5,wiki_zh=0.15,chat_en=0.25,chat_zh=0.1',
                   help='source=weight pairs; sources are keys of SOURCES')
    p.add_argument('--from_files', type=str, nargs='+', default=None,
                   help='offline mode: globs of local .jsonl/.txt files; '
                        'ignores --mix and the hub')
    p.add_argument('--min_chars', type=int, default=400,
                   help='drop tiny docs (too few KL positions to be useful)')
    p.add_argument('--max_chars', type=int, default=20000,
                   help='store at most this many chars per doc (trainers '
                        'truncate at ~1024 tokens anyway)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle_buffer', type=int, default=10000,
                   help='streaming shuffle buffer per hub source')
    return p.parse_args()


# ============================================================================
# Converters -> {"text": ...} or {"messages": [...]}, or None to drop
# ============================================================================

def _clip(t, max_chars):
    return t[:max_chars]


def convert_text(ex, min_chars, max_chars):
    t = (ex.get('text') or '').strip()
    if len(t) < min_chars:
        return None
    return {'text': _clip(t, max_chars)}


def convert_messages(ex, min_chars, max_chars):
    msgs = ex.get('messages') or []
    clean = []
    for m in msgs:
        role, content = m.get('role'), (m.get('content') or '').strip()
        if role not in ('system', 'user', 'assistant') or not content:
            return None
        clean.append({'role': role, 'content': _clip(content, max_chars)})
    if not clean or sum(len(m['content']) for m in clean) < min_chars:
        return None
    return {'messages': clean}


def convert_belle(ex, min_chars, max_chars):
    instr = (ex.get('instruction') or '').strip()
    extra = (ex.get('input') or '').strip()
    out = (ex.get('output') or '').strip()
    if not instr or not out:
        return None
    user = instr + ('\n' + extra if extra else '')
    if len(user) + len(out) < min_chars:
        return None
    return {'messages': [{'role': 'user', 'content': _clip(user, max_chars)},
                         {'role': 'assistant', 'content': _clip(out, max_chars)}]}


CONVERTERS = {'text': convert_text, 'messages': convert_messages,
              'belle': convert_belle}


def doc_key(doc):
    """Near-dup key: hash of the first 200 normalized chars."""
    if 'text' in doc:
        head = doc['text']
    else:
        head = ' '.join(m['content'] for m in doc['messages'])
    head = ' '.join(head.split())[:200].lower()
    return hashlib.md5(head.encode('utf-8')).hexdigest()


# ============================================================================
# Collection
# ============================================================================

def collect_from_hub(args):
    from datasets import load_dataset  # pip install datasets
    mix = {}
    for part in args.mix.split(','):
        k, v = part.split('=')
        mix[k.strip()] = float(v)
    total_w = sum(mix.values())
    quotas = {k: int(round(args.n_docs * w / total_w)) for k, w in mix.items()}

    docs, seen = [], set()
    for src_name, quota in quotas.items():
        if quota <= 0:
            continue
        spec = SOURCES[src_name]
        conv = CONVERTERS[spec['kind']]
        print(f'[{src_name}] streaming {spec["path"]}'
              f'{" / " + spec["name"] if spec["name"] else ""} '
              f'(quota {quota})...')
        try:
            ds = load_dataset(spec['path'], spec['name'], split=spec['split'],
                              streaming=True)
            ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
            taken = 0
            for ex in ds:
                doc = conv(ex, args.min_chars, args.max_chars)
                if doc is None:
                    continue
                key = doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                doc['_source'] = src_name
                docs.append(doc)
                taken += 1
                if taken >= quota:
                    break
            print(f'[{src_name}] collected {taken}')
        except Exception as e:
            print(f'[{src_name}] FAILED ({e}); continuing without it — '
                  f'swap the entry in SOURCES or adjust --mix')
    return docs


def collect_from_files(args):
    docs, seen = [], set()
    paths = []
    for pattern in args.from_files:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f'no files match {args.from_files}')
    for path in paths:
        p = Path(path)
        if p.suffix == '.txt':
            candidates = [{'text': p.read_text(errors='replace')}]
        else:
            candidates = []
            with open(p) as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
        for raw in candidates:
            if 'messages' in raw:
                doc = convert_messages(raw, args.min_chars, args.max_chars)
            else:
                doc = convert_text(raw, args.min_chars, args.max_chars)
            if doc is None:
                continue
            key = doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            doc['_source'] = p.name
            docs.append(doc)
        if len(docs) >= args.n_docs:
            break
    return docs[:args.n_docs] if len(docs) > args.n_docs else docs


def main():
    args = parse_args()
    docs = (collect_from_files(args) if args.from_files
            else collect_from_hub(args))
    if not docs:
        raise SystemExit('no documents collected')

    # Global shuffle so the first N lines (the trainers' validation reserve)
    # are a random draw over the whole mix.
    random.Random(args.seed).shuffle(docs)

    counts, n_chars = {}, 0
    with open(args.output, 'w') as f:
        for doc in docs:
            src = doc.pop('_source', '?')
            counts[src] = counts.get(src, 0) + 1
            n_chars += (len(doc['text']) if 'text' in doc else
                        sum(len(m['content']) for m in doc['messages']))
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f'\nWrote {len(docs)} docs to {args.output} '
          f'(mean {n_chars / len(docs):.0f} chars/doc)')
    for src, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {src:12s} {c}')
    print('\nReminder: the trainers reserve the first --replay_val_docs (64) '
          'lines for validation KL/perplexity;\nkeep this exact file fixed '
          'across the beta sweep and both arms so the guardrail metrics are '
          'comparable.')


if __name__ == '__main__':
    main()
