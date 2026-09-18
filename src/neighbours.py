"""Neighbouring-clip evidence (09-13). Recording sessions put the same activity script on
back-to-back clips (features/unit_times.csv: first/last Skeleton frame per unit). The option
texts of a clip's time neighbours -- its sequence options (known-present actions) and its
multi / single options -- name that script's actions. No answers are used, so the same
evidence exists for test clips.

Neighbours are chained by time gap only (< GAP seconds, user ignored), exactly as on test.
    nb_pool(df) -> {path: {normalised action text: score in [0, 1]}}
"""
import collections, os
import pandas as pd
from solver import opts, acts_of

GAP = 1800.0


def nz(x):
    return ' '.join(str(x).strip().lower().split())


def unit_of(path):
    q = str(path).replace(chr(92), '/')
    return '/'.join(q.split('/')[:2]) if q.startswith('large_model_track_test') else q


def nb_pool(df, gap=GAP, times='features/unit_times.csv'):
    ut = pd.read_csv(times, parse_dates=['t0', 't1']).set_index('unit')
    df = df[df.source == 'HAU']
    own, spans = {}, []
    for p, g in df.groupby('path'):
        u = unit_of(p)
        if u not in ut.index:
            continue
        c = collections.Counter()
        for _, r in g.iterrows():
            if r.category in ('sequence', 'multi', 'single'):
                for L in opts(r):
                    c[nz(r[L])] += 2 if r.category == 'sequence' else 1
        own[p] = c
        spans.append((ut.loc[u, 't0'], ut.loc[u, 't1'], p))
    spans.sort()
    nb = collections.defaultdict(list)
    for (a0, a1, a), (b0, b1, b) in zip(spans, spans[1:]):
        if (b0 - a1).total_seconds() < gap:
            nb[a].append(b); nb[b].append(a)
    out = {}
    for p in own:
        pool = collections.Counter()
        for q in nb[p]:
            pool.update(own[q])
        out[p] = {a: min(n, 4) / 4.0 for a, n in pool.items()}
    return out
