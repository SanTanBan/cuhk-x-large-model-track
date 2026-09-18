"""Sequence pooling across consecutive same-script clips, head-to-head in the pipeline (09-13).

Consecutive clips (gap < 30 min, users ignored) whose sequence questions offer the same four
actions have the IDENTICAL true order 74 / 74 times in training (features/unit_times.csv). Within
such a run the order is decoded once, from the summed slot-model evidence (times alpha) plus the
precedence prior, and every clip in the run gets that order mapped to its own option letters.
alpha is tuned on one half of the subjects and scored on the other; accepted only if pooling
beats the independent per-clip answer (= sub12's) on BOTH halves. Chaining uses time gaps only,
exactly as on test. Nothing shared is edited.

    python src/seq_pool_validate.py [--base=src/best_nv_final_W.json] [--dry-run]
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]
BASE_F = next((a.split('=', 1)[1] for a in ARGS if a.startswith('--base=')), 'src/best_nv_final_W.json')
DRY = '--dry-run' in ARGS
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

import numpy as np, pandas as pd
from solver import opts

W = dict(DEFAULT_W); W.update(json.load(open(BASE_F)))
WS, WP = W.get('w_seq', 1.0), W.get('w_prec', 0.0)
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')
norm = lambda v: ' '.join(str(v).strip().lower().split())
rows, mism, nchk = [], 0, 0
for (M_, MM_, HP_, SQ_, clips), f in zip(FOLD, folds):
    for g, feats_, hs, e_ in clips:
        sq = g[g.category == 'sequence']
        if not len(sq):
            continue
        a = answer_clip_fused(g, M_, W, VLM, feats_, MM_, HP_, SQ_, e_)
        for _, r in sq.iterrows():
            O = opts(r); txt = {L: norm(r[L]) for L in O}
            seq, prec = e_['seq'].get(r.qa_id) or {}, e_.get('prec', {}).get(r.qa_id) or {}
            slot = {tuple(txt[L] for L in p): v for p, v in seq.items()}
            pr = {tuple(txt[L] for L in p): v for p, v in prec.items()}
            perms = [''.join(p) for p in itertools.permutations(O)]
            tot = [(WS * seq.get(p, 0.0) if seq else 0.0) + (WP * prec.get(p, 0.0) if (prec and WP) else 0.0) for p in perms]
            ind = perms[int(np.argmax(tot))] if (seq or prec) else a[r.qa_id]
            nchk += 1; mism += ind != a[r.qa_id]
            rows.append(dict(user=r.user, path=g.iloc[0].path, t0=U.t0.get(g.iloc[0].path), t1=U.t1.get(g.iloc[0].path),
                             set=frozenset(txt.values()), txt=txt, slot=slot, pr=pr, ind=ind, ans=str(r.answer)))
print('regression: rebuilt independent answer vs pipeline mismatches %d / %d' % (mism, nchk))
R = pd.DataFrame(rows).dropna(subset=['t0']).sort_values('t0').reset_index(drop=True)
print('validation sequence questions with recording times: %d' % len(R))


def runs(sub):
    out, cur = [], []
    for r in sub.sort_values('t0').itertuples():
        if cur and ((r.t0 - cur[-1].t1).total_seconds() >= 1800 or r.set != cur[-1].set):
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def acc(sub, alpha):
    ok = n = 0
    for run in runs(sub):
        if alpha is None or len(run) == 1:
            pred = [r.ind for r in run]
        else:
            orders = list(run[0].pr) or list(run[0].slot)
            if not orders:
                pred = [r.ind for r in run]
            else:
                score = {o: WP * run[0].pr.get(o, 0.0) + alpha * sum(WS * r.slot.get(o, 0.0) for r in run) for o in orders}
                best = max(orders, key=lambda o: score[o])
                inv = [{t: L for L, t in r.txt.items()} for r in run]
                pred = [''.join(iv[t] for t in best) for iv in inv]
        for r, p in zip(run, pred):
            n += 1; ok += p == r.ans
    return ok / max(n, 1), n


users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
ALPHAS = [0.5, 1, 2]
gains, fits = [], []
for i in (0, 1):
    fit, ev = R[R.user.isin(HALVES[1 - i])], R[R.user.isin(HALVES[i])]
    al = max(ALPHAS, key=lambda x: acc(fit, x)[0]); fits.append(al)
    a0, n = acc(ev, None); a1, _ = acc(ev, al); gains.append(a1 - a0)
    print('held-out half %d: sequence exact independent %.4f -> pooled %.4f (%+.4f, n=%d, alpha %s)' % (i, a0, a1, a1 - a0, n, al))
al_all = max(ALPHAS, key=lambda x: acc(R, x)[0])
a0, n = acc(R, None); a1, _ = acc(R, al_all)
ok = all(x > 1e-9 for x in gains)
w_s = sum(v for k, v in TESTC.items() if k[0] == 'sequence|HAU') / sum(TESTC.values())
print('all subjects: %.4f -> %.4f (alpha %s) | overall test-weighted %+.4f | verdict %s' % (a0, a1, al_all, (a1 - a0) * w_s, 'ACCEPT' if ok else 'reject'))
print('same-set runs among validation clips: %s' % dict(sorted(collections.Counter(len(x) for x in runs(R)).items())))
if not DRY:
    json.dump(dict(held_out_gains=gains, alphas=fits, alpha_all=al_all, acc_all=[a0, a1], accepted=ok, base=BASE_F,
                   regression_mismatch=[mism, nchk]), open('src/seq_pool_report.json', 'w'), indent=1)
    print('wrote src/seq_pool_report.json')
