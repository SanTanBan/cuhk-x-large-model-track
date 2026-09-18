"""Emotion, higher-order session decoding (09-14): fast head-to-head on held-out subjects.

Training adverb repeats by lag along gap-sessions (chance 0.043): lag 1 0.000, 2 0.011, 3 0.065, 4 0.007,
5 0.021, 6 0.096. sub14e's decoder (viterbi2 + neighbour-option term) uses only lag 1 (hard) and lag 2
(gamma2). The variants add lag terms whose weights come from the FIT half only:
g_L = lam * log(rate_L / chance), with rate_L smoothed toward chance.
  lagLR6       lags 2..6 all from the fit half's likelihood ratios (gamma2 dropped)
  order2+LR36  the baseline's fitted gamma2 at lag 2, plus lam * log LR at lags 3..6
Protocol, for each half:
  1. Tune sub14e's decoder on the other half over emo_fast_validate2.py's order2+nb grid, then score it on
     this half.
  2. Each variant keeps that fit's (w, tau, beta, gamma2), tunes only lam on the other half, and is scored on
     this half.
A variant is accepted only if it wins on BOTH halves; between accepted variants the larger minimum gain wins.
Nothing is refitted: the per-fold cache of src/emo_fast_validate2.py is reused. Decoding uses
src/emo_decode.py viterbiL, the same function the test-time builder calls.

    python src/emo_fast_validate3.py [--cache=PATH] [--beam=512]
"""
import sys, os, json, pickle, collections, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from emo_decode import emissions, viterbi2, viterbiL

A = sys.argv[1:]
CACHE = next((a.split('=', 1)[1] for a in A if a.startswith('--cache=')),
             os.path.join(os.environ.get('TEMP', '.'), 'emo_norm_rows.pkl'))
BEAM = int(next((a.split('=', 1)[1] for a in A if a.startswith('--beam=')), '512'))
rows, W0 = pickle.load(open(CACHE, 'rb'))
R = pd.DataFrame(rows)
WV = W0.get('w_vlm_emotion', 0.0)
print('loaded %d cached emotion rows from %s (w_vlm_emotion %s, beam %d)' % (len(R), CACHE, WV, BEAM), flush=True)


def chains(df):
    out, cur = [], []
    for r in df.sort_values('t0').itertuples():
        if pd.isna(r.t0):
            out.append([r]); continue
        if cur and (r.t0 - cur[-1].t1).total_seconds() >= 1800:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


users = sorted(R.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
CH = {'all': chains(R), 'h0': chains(R[R.user.isin(HALVES[0])]), 'h1': chains(R[R.user.isin(HALVES[1])])}
print('chains: %s' % {k: (len(v), max(len(c) for c in v)) for k, v in CH.items()}, flush=True)
EMC = {}


def ems(key, w, tau, beta):
    k = (key, w, tau, beta)
    if k not in EMC:
        EMC[k] = [emissions(ch, w, tau, beta, WV) for ch in CH[key]]
    return EMC[k]


def acc(key, w, tau, beta, g2, lagw=None, beam=BEAM):
    ok = n = 0
    for ch, em in zip(CH[key], ems(key, w, tau, beta)):
        path = viterbi2(ch, em, g2) if lagw is None else viterbiL(ch, em, lagw, beam)
        for r, k in zip(ch, path):
            n += 1; ok += r.O[k] == r.ans
    return ok / max(n, 1)


def lag_lr(key, lags=range(2, 7), alpha=20.0):
    ans = [[r.adv[list(r.O).index(r.ans)] if r.ans in list(r.O) else None for r in ch] for ch in CH[key]]
    cnt = collections.Counter(a for c in ans for a in c if a is not None)
    tot = sum(cnt.values())
    chance = sum((v / tot) ** 2 for v in cnt.values())
    table = {}
    for L in lags:
        same = pairs = 0
        for c in ans:
            for i in range(len(c) - L):
                if c[i] is not None and c[i + L] is not None:
                    pairs += 1; same += c[i] == c[i + L]
        rate = (same + alpha * chance) / (pairs + alpha)
        table[L] = dict(logLR=float(np.log(rate / chance)), same=same, pairs=pairs)
    return table, chance


VARIANTS = {
    'lagLR6': lambda T, g2, lam: {L: lam * T[L]['logLR'] for L in range(2, 7)},
    'order2+LR36': lambda T, g2, lam: dict([(2, g2)] + [(L, lam * T[L]['logLR']) for L in range(3, 7)]),
}
G14 = [(w, t, b, g) for w in (5, 8, 12) for t in (0.25, 0.5) for g in (-1.5, -4.0) for b in (0.1, 0.25)]
LAMS = (0.5, 1.0, 2.0, 3.0, 4.0)

agree = tot = 0
for ch, em in zip(CH['all'], ems('all', 5, 0.5, 0.25)):
    p2, pL = viterbi2(ch, em, -4.0), viterbiL(ch, em, {2: -4.0}, BEAM)
    agree += sum(a == b for a, b in zip(p2, pL)); tot += len(ch)
print('regression: viterbiL({2: -4}) vs viterbi2 agree on %d / %d clips' % (agree, tot), flush=True)
T_ALL, CHANCE_ALL = lag_lr('all')
print('lag table, all subjects (chance %.3f): %s' % (CHANCE_ALL, {L: (v['same'], v['pairs'], round(v['same'] / max(v['pairs'], 1), 3), round(v['logLR'], 2)) for L, v in T_ALL.items()}), flush=True)
print('sub14e decoder, all subjects: %.4f (w 5, tau 0.5, beta 0.25, gamma2 -4)' % acc('all', 5, 0.5, 0.25, -4.0), flush=True)

report = dict(lag_table_all=T_ALL, chance_all=CHANCE_ALL, beam=BEAM)
fits = {}
for i in (0, 1):
    fk, ek = 'h%d' % (1 - i), 'h%d' % i
    w, t, b, g = max(G14, key=lambda x: acc(fk, *x))
    base = acc(ek, w, t, b, g)
    T, _ = lag_lr(fk)
    fits[i] = (w, t, b, g, T, base)
    print('half %d: baseline fit (w %s, tau %s, beta %s, gamma2 %s) -> held-out %.4f | fit-half logLR %s'
          % (i, w, t, b, g, base, {L: round(v['logLR'], 2) for L, v in T.items()}), flush=True)
BASE_ALL = max(G14, key=lambda x: acc('all', *x))
print('baseline fit, all subjects: %s -> %.4f' % (BASE_ALL, acc('all', *BASE_ALL)), flush=True)

for name, mk in VARIANTS.items():
    gains, lams = [], []
    for i in (0, 1):
        w, t, b, g, T, base = fits[i]
        fk, ek = 'h%d' % (1 - i), 'h%d' % i
        lam = max(LAMS, key=lambda l: acc(fk, w, t, b, g, mk(T, g, l)))
        new = acc(ek, w, t, b, g, mk(T, g, lam))
        gains.append(new - base); lams.append(lam)
        print('   %-12s half %d: lam %s -> held-out %.4f (%+.4f)' % (name, i, lam, new, new - base), flush=True)
    ok = all(x > 1e-9 for x in gains)
    w, t, b, g = BASE_ALL
    lam = max(LAMS, key=lambda l: acc('all', w, t, b, g, mk(T_ALL, g, l)))
    lagw = mk(T_ALL, g, lam)
    a_all, a_big = acc('all', w, t, b, g, lagw), acc('all', w, t, b, g, lagw, beam=2048)
    report[name] = dict(held_out_gains=gains, lams=lams, accepted=ok, final=dict(w=w, tau=t, beta=b, gamma2=g, lam=lam),
                        lagw={str(k): v for k, v in lagw.items()}, acc_all=a_all, acc_all_beam2048=a_big)
    print('%-12s held-out %+.4f / %+.4f  %s | all subjects %.4f (beam 2048: %.4f) with lam %s, lagw %s'
          % (name, gains[0], gains[1], 'ACCEPT' if ok else 'reject', a_all, a_big, lam,
             {k: round(v, 2) for k, v in lagw.items()}), flush=True)
json.dump(report, open('src/emo_fast3_report.json', 'w'), indent=1, default=str)
print('wrote src/emo_fast3_report.json')
