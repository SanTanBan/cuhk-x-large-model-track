"""`multi` is scored by exact set match, so it should be decoded as a set, not as four
independent yes/no decisions.

Training answer-set sizes are very skewed -- 1:312, 2:298, 3:197, 4:2 -- so a size prior
is real information that independent thresholding throws away. This enumerates all 15
non-empty subsets and scores

    sum_{L in S} log p_L  +  sum_{L not in S} log(1 - p_L)  +  lam * log P(|S|)

against the current thresholding rule, under the same subject-grouped protocol.
"""
import numpy as np, pandas as pd, collections, itertools, sys, math
sys.path.insert(0, 'src')
from solver import load, opts, lo
from joint import Joint, clip_features
from motion_model import HAUPresence
from feats import task_features

tr, te = load()
TF = task_features()
X, SEG = TF['X'], TF['SEG']
users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]

W_BEL_M, THR = 1.0, -1.5
REC = []
for f in folds:
    trn = tr[~tr.user.isin(f)]
    M = Joint(trn)
    HP = HAUPresence(trn, X)
    sizes = collections.Counter(
        len(str(r.answer)) for _, r in trn[trn.category == 'multi'].iterrows())
    tot = sum(sizes.values())
    logp_size = {k: math.log((sizes.get(k, 0) + 1) / (tot + 4)) for k in (1, 2, 3, 4)}
    for p, g in tr[tr.user.isin(f)].groupby('path'):
        mq = g[g.category == 'multi']
        if len(mq) == 0:
            continue
        K, comb = clip_features(g, M)
        bel = {}
        if comb:
            from joint import sm
            sc = [comb['marg'][i] + 3.0 * comb['pmi'][i] + 25.0 * comb['ovl'][i]
                  for i in range(len(comb['O']))]
            pr = sm([0.5 * x for x in sc])
            for i, S in enumerate(comb['sets']):
                for a in S:
                    bel[a] = bel.get(a, 0.0) + pr[i]
        for a in K:
            bel[a] = 1.0
        mp = HP.logodds(p)
        r = mq.iloc[0]
        O = opts(r)
        sc = []
        for L in O:
            t = str(r[L]).strip()
            b = lo(min(max(bel.get(t, 0.0), 1e-4), .9999)) + 0.25 * mp.get(t, 0.0)
            sc.append(lo(M.p('multi|HAU', r[L])) + W_BEL_M * b)
        REC.append((sc, O, str(r.answer), logp_size))
print('cached %d multi questions\n' % len(REC))


def thresholded(sc, O, thr):
    sel = [O[i] for i in range(len(O)) if sc[i] > thr]
    if not sel:
        sel = [O[int(np.argmax(sc))]]
    return ''.join(sorted(sel))


def set_decode(sc, O, logp_size, lam, temp):
    p = 1 / (1 + np.exp(-np.array(sc) / temp))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    best, bs = None, -1e18
    for k in range(1, len(O) + 1):
        for S in itertools.combinations(range(len(O)), k):
            s = sum(math.log(p[i]) if i in S else math.log(1 - p[i]) for i in range(len(O)))
            s += lam * logp_size.get(k, -9)
            if s > bs:
                bs, best = s, S
    return ''.join(sorted(O[i] for i in best))


print('threshold rule:')
for thr in (-2, -1.5, -1, -0.5, 0):
    acc = np.mean([thresholded(sc, O, thr) == a for sc, O, a, _ in REC])
    print('   thr=%5.1f  acc=%.4f' % (thr, acc))
print('\nset decode with size prior:')
for temp in (1.0, 2.0, 3.0):
    for lam in (0.0, 0.5, 1.0, 2.0, 3.0):
        acc = np.mean([set_decode(sc, O, ls, lam, temp) == a for sc, O, a, ls in REC])
        print('   temp=%.1f lam=%.1f  acc=%.4f' % (temp, lam, acc))
