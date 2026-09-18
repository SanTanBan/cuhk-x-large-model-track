"""Predict the `emotion` (manner-adverb) answer from motion features.

For each adverb we fit a shrunken diagonal Gaussian over the clip's motion features using
the training clips where that adverb is the correct answer, then score the four offered
adverbs by  log P(adverb) + w * log N(x | adverb).  Evaluated cross-subject.
"""
import pandas as pd, numpy as np, collections, math, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts, lo

FEATS = ['motion', 'motion_sd', 'motion_p90', 'busy', 'dur']


def prep():
    tr, te = load()
    parts = [pd.read_csv('features/motion_train.csv')]
    if os.path.exists('features/motion_test.csv'):
        parts.append(pd.read_csv('features/motion_test.csv'))
    m = pd.concat(parts, ignore_index=True)
    for c in ['motion', 'motion_sd', 'motion_p90', 'dur']:
        m[c] = np.log1p(m[c])
    mu, sd = m[FEATS].mean(), m[FEATS].std().replace(0, 1)
    m[FEATS] = (m[FEATS] - mu) / sd
    X = {r.path: np.array([getattr(r, f) for f in FEATS]) for r in m.itertuples(index=False)}
    return tr, te, X


class AdverbGauss:
    """Diagonal Gaussian per adverb, shrunk toward the global feature distribution."""
    def __init__(self, df, X, shrink=8.0, min_n=5):
        self.shrink = shrink
        pos = collections.defaultdict(list)
        app = collections.Counter(); cor = collections.Counter()
        for _, r in df[df.qt == 'emotion|HAU'].iterrows():
            if r.path not in X:
                continue
            for L in opts(r):
                app[r[L]] += 1
                if L in str(r.answer):
                    cor[r[L]] += 1
                    pos[r[L]].append(X[r.path])
        allx = np.array([X[p] for p in df.path.unique() if p in X])
        self.g_mu = allx.mean(0) if len(allx) else np.zeros(len(FEATS))
        self.g_var = allx.var(0) + 1e-3 if len(allx) else np.ones(len(FEATS))
        self.par = {}
        for a, v in pos.items():
            if len(v) < min_n:
                continue
            v = np.array(v); n = len(v)
            w = n / (n + shrink)
            mu = w * v.mean(0) + (1 - w) * self.g_mu
            var = w * (v.var(0) + 1e-3) + (1 - w) * self.g_var
            self.par[a] = (mu, var)
        base = sum(cor.values()) / max(1, sum(app.values())) or .25
        self.app, self.cor, self.base = app, cor, base

    def prior(self, a, alpha=6.0):
        return (self.cor[a] + alpha * self.base) / (self.app[a] + alpha)

    def loglik(self, a, x):
        if a not in self.par or x is None:
            return 0.0
        mu, var = self.par[a]
        return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (x - mu) ** 2 / var))

    def gll(self, x):
        if x is None:
            return 0.0
        return float(-0.5 * np.sum(np.log(2 * np.pi * self.g_var) + (x - self.g_mu) ** 2 / self.g_var))


def cv(w_grid=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), shrink=8.0):
    tr, te, X = prep()
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    recs = []
    for f in folds:
        M = AdverbGauss(tr[~tr.user.isin(f)], X, shrink)
        val = tr[(tr.user.isin(f)) & (tr.qt == 'emotion|HAU')]
        for _, r in val.iterrows():
            O = opts(r); x = X.get(r.path)
            recs.append(([(lo(M.prior(r[L])), M.loglik(r[L], x) - M.gll(x)) for L in O],
                         O.index(str(r.answer))))
    out = {}
    for w in w_grid:
        ok = sum(1 for fs, ci in recs
                 if max(range(len(fs)), key=lambda i: fs[i][0] + w * fs[i][1]) == ci)
        out[w] = ok / len(recs)
    return out, len(recs)


if __name__ == '__main__':
    for shrink in (4.0, 8.0, 16.0, 32.0):
        res, n = cv(shrink=shrink)
        best = max(res, key=res.get)
        print('shrink=%5.1f  n=%d  ' % (shrink, n)
              + '  '.join('w=%.2f:%.4f' % (k, v) for k, v in res.items())
              + '   BEST w=%.2f -> %.4f' % (best, res[best]))
