"""Action-presence model trained on the supervision that actually exists.

`HAUPresence` derives labels from the `combination` answer and calls everything else a
negative -- which is wrong, because an action absent from the (short) combination string
may still occur in the video.

The `multi` questions give the real thing: for each clip and each of four offered actions
we are told whether it occurs. That is ~3200 labelled (clip, action) pairs with *explicit,
trustworthy negatives*. Combining those with the positives from `combination` and
`sequence` gives a properly supervised presence model.

Evaluated by exact-set `multi` accuracy against the current pipeline.
"""
import numpy as np, pandas as pd, collections, itertools, sys, math
sys.path.insert(0, 'src')
from solver import load, opts, acts_of, lo
from joint import Joint, clip_features, sm
from motion_model import HAUPresence
from feats import task_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PresenceV2:
    def __init__(self, tr, X, C=1.0, min_each=8, source='all'):
        pos = collections.defaultdict(set)
        neg = collections.defaultdict(set)
        for path, g in tr[tr.source == 'HAU'].groupby('path'):
            if path not in X:
                continue
            for _, r in g[g.category == 'multi'].iterrows():
                ans = str(r.answer)
                for L in opts(r):
                    a = str(r[L]).strip()
                    (pos if L in ans else neg)[a].add(path)
            if source == 'all':
                cb = g[g.category == 'combination']
                if len(cb):
                    for a in acts_of(cb.iloc[0][str(cb.iloc[0].answer)]):
                        pos[a.strip()].add(path)
                sq = g[g.category == 'sequence']
                if len(sq):
                    for L in opts(sq.iloc[0]):
                        pos[str(sq.iloc[0][L]).strip()].add(path)
        self.X = X
        self.models, self.base = {}, {}
        allp = sorted({p for s in list(pos.values()) + list(neg.values()) for p in s})
        if not allp:
            self.sc = None
            return
        self.sc = StandardScaler().fit(np.array([X[p] for p in allp]))
        for a in sorted(set(pos) | set(neg)):
            P = pos[a]
            N = neg[a] - P            # a positive from any source overrides a multi negative
            if len(P) < min_each or len(N) < min_each:
                continue
            paths = sorted(P) + sorted(N)
            y = np.array([1] * len(P) + [0] * len(N))
            Z = self.sc.transform(np.array([X[p] for p in paths]))
            self.models[a] = LogisticRegression(max_iter=2000, C=C).fit(Z, y)
            self.base[a] = len(P) / (len(P) + len(N))

    def logodds(self, path):
        if self.sc is None or path not in self.X or not self.models:
            return {}
        z = self.sc.transform(self.X[path].reshape(1, -1))
        out = {}
        for a, clf in self.models.items():
            p = float(np.clip(clf.predict_proba(z)[0, 1], 1e-4, 1 - 1e-4))
            b = float(np.clip(self.base[a], 1e-4, 1 - 1e-4))
            out[a] = math.log(p / (1 - p)) - math.log(b / (1 - b))
        return out


if __name__ == '__main__':
    tr, te = load()
    X = task_features()['X']
    users = sorted(tr.user.unique())
    folds = [users[i::6] for i in range(6)]
    REC = []
    for f in folds:
        trn = tr[~tr.user.isin(f)]
        M = Joint(trn)
        HPs = {'v1 (combination-derived)': HAUPresence(trn, X),
               'v2 (multi-supervised)': PresenceV2(trn, X, source='multi'),
               'v2 (multi+comb+seq)': PresenceV2(trn, X, source='all')}
        print('  fold: v2 covers %d actions (v1 %d)'
              % (len(HPs['v2 (multi+comb+seq)'].models), len(HPs['v1 (combination-derived)'].models)),
              flush=True)
        for p, g in tr[tr.user.isin(f)].groupby('path'):
            mq = g[g.category == 'multi']
            if len(mq) == 0:
                continue
            K, comb = clip_features(g, M)
            bel = {}
            if comb:
                sc = [comb['marg'][i] + 3.0 * comb['pmi'][i] + 25.0 * comb['ovl'][i]
                      for i in range(len(comb['O']))]
                pr = sm([0.5 * x for x in sc])
                for i, S in enumerate(comb['sets']):
                    for a in S:
                        bel[a] = bel.get(a, 0.0) + pr[i]
            for a in K:
                bel[a] = 1.0
            r = mq.iloc[0]
            O = opts(r)
            base = [lo(M.p('multi|HAU', r[L]))
                    + lo(min(max(bel.get(str(r[L]).strip(), 0.0), 1e-4), .9999)) for L in O]
            pres = {n: [HP.logodds(p).get(str(r[L]).strip(), 0.0) for L in O]
                    for n, HP in HPs.items()}
            REC.append((base, pres, O, str(r.answer)))
    print('\ncached %d multi questions\n' % len(REC))
    for name in ['none'] + list(REC[0][1]):
        for w in ([0.0] if name == 'none' else [0.25, 0.5, 1.0, 2.0, 3.0]):
            best = 0; bthr = None
            for thr in (-3, -2.5, -2, -1.5, -1, -0.5):
                ok = 0
                for base, pres, O, ans in REC:
                    sc = base if name == 'none' else [base[i] + w * pres[name][i]
                                                      for i in range(len(base))]
                    sel = [O[i] for i in range(len(O)) if sc[i] > thr] or [O[int(np.argmax(sc))]]
                    ok += (''.join(sorted(sel)) == ans)
                if ok / len(REC) > best:
                    best, bthr = ok / len(REC), thr
            print('  %-28s w=%.2f  multi exact=%.4f (thr=%.1f)' % (name, w, best, bthr))
