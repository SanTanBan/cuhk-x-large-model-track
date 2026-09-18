"""Better `emotion` model.

Two problems with the plain 58-way classifier:
  * too few features  -> add the per-segment descriptors (how the motion evolves), not just
                         clip-level aggregates
  * label fragmentation -> Quickly / Rapidly / Swiftly / Hastily / Hurriedly are near
                         synonyms competing for the same probability mass. Grouping them
                         and scoring  log P(group) + log prior(adverb | group)  spends the
                         model's capacity on the distinction that is actually learnable.
"""
import pandas as pd, numpy as np, collections, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts, lo
from motion_model import motion_table
from seq_model import load_segments
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def features():
    X, cols = motion_table('motion2')
    S = load_segments()
    out = {}
    for p, v in X.items():
        seg = S.get(p)
        out[p] = np.concatenate([v, seg.flatten()]) if seg is not None else \
                 np.concatenate([v, np.zeros(44)])
    return out


def evaluate(tr, F, K=0, C=1.0, ws=(0.0, .25, .5, 1., 2., 3.), min_n=5):
    """K=0 -> plain per-adverb classifier; K>0 -> cluster adverbs into K groups first."""
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    recs = []
    for f in folds:
        trn = tr[(~tr.user.isin(f)) & (tr.qt == 'emotion|HAU') & (tr.path.isin(F))]
        val = tr[(tr.user.isin(f)) & (tr.qt == 'emotion|HAU') & (tr.path.isin(F))]
        lab = {r.path: str(r[str(r.answer)]).strip() for _, r in trn.iterrows()}
        cnt = collections.Counter(lab.values())
        keep = {k for k, v in cnt.items() if v >= min_n}
        paths = [p for p in lab if lab[p] in keep]
        if len(paths) < 50:
            continue
        Xm = np.array([F[p] for p in paths]); y = np.array([lab[p] for p in paths])
        sc = StandardScaler().fit(Xm); Z = sc.transform(Xm)

        if K > 0:
            advs = sorted(set(y))
            cent = np.array([Z[y == a].mean(0) for a in advs])
            km = KMeans(n_clusters=min(K, len(advs)), n_init=10, random_state=0).fit(cent)
            a2g = {a: int(g) for a, g in zip(advs, km.labels_)}
            yg = np.array([a2g[v] for v in y])
            clf = LogisticRegression(max_iter=3000, C=C).fit(Z, yg)
            gidx = {g: i for i, g in enumerate(clf.classes_)}
        else:
            clf = LogisticRegression(max_iter=3000, C=C).fit(Z, y)
            gidx = {c: i for i, c in enumerate(clf.classes_)}

        app = collections.Counter(); cor = collections.Counter()
        for _, r in trn.iterrows():
            for L in opts(r):
                app[r[L]] += 1
                if L in str(r.answer): cor[r[L]] += 1
        base = sum(cor.values()) / max(1, sum(app.values()))
        pri = lambda o: lo((cor[o] + 6 * base) / (app[o] + 6))

        for _, r in val.iterrows():
            lp = clf.predict_log_proba(sc.transform(F[r.path].reshape(1, -1)))[0]
            O = opts(r); fs = []
            for L in O:
                t = str(r[L]).strip()
                if K > 0:
                    g = a2g.get(t)
                    v = lp[gidx[g]] if (g is not None and g in gidx) else math.log(1e-3)
                else:
                    v = lp[gidx[t]] if t in gidx else math.log(1e-3)
                fs.append((pri(t), v))
            recs.append((fs, O.index(str(r.answer))))
    out = {}
    for w in ws:
        ok = sum(1 for fs, ci in recs
                 if max(range(len(fs)), key=lambda i: fs[i][0] + w * fs[i][1]) == ci)
        out[w] = ok / max(1, len(recs))
    return out, len(recs)


if __name__ == '__main__':
    tr, te = load()
    F = features()
    print('features: %d clips x %d dims\n' % (len(F), len(next(iter(F.values())))))
    for K in (0, 4, 6, 8, 12, 16):
        for C in (0.3, 1.0):
            res, n = evaluate(tr, F, K=K, C=C)
            best = max(res, key=res.get)
            tag = 'plain ' if K == 0 else 'K=%-3d' % K
            print('  %s C=%.1f  n=%d  ' % (tag, C, n)
                  + '  '.join('w=%.2f:%.4f' % (k, v) for k, v in res.items())
                  + '   BEST %.4f' % res[best])
