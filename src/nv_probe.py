"""Standalone cross-subject probe of the non-visual features (09-11): which feature set picks
the right option best, by itself, for `emotion|HAU`, `single|HARn` and
`object_interaction|HARn`? Same 6 subject folds as cv_full. Each validation question is
answered by the classifier's log-probability of its options; questions whose options the
classifier has never seen count as misses. This only ranks evidence sources -- the chosen
one still goes through the fusion and the half-split validation before it can ship.

    python src/nv_probe.py
"""
import sys, collections, warnings
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from solver import load
from feats import task_features

tr, te = load()
TF = task_features('motion+clip')
MOTION, CLIP = TF['X_emo'], TF['X_harn']


def table(f):
    d = pd.read_csv(f, index_col=0)
    d = d.loc[:, d.notna().any()]
    d = d.fillna(d.median())
    return {k: v for k, v in zip(d.index, d.to_numpy(float))}


IMU, SKEL = table('features/imu_feats.csv'), table('features/skel_feats.csv')


def cat(*ds):
    keys = set(ds[0]).intersection(*ds[1:])
    return {k: np.concatenate([d[k] for d in ds]) for k in keys}


SETS = {'motion': MOTION, 'clip': CLIP, 'imu': IMU, 'skel': SKEL, 'imu+motion': cat(IMU, MOTION),
        'imu+skel': cat(IMU, SKEL), 'clip+skel': cat(CLIP, SKEL), 'skel+imu': cat(SKEL, IMU)}
users = sorted(tr.user.unique())
FOLDS = [users[i::6] for i in range(6)]


def fit(xs, ys, kind):
    if kind == 'rf':
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
        clf.fit(xs, ys)
        return None, clf
    sc = StandardScaler().fit(xs)
    clf = LogisticRegression(max_iter=3000, C=float(kind[2:]))
    clf.fit(sc.transform(xs), ys)
    return sc, clf


def logp(sc, clf, x):
    p = clf.predict_proba((sc.transform(x[None]) if sc is not None else x[None]))[0]
    return {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}


def probe_emotion(X, kind):
    e = tr[tr.qt == 'emotion|HAU'].copy()
    e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
    ok = n = cov = 0
    for f in FOLDS:
        trn = e[~e.user.isin(f) & e.path.isin(X)]
        keep = {k for k, v in collections.Counter(trn.lab).items() if v >= 5}
        trn = trn[trn.lab.isin(keep)]
        sc, clf = fit(np.array([X[p] for p in trn.path]), np.array(trn.lab), kind)
        for _, r in e[e.user.isin(f)].iterrows():
            n += 1
            if r.path not in X:
                continue
            lp = logp(sc, clf, X[r.path])
            sc_o = {L: lp[str(r[L]).strip()] for L in 'ABCD' if str(r[L]).strip() in lp}
            if sc_o:
                cov += 1
                ok += max(sc_o, key=sc_o.get) == str(r.answer)
    return ok / n, cov / n


def probe_harn(X, kind):
    h = tr[tr.qt == 'single|HARn'].copy(); h['action'] = h.path.str.split('/').str[1]
    oi = tr[tr.qt == 'object_interaction|HARn'].copy(); oi['action'] = oi.path.str.split('/').str[1]
    res = collections.Counter()
    for f in FOLDS:
        trn = h[~h.user.isin(f) & h.path.isin(X)]
        act2txt = {r.action: str(r[str(r.answer)]).strip() for _, r in trn.iterrows()}
        sc, clf = fit(np.array([X[p] for p in trn.path]), np.array(trn.action), kind)
        o2 = collections.defaultdict(collections.Counter)
        for _, r in oi[~oi.user.isin(f)].iterrows():
            o2[r.action][str(r[str(r.answer)]).strip()] += 1
        for _, r in h[h.user.isin(f)].iterrows():
            res['h_n'] += 1
            if r.path not in X:
                continue
            lp = logp(sc, clf, X[r.path]); t = {}
            for a, v in lp.items():
                if a in act2txt:
                    t[act2txt[a]] = max(t.get(act2txt[a], -1e9), v)
            s = {L: t[str(r[L]).strip()] for L in 'ABCD' if str(r[L]).strip() in t}
            if s:
                res['h_ok'] += max(s, key=s.get) == str(r.answer)
        for _, r in oi[oi.user.isin(f)].iterrows():
            res['o_n'] += 1
            if r.path not in X:
                continue
            lp = logp(sc, clf, X[r.path]); acc = collections.defaultdict(float)
            for a, v in lp.items():
                c = o2.get(a)
                if c:
                    tot = sum(c.values())
                    for o, k in c.items():
                        acc[o] += np.exp(v) * k / tot
            s = {L: acc[str(r[L]).strip()] for L in 'ABCD' if str(r[L]).strip() in acc}
            if s:
                res['o_ok'] += max(s, key=s.get) == str(r.answer)
    return res['h_ok'] / res['h_n'], res['o_ok'] / res['o_n']


if __name__ == '__main__':
    print('feature sets (clips covered): ' + ', '.join('%s %d' % (k, len(v)) for k, v in SETS.items()))
    print('\nemotion|HAU (n=809): accuracy / coverage   [current model: motion + lr1.0]')
    for name in ('motion', 'imu', 'imu+motion', 'imu+skel', 'skel'):
        for kind in ('lr0.1', 'lr1.0', 'rf'):
            a, c = probe_emotion(SETS[name], kind)
            print('   %-11s %-6s acc %.3f  cov %.2f' % (name, kind, a, c), flush=True)
    print('\nsingle|HARn (n=429) and object_interaction|HARn (n=133), object via P(object|action)'
          '   [current model: clip + lr1.0]')
    for name in ('clip', 'skel', 'clip+skel', 'skel+imu'):
        for kind in ('lr0.1', 'lr1.0', 'rf'):
            a, o = probe_harn(SETS[name], kind)
            print('   %-11s %-6s single %.3f  object %.3f' % (name, kind, a, o), flush=True)
