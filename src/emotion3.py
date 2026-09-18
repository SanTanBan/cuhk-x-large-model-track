"""`emotion` from motion + the clip's action set.

Manner adverbs are weakly action-dependent (wiping tends to be "Meticulously", phone use
"Hurriedly"; measured KL ~0.49 over a high-entropy label distribution). On its own that is
too weak, but the pipeline now infers the action set at ~0.87 accuracy, so it can be fed
to the emotion classifier as features alongside the motion descriptors.

Uses the *inferred* set (not the true one) so the number is honest.
"""
import numpy as np, pandas as pd, collections, sys, math
sys.path.insert(0, 'src')
from solver import load, opts, lo
from joint import Joint, clip_features, sm
from feats import task_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

tr, te = load()
TF = task_features()
Xmot = TF['X_emo']

# global action vocabulary for the indicator block
VOCAB = sorted({a.strip() for _, r in tr[tr.category == 'combination'].iterrows()
                for L in opts(r) for a in str(r[L]).split(',')})
VIDX = {a: i for i, a in enumerate(VOCAB)}
print('action vocabulary: %d' % len(VOCAB))


def belief_vector(g, M):
    """Soft action-set indicator from the combination question + sequence leak."""
    K, comb = clip_features(g, M)
    v = np.zeros(len(VOCAB))
    if comb:
        sc = [comb['marg'][i] + 3.0 * comb['pmi'][i] + 25.0 * comb['ovl'][i]
              for i in range(len(comb['O']))]
        pr = sm([0.5 * x for x in sc])
        for i, S in enumerate(comb['sets']):
            for a in S:
                if a in VIDX:
                    v[VIDX[a]] += pr[i]
    for a in K:
        if a in VIDX:
            v[VIDX[a]] = 1.0
    return v


users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]
REC = []
for f in folds:
    trn = tr[~tr.user.isin(f)]
    M = Joint(trn)
    # build training rows
    rows, labs = [], []
    for p, g in trn[trn.source == 'HAU'].groupby('path'):
        eq = g[g.category == 'emotion']
        if len(eq) == 0 or p not in Xmot:
            continue
        rows.append((Xmot[p], belief_vector(g, M)))
        labs.append(str(eq.iloc[0][str(eq.iloc[0].answer)]).strip())
    cnt = collections.Counter(labs)
    keep = {k for k, v in cnt.items() if v >= 5}
    idx = [i for i, l in enumerate(labs) if l in keep]
    app = collections.Counter(); cor = collections.Counter()
    for _, r in trn[trn.qt == 'emotion|HAU'].iterrows():
        for L in opts(r):
            app[r[L]] += 1
            if L in str(r.answer): cor[r[L]] += 1
    base = sum(cor.values()) / max(1, sum(app.values()))
    pri = lambda o: lo((cor[o] + 6 * base) / (app[o] + 6))

    FS = {'motion': lambda m, b: m,
          'motion+actions': lambda m, b: np.concatenate([m, b]),
          'actions': lambda m, b: b}
    clfs = {}
    for name, fn in FS.items():
        Z = np.array([fn(*rows[i]) for i in idx])
        sc = StandardScaler().fit(Z)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Z),
                                                           np.array([labs[i] for i in idx]))
        clfs[name] = (sc, clf, {c: j for j, c in enumerate(clf.classes_)})

    for p, g in tr[tr.user.isin(f)].groupby('path'):
        eq = g[g.category == 'emotion']
        if len(eq) == 0 or p not in Xmot:
            continue
        m, b = Xmot[p], belief_vector(g, M)
        r = eq.iloc[0]; O = opts(r)
        lps = {}
        for name, (sc, clf, ci) in clfs.items():
            lp = clf.predict_log_proba(sc.transform(FS[name](m, b).reshape(1, -1)))[0]
            lps[name] = [lp[ci[str(r[L]).strip()]] if str(r[L]).strip() in ci else math.log(1e-3)
                         for L in O]
        REC.append(([pri(str(r[L]).strip()) for L in O], lps, O.index(str(r.answer))))
print('cached %d emotion questions\n' % len(REC))

for name in ['prior only'] + list(REC[0][1]):
    for w in ([0.0] if name == 'prior only' else [0.25, 0.5, 1.0, 2.0, 3.0]):
        ok = 0
        for pri_s, lps, ci in REC:
            sc = pri_s if name == 'prior only' else [pri_s[i] + w * lps[name][i]
                                                     for i in range(len(pri_s))]
            ok += (int(np.argmax(sc)) == ci)
        print('  %-18s w=%.2f  emotion=%.4f' % (name, w, ok / len(REC)))
