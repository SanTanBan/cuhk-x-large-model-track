"""Discriminative motion models.

Two places where cheap frame-difference features carry real signal:

  single|HARn  each clip is ONE action and the training path names it, so we get clean
               labels for a 44-way classifier; at question time we restrict its posterior
               to the 3 offered options.
  emotion|HAU  manner adverbs are largely about speed / burstiness of movement.

Both are evaluated subject-grouped, against the text-prior baseline they'd replace.
"""
import pandas as pd, numpy as np, collections, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts, lo
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATS = ['motion', 'motion_sd', 'motion_p90', 'busy', 'dur']


def motion_table():
    parts = [pd.read_csv('features/motion_train.csv')]
    if os.path.exists('features/motion_test.csv'):
        parts.append(pd.read_csv('features/motion_test.csv'))
    m = pd.concat(parts, ignore_index=True)
    for c in ['motion', 'motion_sd', 'motion_p90', 'dur']:
        m[c] = np.log1p(m[c])
    m['ratio'] = m.motion_p90 - m.motion          # burstiness, in log space
    cols = FEATS + ['ratio']
    return {r.path: np.array([getattr(r, c) for c in cols]) for r in m.itertuples(index=False)}, cols


def fit_clf(paths, labels, X, C=0.3):
    xs = np.array([X[p] for p in paths]); ys = np.array(labels)
    sc = StandardScaler().fit(xs)
    clf = LogisticRegression(max_iter=2000, C=C, multi_class='multinomial')
    clf.fit(sc.transform(xs), ys)
    return sc, clf


def logpost(sc, clf, x):
    lp = clf.predict_log_proba(sc.transform(x.reshape(1, -1)))[0]
    return {c: lp[i] for i, c in enumerate(clf.classes_)}


def run_harn(tr, X, Cs=(0.03, 0.1, 0.3, 1.0), ws=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)):
    h = tr[tr.qt == 'single|HARn'].copy()
    h['action'] = h.path.str.split('/').str[1]
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    print('--- single|HARn : 44-way motion classifier, restricted to the 3 offered options ---')
    for C in Cs:
        recs = []
        for f in folds:
            trn = h[~h.user.isin(f)]; val = h[h.user.isin(f)]
            trn = trn[trn.path.isin(X)]
            sc, clf = fit_clf(list(trn.path), list(trn.action), X, C)
            # map action folder -> the option text used for it (constant per action)
            a2t = {}
            for _, r in trn.iterrows():
                a2t[r.action] = r[str(r.answer)]
            for _, r in val.iterrows():
                if r.path not in X: continue
                lp = logpost(sc, clf, X[r.path])
                t2lp = {}
                for a, t in a2t.items():
                    t2lp[t] = max(t2lp.get(t, -1e9), lp.get(a, -1e9))
                O = opts(r)
                recs.append(([t2lp.get(str(r[L]).strip(), -1e9) for L in O], O.index(str(r.answer))))
        line = []
        for w in ws:
            ok = sum(1 for fs, ci in recs if max(range(len(fs)), key=lambda i: w * fs[i]) == ci)
            line.append('w=%.2f:%.4f' % (w, ok / len(recs)))
        print('  C=%.2f n=%d  ' % (C, len(recs)) + '  '.join(line))


def run_emotion(tr, X, Cs=(0.03, 0.1, 0.3, 1.0), ws=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0)):
    e = tr[tr.qt == 'emotion|HAU'].copy()
    e['lab'] = [r[str(r.answer)] for _, r in e.iterrows()]
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    print('--- emotion|HAU : adverb classifier on motion features ---')
    for C in Cs:
        recs = []
        for f in folds:
            trn = e[(~e.user.isin(f)) & (e.path.isin(X))]; val = e[e.user.isin(f)]
            keep = {k for k, v in collections.Counter(trn.lab).items() if v >= 5}
            trn2 = trn[trn.lab.isin(keep)]
            sc, clf = fit_clf(list(trn2.path), list(trn2.lab), X, C)
            app = collections.Counter(); cor = collections.Counter()
            for _, r in trn.iterrows():
                for L in opts(r):
                    app[r[L]] += 1
                    if L in str(r.answer): cor[r[L]] += 1
            base = sum(cor.values()) / max(1, sum(app.values()))
            pri = lambda o: lo((cor[o] + 6 * base) / (app[o] + 6))
            for _, r in val.iterrows():
                if r.path not in X: continue
                lp = logpost(sc, clf, X[r.path]); O = opts(r)
                recs.append(([(pri(r[L]), lp.get(str(r[L]).strip(), math.log(1e-4))) for L in O],
                             O.index(str(r.answer))))
        line = []
        for w in ws:
            ok = sum(1 for fs, ci in recs
                     if max(range(len(fs)), key=lambda i: fs[i][0] + w * fs[i][1]) == ci)
            line.append('w=%.2f:%.4f' % (w, ok / len(recs)))
        print('  C=%.2f n=%d  ' % (C, len(recs)) + '  '.join(line))


if __name__ == '__main__':
    tr, te = load()
    X, cols = motion_table()
    print('motion features:', cols, '| clips covered:', len(X), '\n')
    run_harn(tr, X)
    print()
    run_emotion(tr, X)
