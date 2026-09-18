"""Emotion session probe (09-13), standalone and cross-subject (same 6 subject folds as cv_full).

Two unlabelled structures from the recording sessions (timestamps in the organisers'
Skeleton frame names, features/unit_times.csv):
  1. per-user normalisation: manner is relative to a person's pace, so IMU features are
     z-scored within each user's HAU clips (at test time: within each test session);
  2. consecutive clips of one user never share the emotion adverb (0 / 724 in training):
     Viterbi decoding along each session with that as a hard pairwise constraint.
The classifier is sub12's (RandomForest 500 trees, min_samples_leaf=2, labels seen >= 5).

    python src/emo_session_probe.py
"""
import sys, collections, warnings
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier as RF
from solver import load

tr, _ = load()
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')
imu = pd.read_csv('features/imu_feats.csv', index_col=0)
imu = imu.loc[:, imu.notna().any()]
imu = imu.fillna(imu.median())
e = tr[(tr.qt == 'emotion|HAU') & tr.path.isin(imu.index)].copy()
e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
e['t0'] = e.path.map(U.t0); e['t1'] = e.path.map(U.t1)
hau_users = tr[tr.source == 'HAU'].groupby('user').path.unique()

# per-user z-score over ALL of the user's HAU clips (labels not used)
Z = {}
for u, paths in hau_users.items():
    ps = [p for p in paths if p in imu.index]
    X = imu.loc[ps]
    Z.update({p: row for p, row in zip(ps, ((X - X.mean()) / (X.std() + 1e-6)).to_numpy(float))})
ZN = pd.DataFrame.from_dict(Z, orient='index', columns=imu.columns)
SETS = {'raw': imu, 'usernorm': ZN, 'raw+usernorm': imu.join(ZN, rsuffix='_z', how='inner')}
users = sorted(tr.user.unique())
FOLDS = [users[i::6] for i in range(6)]


def viterbi(chain):
    """chain: list of (opts: [(letter, adverb)], logp: [float]) -> chosen letters."""
    best = [chain[0][1][:]]; back = []
    for i in range(1, len(chain)):
        prev_opts, cur_opts = chain[i - 1][0], chain[i][0]
        b, bk = [], []
        for j, (L, adv) in enumerate(cur_opts):
            cands = [(best[-1][k], k) for k, (_, padv) in enumerate(prev_opts) if padv != adv]
            v, k = max(cands) if cands else (-1e18, 0)
            b.append(v + chain[i][1][j]); bk.append(k)
        best.append(b); back.append(bk)
    k = int(np.argmax(best[-1])); out = [k]
    for bk in reversed(back):
        k = bk[k]; out.append(k)
    out.reverse()
    return [chain[i][0][k][0] for i, k in enumerate(out)]


for name, df in SETS.items():
    ok_i = ok_v = n = 0
    for f in FOLDS:
        trn = e[~e.user.isin(f) & e.path.isin(df.index)]
        keep = {k for k, v in collections.Counter(trn.lab).items() if v >= 5}
        trn = trn[trn.lab.isin(keep)]
        clf = RF(500, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(df.loc[trn.path].to_numpy(float), trn.lab.to_numpy())
        pos = {c: i for i, c in enumerate(clf.classes_)}
        val = e[e.user.isin(f) & e.path.isin(df.index)].sort_values(['user', 't0'])
        P = dict(zip(val.path, clf.predict_proba(df.loc[val.path].to_numpy(float))))
        for u, g in val.groupby('user'):
            g = g.sort_values('t0'); sess = []; cur = []
            for r in g.itertuples():
                if cur and (r.t0 - cur[-1].t1).total_seconds() >= 1800:
                    sess.append(cur); cur = []
                cur.append(r)
            if cur:
                sess.append(cur)
            for s in sess:
                chain = []
                for r in s:
                    opts = [(L, str(getattr(r, L)).strip()) for L in 'ABCD']
                    pr = np.array([P[r.path][pos[a]] if a in pos else 0.0 for _, a in opts]) + 1e-3
                    chain.append((opts, list(np.log(pr / pr.sum()))))
                indep = [c[0][int(np.argmax(c[1]))][0] for c in chain]
                joint = viterbi(chain)
                for r, a, b in zip(s, indep, joint):
                    n += 1; ok_i += a == r.answer; ok_v += b == r.answer
    print('%-13s emotion acc: independent %.4f | session-constrained %.4f (n=%d)' % (name, ok_i / n, ok_v / n, n), flush=True)
