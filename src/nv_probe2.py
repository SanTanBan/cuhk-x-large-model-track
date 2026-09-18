"""HARn probe, part 2 (09-11). The non-visual package covers ~2,925 HARn units, but only 429
of them carry a `single|HARn` question -- and each unit's action label is in its folder name
(HARn/<action>/<user>/<trial>). So the Skeleton / IMU action classifier can train on all of
them: ~7x the data the question-bearing clips give. Same 6 subject folds as cv_full; units of
validation-fold users are excluded from training, and the action -> option-text map comes
from training-fold questions only.

    python src/nv_probe2.py
"""
import sys, re, collections, warnings
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from solver import load

tr, te = load()


def table(f):
    d = pd.read_csv(f, index_col=0)
    d = d.loc[:, d.notna().any()]
    return d.fillna(d.median())


IMU, SKEL = table('features/imu_feats.csv'), table('features/skel_feats.csv')
both = SKEL.join(IMU, how='inner', rsuffix='_imu')
SETS = {'skel': SKEL, 'imu': IMU, 'skel+imu': both}
users = sorted(tr.user.unique())
FOLDS = [users[i::6] for i in range(6)]
h = tr[tr.qt == 'single|HARn'].copy(); h['action'] = h.path.str.split('/').str[1]
oi = tr[tr.qt == 'object_interaction|HARn'].copy(); oi['action'] = oi.path.str.split('/').str[1]


def units(df):
    idx = [k for k in df.index if k.startswith('HARn/')]
    act = [k.split('/')[1] for k in idx]
    usr = [int(re.search(r'user(\d+)', k).group(1)) for k in idx]
    return idx, np.array(act), np.array(usr)


def fit(xs, ys, kind):
    if kind == 'rf':
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=0)
        return None, clf.fit(xs, ys)
    sc = StandardScaler().fit(xs)
    return sc, LogisticRegression(max_iter=3000, C=float(kind[2:])).fit(sc.transform(xs), ys)


def run(name, kind, all_units=True):
    df = SETS[name]
    idx, act, usr = units(df)
    Xall = df.loc[idx].to_numpy(float)
    pos = {k: i for i, k in enumerate(idx)}
    res = collections.Counter()
    for f in FOLDS:
        if all_units:
            m = ~np.isin(usr, f)
        else:        # only the question-bearing clips (what MotionModel does today)
            q = set(h[~h.user.isin(f)].path)
            m = np.array([k in q for k in idx])
        sc, clf = fit(Xall[m], act[m], kind)
        act2txt = {r.action: str(r[str(r.answer)]).strip() for _, r in h[~h.user.isin(f)].iterrows()}
        o2 = collections.defaultdict(collections.Counter)
        for _, r in oi[~oi.user.isin(f)].iterrows():
            o2[r.action][str(r[str(r.answer)]).strip()] += 1

        def lp(path):
            x = Xall[pos[path]][None]
            p = clf.predict_proba(sc.transform(x) if sc is not None else x)[0]
            return {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}
        for _, r in h[h.user.isin(f)].iterrows():
            res['h_n'] += 1
            if r.path not in pos:
                continue
            t = {}
            for a, v in lp(r.path).items():
                if a in act2txt:
                    t[act2txt[a]] = max(t.get(act2txt[a], -1e9), v)
            s = {L: t[str(r[L]).strip()] for L in 'ABCD' if str(r[L]).strip() in t}
            if s:
                res['h_ok'] += max(s, key=s.get) == str(r.answer)
        for _, r in oi[oi.user.isin(f)].iterrows():
            res['o_n'] += 1
            if r.path not in pos:
                continue
            acc = collections.defaultdict(float)
            for a, v in lp(r.path).items():
                c = o2.get(a)
                if c:
                    tot = sum(c.values())
                    for o, k in c.items():
                        acc[o] += np.exp(v) * k / tot
            s = {L: acc[str(r[L]).strip()] for L in 'ABCD' if str(r[L]).strip() in acc}
            if s:
                res['o_ok'] += max(s, key=s.get) == str(r.answer)
        res['train_n'] += int(m.sum())
    return res['h_ok'] / res['h_n'], res['o_ok'] / res['o_n'], res['train_n'] // 6


if __name__ == '__main__':
    print('HARn units with features: ' + ', '.join('%s %d' % (k, len(units(v)[0])) for k, v in SETS.items()))
    print('single|HARn (n=429) / object_interaction|HARn (n=133)   [current CLIP model: 0.699 / 0.857 in the full pipeline]')
    for name in ('skel', 'imu', 'skel+imu'):
        for kind in ('lr0.1', 'lr1.0', 'rf'):
            for allu in (False, True):
                a, o, n = run(name, kind, allu)
                print('   %-9s %-6s %-14s single %.3f  object %.3f  (train ~%d units/fold)'
                      % (name, kind, 'ALL units' if allu else 'question clips', a, o, n), flush=True)
