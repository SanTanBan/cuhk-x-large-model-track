"""Emotion probe (09-13): which non-visual model picks the right `emotion` option best,
standalone, cross-subject (same 6 subject folds as cv_full)? Variants of the random forest in
sub12 (500 trees, min_samples_leaf=2, labels seen >= 5 times), extra-trees, class weights,
rare-label coverage and feature sets. Read-only: imports shared modules, edits nothing.

    python src/emo_probe.py
"""
import sys, collections, warnings, time
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier as RF, ExtraTreesClassifier as ET
from solver import load
from motion_model import motion_table

tr, te = load()


def table(f):
    d = pd.read_csv(f, index_col=0)
    d = d.loc[:, d.notna().any()]
    return d.fillna(d.median())


IMU, SKEL = table('features/imu_feats.csv'), table('features/skel_feats.csv')
M, _ = motion_table('motion2')
MOT = pd.DataFrame.from_dict(M, orient='index')
SETS = {'imu': IMU, 'imu+skel': IMU.join(SKEL, how='inner', rsuffix='_sk'),
        'imu+motion': IMU.join(MOT, how='inner', rsuffix='_mo')}
e = tr[tr.qt == 'emotion|HAU'].copy()
e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
cnt = collections.Counter(e.lab)
print('emotion questions %d | distinct answers %d | answers seen >=5: %d (covering %.0f%% of questions)'
      % (len(e), len(cnt), sum(v >= 5 for v in cnt.values()),
         100 * sum(v for v in cnt.values() if v >= 5) / len(e)))
users = sorted(tr.user.unique())
FOLDS = [users[i::6] for i in range(6)]


def run(df, make, min_n=5):
    ok = n = cov = 0
    for f in FOLDS:
        trn = e[~e.user.isin(f) & e.path.isin(df.index)]
        keep = {k for k, v in collections.Counter(trn.lab).items() if v >= min_n}
        trn = trn[trn.lab.isin(keep)]
        clf = make().fit(df.loc[trn.path].to_numpy(float), trn.lab.to_numpy())
        val = e[e.user.isin(f)]
        have = val[val.path.isin(df.index)]
        P = clf.predict_proba(df.loc[have.path].to_numpy(float)) if len(have) else []
        pos = {c: i for i, c in enumerate(clf.classes_)}
        for (_, r), p in zip(have.iterrows(), P):
            s = {L: p[pos[str(r[L]).strip()]] for L in 'ABCD' if str(r[L]).strip() in pos}
            if s:
                cov += 1
                ok += max(s, key=s.get) == str(r.answer)
        n += len(val)
    return ok / n, cov / n


V = [
    ('imu', 5, 'rf500_leaf2 (sub12)', lambda: RF(500, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ('imu', 5, 'rf1000_leaf1', lambda: RF(1000, min_samples_leaf=1, n_jobs=-1, random_state=0)),
    ('imu', 5, 'rf1000_leaf3_mf0.2', lambda: RF(1000, min_samples_leaf=3, max_features=0.2, n_jobs=-1, random_state=0)),
    ('imu', 5, 'rf1000_leaf2_log2', lambda: RF(1000, min_samples_leaf=2, max_features='log2', n_jobs=-1, random_state=0)),
    ('imu', 5, 'rf1000_leaf2_balanced', lambda: RF(1000, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=0)),
    ('imu', 5, 'et1000_leaf2', lambda: ET(1000, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ('imu', 5, 'et1000_leaf1_mf0.3', lambda: ET(1000, min_samples_leaf=1, max_features=0.3, n_jobs=-1, random_state=0)),
    ('imu', 2, 'rf1000_leaf2 min_n=2', lambda: RF(1000, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ('imu+skel', 5, 'rf1000_leaf2', lambda: RF(1000, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ('imu+skel', 5, 'et1000_leaf2', lambda: ET(1000, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    ('imu+motion', 5, 'et1000_leaf2', lambda: ET(1000, min_samples_leaf=2, n_jobs=-1, random_state=0)),
]
print('%-11s %-26s %8s %6s %6s' % ('features', 'model', 'acc', 'cov', 'sec'))
for name, mn, label, make in V:
    t = time.time()
    a, c = run(SETS[name], make, mn)
    print('%-11s %-26s %8.3f %6.2f %6.0f' % (name, label, a, c, time.time() - t), flush=True)
