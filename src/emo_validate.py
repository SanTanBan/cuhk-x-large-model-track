"""Head-to-head for `emotion` (09-13): a candidate non-visual emotion classifier against sub12's
(IMU features, RandomForest 500 trees, min_samples_leaf=2, labels seen >= 5 times), inside the
full pipeline, on held-out subjects.

The CV prep runs once (CUHKX_* cleared). For every fold BOTH classifiers are fit on that fold's
training users, and only the emotion evidence (`ev['mot_e']`) is swapped. For each half of the
subjects both versions have (w_mot_emotion, w_vlm_emotion) tuned on the other half and are
scored on this half. The candidate is accepted only if it wins on both halves. Nothing shared is
edited: classifiers are fit here and injected into the evidence dicts.

    python src/emo_validate.py --new=et:400:2:auto[:imu+skel] [--min-n=5]
                               [--base=src/best_nv_final_W.json] [--out=src/best_emo_W.json] [--dry-run]

spec = model:trees:min_samples_leaf:max_features[:features], model in rf|et, features imu|imu+skel.
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in ARGS if a.startswith('--%s=' % k)), d)


NEW, MIN_N = arg('new', 'et:400:2:auto'), int(arg('min-n', '5'))
BASE_F, OUT_F, DRY = arg('base', 'src/best_nv_final_W.json'), arg('out', 'src/best_emo_W.json'), '--dry-run' in ARGS
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def table(names):
    ds = []
    for n in names.split('+'):
        d = pd.read_csv('features/%s_feats.csv' % n, index_col=0)
        d = d.loc[:, d.notna().any()]
        ds.append(d.fillna(d.median()).add_prefix(n + '_'))
    out = ds[0]
    for d in ds[1:]:
        out = out.join(d, how='inner')
    return out


def make(spec):
    p = spec.split(':')
    kind, trees, leaf, mf = p[0], int(p[1]), int(p[2]), p[3]
    feats = p[4] if len(p) > 4 else 'imu'
    mf = None if mf == 'none' else (mf if mf in ('sqrt', 'log2', 'auto') else float(mf))
    if mf == 'auto':
        mf = 'sqrt'
    cls = RandomForestClassifier if kind == 'rf' else ExtraTreesClassifier
    return (lambda: cls(n_estimators=trees, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=0)), feats


e = tr[tr.qt == 'emotion|HAU'].copy()
e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]


def evidence(spec, min_n):
    mk, feats = make(spec)
    T = table(feats)
    per_fold = []
    for f in folds:
        trn = e[~e.user.isin(f) & e.path.isin(T.index)]
        keep = {k for k, v in collections.Counter(trn.lab).items() if v >= min_n}
        trn = trn[trn.lab.isin(keep)]
        clf = mk().fit(T.loc[trn.path].to_numpy(float), trn.lab.to_numpy())
        paths = [g.iloc[0].path for g, _, _, _ in FOLD[len(per_fold)][4] if g.iloc[0].path in T.index]
        P = clf.predict_proba(T.loc[paths].to_numpy(float)) if paths else []
        per_fold.append({p: {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, row)}
                         for p, row in zip(paths, P)})
    return per_fold


OLD_SPEC = 'rf:500:2:auto:imu'
EV = {'old': evidence(OLD_SPEC, 5), 'new': evidence(NEW, MIN_N)}


def fold_set(which):
    out = []
    for (M_, MM_, HP_, SQ_, clips), lp in zip(FOLD, EV[which]):
        new = []
        for g, feats_, hs, e_ in clips:
            e2 = dict(e_); e2['mot_e'] = lp.get(g.iloc[0].path, {})
            new.append((g, feats_, hs, e2))
        out.append((M_, MM_, HP_, SQ_, new))
    return out


FL = {'old': fold_set('old'), 'new': fold_set('new')}
CATS = ['emotion|HAU']
KEYS = ['w_mot_emotion', 'w_vlm_emotion']
GRID = {'w_mot_emotion': [0, 1, 2, 3, 5, 8, 12], 'w_vlm_emotion': [0, 0.25, 0.5]}
START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]


def score(fl, W, keep):
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in fl:
        for g, feats_, hs, e_ in clips:
            if (keep is not None and g.iloc[0].user not in keep) or not g.qt.isin(CATS).any():
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats_, MM_, HP_, SQ_, e_)
            for _, r in g[g.qt.isin(CATS)].iterrows():
                k = (r['qt'], hs); res[k][1] += 1; res[k][0] += a[r.qa_id] == str(r.answer)
    s = sum(TESTC[k] * ok / n for k, (ok, n) in res.items() if n >= 5) / N_TEST
    acc = sum(ok for ok, n in res.values()) / max(1, sum(n for ok, n in res.values()))
    return s, acc


def tune(fl, keep):
    best_v, best_s = None, -1.0
    for combo in itertools.product(*[GRID[k] for k in KEYS]):
        W = dict(START); W.update(dict(zip(KEYS, combo)))
        s = score(fl, W, keep)[0]
        if s > best_s + 1e-12:
            best_s, best_v = s, dict(zip(KEYS, combo))
    return best_v


def W_(v):
    W = dict(START); W.update(v); return W


print('\nemotion head-to-head: old %s vs new %s (min_n %d)' % (OLD_SPEC, NEW, MIN_N))
gains = []
for i in (0, 1):
    vo, vn = tune(FL['old'], HALVES[1 - i]), tune(FL['new'], HALVES[1 - i])
    so, ao = score(FL['old'], W_(vo), HALVES[i]); sn, an = score(FL['new'], W_(vn), HALVES[i])
    gains.append(sn - so)
    print('held-out half %d: emotion acc %.4f -> %.4f | overall %+.4f | fits old %s new %s' % (i, ao, an, sn - so, vo, vn))
vo_all, vn_all = tune(FL['old'], None), tune(FL['new'], None)
ao_all, an_all = score(FL['old'], W_(vo_all), None)[1], score(FL['new'], W_(vn_all), None)[1]
ok = all(g > 1e-9 for g in gains)
print('all subjects (in-sample): emotion acc %.4f -> %.4f | verdict %s | final weights %s' % (ao_all, an_all, 'ACCEPT' if ok else 'reject', vn_all))
rep = dict(new=NEW, min_n=MIN_N, old=OLD_SPEC, held_out_gains=gains, accepted=ok, weights=vn_all,
           acc_all=[ao_all, an_all], base=BASE_F)
if not DRY:
    if ok:
        json.dump(W_(vn_all), open(OUT_F, 'w'), indent=1, sort_keys=True)
    json.dump(rep, open(OUT_F.replace('.json', '_report.json'), 'w'), indent=1)
    print('wrote', (OUT_F + ' and ') if ok else '', OUT_F.replace('.json', '_report.json'))
