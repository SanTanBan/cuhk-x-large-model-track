"""Timestamp-free emotion check (09-14): IMU normalisation without recording order.

The organisers treated test-filename timestamps as a leak vector (Small-track topic 714827), so this looks for an
emotion gain that needs no timestamps. sub12's emotion = option prior + VLM letters + RF on raw IMU features,
decided per clip. Variants of the RF's IMU input, each decided per clip (no session decoding):
  raw        sub12 (baseline)
  setnorm    z-score the training clips with the training set's mean/std and the held-out clips with the
             held-out set's own mean/std (at test time: the whole test set). Transductive, no order, no ids.
  usernorm   z-score within each subject by the true user id. An upper bound only: test paths carry no user id.
  session    per-session normalisation exactly as the cache of src/emo_fast_validate2.py (needs timestamps).
             Regression check (must reproduce the cached IMU scores) and reference.
Folds as in the cache (6 cross-subject folds; the other folds train the RF). For each half of the subjects,
w_mot_emotion is tuned on the other half and scored on this half. A variant is accepted only if it beats raw on
BOTH halves. Option prior and VLM scores come from the cache.

    python src/emo_notime_validate.py [--cache=PATH]
"""
import sys, os, json, pickle, collections, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from solver import load, opts
from fuse import _centred
from feats import nonvisual, task_features

A = sys.argv[1:]
CACHE = next((a.split('=', 1)[1] for a in A if a.startswith('--cache=')),
             os.path.join(os.environ.get('TEMP', '.'), 'emo_norm_rows.pkl'))
rows, W0 = pickle.load(open(CACHE, 'rb'))
WV = W0.get('w_vlm_emotion', 0.0)
QT = 'emotion|HAU'
tr, te = load()
XKEYS = set(task_features('motion+clip')['X'])
IMU, _ = nonvisual('imu')
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')
users = sorted(tr.user.unique())
FOLDS = [users[i::6] for i in range(6)]
e_all = tr[tr.qt == QT].copy()
e_all['lab'] = [str(r[str(r.answer)]).strip() for _, r in e_all.iterrows()]
order = [r for f in FOLDS for _, r in tr[tr.user.isin(f) & (tr.qt == QT)].iterrows()]
assert len(order) == len(rows), (len(order), len(rows))
assert all(int(a.user) == int(b['user']) and str(a.answer) == b['ans'] for a, b in zip(order, rows)), 'cache order'
print('cache rows aligned: %d' % len(rows), flush=True)


def znorm(paths, table, group=None):
    by = collections.defaultdict(list)
    for p in dict.fromkeys(paths):
        if p in table:
            by[group(p) if group else 0].append(p)
    out = {}
    for ps in by.values():
        X = np.array([table[p] for p in ps])
        m, s = X.mean(0), X.std(0) + 1e-6
        for p, x in zip(ps, X):
            out[p] = (x - m) / s
    return out


def session_norm(paths_times, table):
    items = sorted([(t0, t1, p) for p, (t0, t1) in paths_times.items() if p in table and pd.notna(t0)])
    allX = np.array([table[p] for _, _, p in items]); gm, gs = allX.mean(0), allX.std(0) + 1e-6
    out, cur = {}, []

    def flush():
        X = np.array([table[p] for _, _, p in cur])
        m, s = (X.mean(0), X.std(0) + 1e-6) if len(cur) >= 3 else (gm, gs)
        for (_, _, p), x in zip(cur, X):
            out[p] = (x - m) / s

    for it in items:
        if cur and (it[0] - cur[-1][1]).total_seconds() >= 1800:
            flush(); cur = []
        cur.append(it)
    if cur:
        flush()
    return out


hau = tr[tr.source == 'HAU']
user_of = dict(zip(hau.path, hau.user))
SESS = session_norm({p: (U.t0.get(p), U.t1.get(p)) for p in hau.path.unique()}, IMU)
VAR = ['raw', 'setnorm', 'usernorm', 'session']
IM = {v: [None] * len(rows) for v in VAR}
idx = 0
for f in FOLDS:
    trn_p, hold_p = hau[~hau.user.isin(f)].path.unique(), hau[hau.user.isin(f)].path.unique()
    TT = {'raw': IMU, 'setnorm': {**znorm(trn_p, IMU), **znorm(hold_p, IMU)},
          'usernorm': znorm(list(trn_p) + list(hold_p), IMU, user_of.get), 'session': SESS}
    e = e_all[~e_all.user.isin(f) & e_all.path.isin(XKEYS)]
    keep = {k for k, v in collections.Counter(e.lab).items() if v >= 5}
    held = [r for _, r in tr[tr.user.isin(f) & (tr.qt == QT)].iterrows()]
    for v in VAR:
        T = TT[v]
        e2 = e[e.lab.isin(keep) & e.path.isin(T)]
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
        clf.fit(np.array([T[p] for p in e2.path]), e2.lab.to_numpy())
        for j, r in enumerate(held):
            O = opts(r); keys = [str(r[L]).strip() for L in O]
            mc = None
            if r.path in T:
                pr = clf.predict_proba(T[r.path].reshape(1, -1))[0]
                mot = {c: float(np.log(max(x, 1e-9))) for c, x in zip(clf.classes_, pr)}
                mc = _centred({k: mot[k] for k in keys if k in mot}, keys)
            IM[v][idx + j] = np.array([mc[k] if mc else 0.0 for k in keys])
    idx += len(held)
    print('fold %s done' % f, flush=True)

dev = max(float(np.abs(IM['session'][i] - rows[i]['imu']).max()) for i in range(len(rows)))
print('REGRESSION session variant vs cached IMU scores: max abs diff %.2e' % dev, flush=True)

RU = sorted({r['user'] for r in rows})
HALVES = [set(RU[0::2]), set(RU[1::2])]
ALL = set(RU)
WGRID = (2, 3, 5, 8, 12)


def acc(v, w, keep):
    ok = n = 0
    for i, r in enumerate(rows):
        if r['user'] in keep:
            s = r['prior'] + WV * r['vlm'] + w * IM[v][i]
            n += 1; ok += r['O'][int(np.argmax(s))] == r['ans']
    return ok / max(n, 1)


report = {}
for v in ['setnorm', 'usernorm', 'session']:
    gains, fits = [], []
    for i in (0, 1):
        fk, ek = HALVES[1 - i], HALVES[i]
        wb = max(WGRID, key=lambda w: acc('raw', w, fk))
        wn = max(WGRID, key=lambda w: acc(v, w, fk))
        gains.append(acc(v, wn, ek) - acc('raw', wb, ek)); fits.append((wb, wn))
    ok = all(g > 1e-9 for g in gains)
    wb = max(WGRID, key=lambda w: acc('raw', w, ALL))
    wn = max(WGRID, key=lambda w: acc(v, w, ALL))
    report[v] = dict(held_out_gains=gains, fits=fits, accepted=ok, w_raw=wb, w_new=wn,
                     acc_raw=acc('raw', wb, ALL), acc_new=acc(v, wn, ALL))
    print('%-9s held-out %+.4f / %+.4f  %s | fits (raw w, new w) %s | all subjects raw %.4f (w %s) -> %.4f (w %s)'
          % (v, gains[0], gains[1], 'ACCEPT' if ok else 'reject', fits, report[v]['acc_raw'], wb,
             report[v]['acc_new'], wn), flush=True)
json.dump(report, open('src/emo_notime_report.json', 'w'), indent=1, default=str)
print('wrote src/emo_notime_report.json')
