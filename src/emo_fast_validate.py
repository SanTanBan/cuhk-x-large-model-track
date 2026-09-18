"""Emotion, fast head-to-head (09-13): recording-session decoding and per-session IMU
normalisation, without the full CV prep. The emotion answer depends only on the option prior,
the VLM letter evidence and the IMU evidence, so those are rebuilt per subject fold directly
(same 6 folds as cv_full), and a regression check confirms the rebuilt argmax equals
answer_clip_fused's emotion answer.

Variants (baseline = raw, which is exactly sub12's emotion):
  raw        IMU RandomForest (500 trees, leaf 2, labels seen >= 5), independent argmax
  raw+sess   Viterbi along recording sessions: consecutive clips never share the adverb
  norm       IMU features z-scored within each recording session (unlabelled; same on test)
  norm+sess
Sessions = HAU clips sorted by recording time, split at gaps >= 30 min, users ignored (as on test).
Each non-baseline variant tunes w_mot_emotion and the temperature tau on one half of the
subjects, is scored on the other, and is accepted only if it beats raw on BOTH halves.

    python src/emo_fast_validate.py [--base=src/best_nv_final_W.json]
"""
import sys, os, json, collections, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from solver import load, lo, opts
from joint import Joint, clip_features
from fuse import _centred, load_vlm, DEFAULT_W, answer_clip_fused
from feats import nonvisual, task_features

BASE_F = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--base=')), 'src/best_nv_final_W.json')
W0 = dict(DEFAULT_W); W0.update(json.load(open(BASE_F)))
QT = 'emotion|HAU'
tr, te = load()
VLM = load_vlm('vlm/vlm_scores_train.csv')
XKEYS = set(task_features('motion+clip')['X'])
IMU, _ = nonvisual('imu')
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')


def session_norm(paths_times, table):
    """z-score each clip's features within its recording session (gap >= 30 min splits)."""
    items = sorted([(t0, t1, p) for p, (t0, t1) in paths_times.items() if p in table and pd.notna(t0)])
    allX = np.array([table[p] for _, _, p in items])
    gm, gs = allX.mean(0), allX.std(0) + 1e-6
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


hau_train = tr[tr.source == 'HAU'].path.unique()
IMU_N = session_norm({p: (U.t0.get(p), U.t1.get(p)) for p in hau_train}, IMU)
FEATS = {'raw': IMU, 'norm': IMU_N}
users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]
e_all = tr[tr.qt == QT].copy()
e_all['lab'] = [str(r[str(r.answer)]).strip() for _, r in e_all.iterrows()]

rows = collections.defaultdict(list)       # feature set -> list of question dicts
mism = nchk = 0
for f in folds:
    trn = tr[~tr.user.isin(f)]
    M = Joint(trn)
    e = e_all[~e_all.user.isin(f) & e_all.path.isin(XKEYS)]
    keep = {k for k, v in collections.Counter(e.lab).items() if v >= 5}
    val = tr[tr.user.isin(f) & (tr.qt == QT)]
    for fname, T in FEATS.items():
        e2 = e[e.lab.isin(keep) & e.path.isin(T)]
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
        clf.fit(np.array([T[p] for p in e2.path]), e2.lab.to_numpy())
        for _, r in val.iterrows():
            mot_e = {}
            if r.path in T:
                p = clf.predict_proba(T[r.path].reshape(1, -1))[0]
                mot_e = {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}
            O = opts(r)
            c = _centred(VLM.get(r.qa_id, {}).get('letter', {}), O)
            keys = [str(r[L]).strip() for L in O]
            mc = _centred({k: mot_e[k] for k in keys if k in mot_e}, keys) if mot_e else None
            prior = np.array([lo(M.p(QT, r[L])) for L in O])
            vlm = np.array([c[L] for L in O])
            imu = np.array([mc[k] if mc else 0.0 for k in keys])
            if fname == 'raw' and nchk < 120:
                g = tr[tr.path == r.path]
                ev = {'mot_h': {}, 'mot_e': mot_e, 'mot_o': {}, 'mot_p': {}, 'zs_h': {}, 'zs_p': {},
                      'seq': {}, 'prec': {}}
                a = answer_clip_fused(g, M, W0, VLM, clip_features(g, M), None, None, None, ev)
                sc = prior + W0.get('w_vlm_emotion', 0.0) * vlm + W0.get('w_mot_emotion', 0.0) * imu
                nchk += 1; mism += O[int(np.argmax(sc))] != a[r.qa_id]
            rows[fname].append(dict(user=r.user, path=r.path, t0=U.t0.get(r.path), t1=U.t1.get(r.path), O=O,
                                    adv=[k.lower() for k in keys], prior=prior, vlm=vlm, imu=imu,
                                    ans=str(r.answer)))
print('regression: rebuilt raw argmax vs answer_clip_fused mismatches %d / %d' % (mism, nchk), flush=True)
R = {k: pd.DataFrame(v) for k, v in rows.items()}


def chains(df):
    out, cur = [], []
    for r in df.sort_values('t0').itertuples():
        if pd.isna(r.t0):
            out.append([r]); continue
        if cur and (r.t0 - cur[-1].t1).total_seconds() >= 1800:
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def scores(r, wm):
    return r.prior + W0.get('w_vlm_emotion', 0.0) * r.vlm + wm * r.imu


def decode(chain, wm, tau):
    em = []
    for r in chain:
        z = scores(r, wm) / tau
        z = z - z.max()
        em.append(z - np.log(np.exp(z).sum()))
    best, back = [em[0]], []
    for i in range(1, len(chain)):
        b, bk = [], []
        for j, adv in enumerate(chain[i].adv):
            cands = [(best[-1][k], k) for k, padv in enumerate(chain[i - 1].adv) if padv != adv]
            v, k = max(cands) if cands else (-1e18, 0)
            b.append(v + em[i][j]); bk.append(k)
        best.append(np.array(b)); back.append(bk)
    k = int(np.argmax(best[-1])); path = [k]
    for bk in reversed(back):
        k = bk[k]; path.append(k)
    path.reverse()
    return [chain[i].O[k] for i, k in enumerate(path)]


def acc(df, wm, tau):
    ok = n = 0
    for ch in chains(df):
        pred = [r.O[int(np.argmax(scores(r, wm)))] for r in ch] if tau is None else decode(ch, wm, tau)
        for r, p in zip(ch, pred):
            n += 1; ok += p == r.ans
    return ok / max(n, 1)


HALVES = [set(users[0::2]), set(users[1::2])]
WMS, TAUS = [3, 5, 8, 12], [0.25, 0.5, 1, 2, 4, 8]
W8 = W0.get('w_mot_emotion', 8)
report = {}
print('\n%-10s %16s %16s  %s' % ('variant', 'held-out half 0', 'held-out half 1', 'verdict'))
for name, feat, sess in (('raw+sess', 'raw', True), ('norm', 'norm', False), ('norm+sess', 'norm', True)):
    grid = [(wm, t) for wm in (WMS if feat != 'raw' else [W8]) for t in (TAUS if sess else [None])]
    gains, fits = [], []
    for i in (0, 1):
        fitd, evd = R[feat][R[feat].user.isin(HALVES[1 - i])], R[feat][R[feat].user.isin(HALVES[i])]
        wm, t = max(grid, key=lambda x: acc(fitd, x[0], x[1]))
        base = acc(R['raw'][R['raw'].user.isin(HALVES[i])], W8, None)
        gains.append(acc(evd, wm, t) - base); fits.append((wm, t))
    ok = all(g > 1e-9 for g in gains)
    wm, t = max(grid, key=lambda x: acc(R[feat], x[0], x[1]))
    a0, a1 = acc(R['raw'], W8, None), acc(R[feat], wm, t)
    report[name] = dict(held_out_gains=gains, fits=fits, final=[wm, t], acc_all=[a0, a1], accepted=ok)
    print('%-10s %+16.4f %+16.4f  %s   fits %s | all subjects %.4f -> %.4f with (w %s, tau %s)'
          % (name, gains[0], gains[1], 'ACCEPT' if ok else 'reject', fits, a0, a1, wm, t), flush=True)
json.dump(report, open('src/emo_fast_report.json', 'w'), indent=1, default=str)
print('wrote src/emo_fast_report.json')
