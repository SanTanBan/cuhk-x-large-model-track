"""Emotion session decoding, head-to-head in the full pipeline (09-13).

Consecutive HAU clips of one user never share the emotion adverb (0 / 724 training pairs;
recording times from features/unit_times.csv). sub12's fused emotion score per option is
rebuilt exactly (option prior + VLM letter evidence + IMU RandomForest evidence, weights from
the base file). Each user's clips are then decoded jointly along recording sessions (gap < 30
min), with Viterbi under that hard no-repeat constraint and emissions log_softmax(score / tau).
tau is tuned on one half of the subjects and scored on the other; accepted only if decoding
beats the independent argmax (= sub12's answers) on BOTH halves. Nothing shared is edited.

    python src/emo_session_validate.py [--base=src/best_nv_final_W.json] [--dry-run]
"""
import sys, os, json, collections

ARGS = sys.argv[1:]
BASE_F = next((a.split('=', 1)[1] for a in ARGS if a.startswith('--base=')), 'src/best_nv_final_W.json')
DRY = '--dry-run' in ARGS
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, X, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fuse import _centred
from solver import lo, opts
from feats import nonvisual

W = dict(DEFAULT_W); W.update(json.load(open(BASE_F)))
IMU, _ = nonvisual('imu')
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')
QT = 'emotion|HAU'


def emotion_lp(trn):
    """sub12's emotion classifier, mirroring MotionModel (labels counted on clips in X)."""
    e = trn[trn.qt == QT].copy()
    e = e[e.path.isin(X)]
    e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
    keep = {k for k, v in collections.Counter(e.lab).items() if v >= 5}
    e2 = e[e.lab.isin(keep) & e.path.isin(IMU)]
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    clf.fit(np.array([IMU[p] for p in e2.path]), e2.lab.to_numpy())
    return clf


rows = []          # one per validation emotion question: user, path, t0, t1, opts, scores, answer
check_mismatch = check_n = 0
for (M_, MM_, HP_, SQ_, clips), f in zip(FOLD, folds):
    clf = emotion_lp(tr[~tr.user.isin(f)])
    for g, feats_, hs, e_ in clips:
        eq = g[g.qt == QT]
        if not len(eq):
            continue
        path = g.iloc[0].path
        mot_e = {}
        if path in IMU:
            p = clf.predict_proba(IMU[path].reshape(1, -1))[0]
            mot_e = {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}
        e2 = dict(e_); e2['mot_e'] = mot_e
        if check_n < 150:                   # regression: our argmax == the pipeline's answer
            a = answer_clip_fused(g, M_, W, VLM, feats_, MM_, HP_, SQ_, e2)
        for _, r in eq.iterrows():
            O = opts(r)
            c = _centred(VLM.get(r.qa_id, {}).get('letter', {}), O)
            keys = [str(r[L]).strip() for L in O]
            mc = _centred({k: mot_e[k] for k in keys if k in mot_e}, keys) if mot_e else None
            sc = [lo(M_.p(QT, r[L])) + W.get('w_vlm_emotion', 0.0) * c[L]
                  + (W.get('w_mot_emotion', 0.0) * mc[str(r[L]).strip()] if mc else 0.0) for L in O]
            if check_n < 150:
                check_n += 1
                check_mismatch += O[int(np.argmax(sc))] != a[r.qa_id]
            rows.append(dict(user=r.user, path=path, hs=hs, O=O, adv=[k.lower() for k in keys],
                             sc=np.array(sc), ans=str(r.answer),
                             t0=U.t0.get(path), t1=U.t1.get(path)))
print('regression: rebuilt argmax vs pipeline answer mismatches %d / %d' % (check_mismatch, check_n))
R = pd.DataFrame(rows)
print('validation emotion questions: %d | with recording times: %d' % (len(R), R.t0.notna().sum()))


def chains(sub):
    out = []
    for u, g in sub.sort_values('t0').groupby('user'):
        cur = []
        for r in g.itertuples():
            if pd.isna(r.t0):
                out.append([r]); continue
            if cur and (r.t0 - cur[-1].t1).total_seconds() >= 1800:
                out.append(cur); cur = []
            cur.append(r)
        if cur:
            out.append(cur)
    return out


def decode(chain, tau):
    em = []
    for r in chain:
        z = r.sc / tau
        em.append(z - (z.max() + np.log(np.exp(z - z.max()).sum())))
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


def acc(sub, tau):
    ok = n = 0
    for ch in chains(sub):
        pred = [r.O[int(np.argmax(r.sc))] for r in ch] if tau is None else decode(ch, tau)
        for r, p in zip(ch, pred):
            n += 1; ok += p == r.ans
    return ok / max(n, 1), n


users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
TAUS = [0.25, 0.5, 1, 2, 4, 8, 16]
gains, fits = [], []
for i in (0, 1):
    fit = R[R.user.isin(HALVES[1 - i])]; ev = R[R.user.isin(HALVES[i])]
    tau = max(TAUS, key=lambda t: acc(fit, t)[0]); fits.append(tau)
    a0, n = acc(ev, None); a1, _ = acc(ev, tau)
    gains.append(a1 - a0)
    print('held-out half %d: emotion acc independent %.4f -> session-decoded %.4f (%+.4f, n=%d, tau %s)' % (i, a0, a1, a1 - a0, n, tau))
tau_all = max(TAUS, key=lambda t: acc(R, t)[0])
a0, n = acc(R, None); a1, _ = acc(R, tau_all)
ok = all(g > 1e-9 for g in gains)
w_e = sum(v for k, v in TESTC.items() if k[0] == QT) / sum(TESTC.values())
print('all subjects: %.4f -> %.4f (tau %s) | overall test-weighted %+.4f | verdict %s' % (a0, a1, tau_all, (a1 - a0) * w_e, 'ACCEPT' if ok else 'reject'))
for t in TAUS:
    print('   tau %-5s all-subject acc %.4f' % (t, acc(R, t)[0]))
if not DRY:
    json.dump(dict(held_out_gains=gains, taus=fits, tau_all=tau_all, acc_all=[a0, a1], accepted=ok,
                   base=BASE_F, regression_mismatch=[check_mismatch, check_n]),
              open('src/emo_session_report.json', 'w'), indent=1)
    print('wrote src/emo_session_report.json')
