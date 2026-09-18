"""Emotion, refinement of sub13e's session decoder (09-13), fast head-to-head on held-out subjects.

sub13e = per-session normalised IMU evidence + Viterbi along recording sessions with a hard
"consecutive clips never share the adverb" constraint (public +8). Training shows more structure:
  - same adverb at lag 2 is 0.011 vs 0.043 chance (lags 3 and 6 are elevated: manners cycle in threes)
  - a clip's options overlap its neighbours' options more for the answer than for distractors
    (likelihood ratio ~1.5 at lags +-1, ~1.4 at +-2)
Variants, compared with the sub13e decoder (first order, hard lag-1), each tuned on one half of the
subjects and scored on the other, accepted only if better on BOTH halves:
  order2      second-order Viterbi: hard lag-1 + gamma2 * [same adverb at lag 2]
  nbopt       + beta * (number of neighbours at lags +-1, +-2 whose options contain this option)
  order2+nb   both
The per-fold scores are cached in the scratchpad (pickle) so further refinements skip the RF fits.

    python src/emo_fast_validate2.py [--base=src/best_nv_final_W.json] [--cache=PATH]
"""
import sys, os, json, collections, itertools, pickle, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
import numpy as np, pandas as pd

A = sys.argv[1:]
BASE_F = next((a.split('=', 1)[1] for a in A if a.startswith('--base=')), 'src/best_nv_final_W.json')
CACHE = next((a.split('=', 1)[1] for a in A if a.startswith('--cache=')),
             os.path.join(os.environ.get('TEMP', '.'), 'emo_norm_rows.pkl'))

if os.path.exists(CACHE):
    rows, W0 = pickle.load(open(CACHE, 'rb'))
    print('loaded cached per-question scores from', CACHE)
else:
    from sklearn.ensemble import RandomForestClassifier
    from solver import load, lo, opts
    from joint import Joint
    from fuse import _centred, load_vlm, DEFAULT_W
    from feats import nonvisual, task_features
    W0 = dict(DEFAULT_W); W0.update(json.load(open(BASE_F)))
    QT = 'emotion|HAU'
    tr, te = load()
    VLM = load_vlm('vlm/vlm_scores_train.csv')
    XKEYS = set(task_features('motion+clip')['X'])
    IMU, _ = nonvisual('imu')
    U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')

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

    T = session_norm({p: (U.t0.get(p), U.t1.get(p)) for p in tr[tr.source == 'HAU'].path.unique()}, IMU)
    users = sorted(tr.user.unique())
    e_all = tr[tr.qt == QT].copy()
    e_all['lab'] = [str(r[str(r.answer)]).strip() for _, r in e_all.iterrows()]
    rows = []
    for f in [users[i::6] for i in range(6)]:
        M = Joint(tr[~tr.user.isin(f)])
        e = e_all[~e_all.user.isin(f) & e_all.path.isin(XKEYS)]
        keep = {k for k, v in collections.Counter(e.lab).items() if v >= 5}
        e2 = e[e.lab.isin(keep) & e.path.isin(T)]
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
        clf.fit(np.array([T[p] for p in e2.path]), e2.lab.to_numpy())
        for _, r in tr[tr.user.isin(f) & (tr.qt == QT)].iterrows():
            mot_e = {}
            if r.path in T:
                p = clf.predict_proba(T[r.path].reshape(1, -1))[0]
                mot_e = {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}
            O = opts(r); keys = [str(r[L]).strip() for L in O]
            c = _centred(VLM.get(r.qa_id, {}).get('letter', {}), O)
            mc = _centred({k: mot_e[k] for k in keys if k in mot_e}, keys) if mot_e else None
            rows.append(dict(user=r.user, t0=U.t0.get(r.path), t1=U.t1.get(r.path), O=O,
                             adv=[' '.join(k.lower().split()) for k in keys],
                             prior=np.array([lo(M.p(QT, r[L])) for L in O]),
                             vlm=np.array([c[L] for L in O]),
                             imu=np.array([mc[k] if mc else 0.0 for k in keys]), ans=str(r.answer)))
    pickle.dump((rows, W0), open(CACHE, 'wb'))
    print('cached per-question scores to', CACHE)

R = pd.DataFrame(rows)
WV = W0.get('w_vlm_emotion', 0.0)


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


def emissions(chain, w, tau, beta):
    em = []
    for i, r in enumerate(chain):
        s = r.prior + WV * r.vlm + w * r.imu
        if beta:
            nb = [set(chain[j].adv) for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(chain)]
            s = s + beta * np.array([sum(a in o for o in nb) for a in r.adv])
        z = s / tau; z = z - z.max()
        em.append(z - np.log(np.exp(z).sum()))
    return em


def viterbi1(chain, em):
    best, back = [em[0]], []
    for i in range(1, len(chain)):
        b, bk = [], []
        for j, adv in enumerate(chain[i].adv):
            cands = [(best[-1][k], k) for k, pa in enumerate(chain[i - 1].adv) if pa != adv]
            v, k = max(cands) if cands else (-1e18, 0)
            b.append(v + em[i][j]); bk.append(k)
        best.append(np.array(b)); back.append(bk)
    k = int(np.argmax(best[-1])); path = [k]
    for bk in reversed(back):
        k = bk[k]; path.append(k)
    return list(reversed(path))


def viterbi2(chain, em, gamma2):
    """State at i = (choice at i-1, choice at i); hard lag-1, soft lag-2 repeat penalty."""
    n = len(chain)
    if n < 3:
        return viterbi1(chain, em)
    V = {}
    for a, pa in enumerate(chain[0].adv):
        for b, pb in enumerate(chain[1].adv):
            if pa != pb:
                V[(a, b)] = (em[0][a] + em[1][b], None)
    hist = [V]
    for i in range(2, n):
        NV = {}
        for (a, b), (v, _) in hist[-1].items():
            for c, pc in enumerate(chain[i].adv):
                if chain[i - 1].adv[b] == pc:
                    continue
                s = v + em[i][c] + (gamma2 if chain[i - 2].adv[a] == pc else 0.0)
                if (b, c) not in NV or s > NV[(b, c)][0]:
                    NV[(b, c)] = (s, a)
        if not NV:
            return viterbi1(chain, em)
        hist.append(NV)
    (b, c), _ = max(hist[-1].items(), key=lambda kv: kv[1][0])
    path = [c, b]
    for i in range(len(hist) - 1, 0, -1):
        a = hist[i][(b, c)][1]
        path.append(a); b, c = a, b
    return list(reversed(path))


def acc(df, w, tau, gamma2=None, beta=0.0):
    ok = n = 0
    for ch in chains(df):
        em = emissions(ch, w, tau, beta)
        path = viterbi1(ch, em) if gamma2 is None else viterbi2(ch, em, gamma2)
        for r, k in zip(ch, path):
            n += 1; ok += r.O[k] == r.ans
    return ok / max(n, 1)


users = sorted(R.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
BASEGRID = [(w, t) for w in (5, 8, 12) for t in (0.25, 0.5)]
VARIANTS = {
    'order2': lambda: [(w, t, g, 0.0) for w, t in BASEGRID for g in (-0.5, -1.5, -4.0)],
    'nbopt': lambda: [(w, t, None, b) for w, t in BASEGRID for b in (0.1, 0.25, 0.5)],
    'order2+nb': lambda: [(w, t, g, b) for w, t in BASEGRID for g in (-1.5, -4.0) for b in (0.1, 0.25)],
}
print('sub13e decoder, all subjects: %.4f (w 12, tau 0.25)' % acc(R, 12, 0.25))
report = {}
for name, grid_fn in VARIANTS.items():
    grid = grid_fn(); gains, fits = [], []
    for i in (0, 1):
        fitd, evd = R[R.user.isin(HALVES[1 - i])], R[R.user.isin(HALVES[i])]
        bw, bt = max(BASEGRID, key=lambda x: acc(fitd, x[0], x[1]))
        base = acc(evd, bw, bt)
        w, t, g, b = max(grid, key=lambda x: acc(fitd, *x))
        gains.append(acc(evd, w, t, g, b) - base); fits.append((w, t, g, b))
    ok = all(x > 1e-9 for x in gains)
    w, t, g, b = max(grid, key=lambda x: acc(R, *x))
    report[name] = dict(held_out_gains=gains, fits=fits, final=[w, t, g, b], acc_all=acc(R, w, t, g, b), accepted=ok)
    print('%-10s held-out %+.4f / %+.4f  %s | fits %s | all subjects %.4f with %s'
          % (name, gains[0], gains[1], 'ACCEPT' if ok else 'reject', fits, report[name]['acc_all'], (w, t, g, b)), flush=True)
json.dump(report, open('src/emo_fast2_report.json', 'w'), indent=1, default=str)
print('wrote src/emo_fast2_report.json')
