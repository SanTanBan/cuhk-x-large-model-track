"""Test-time build of the recording-session structures (09-13), on top of sub12.

    python src/build_session.py OUT.csv [--emo=raw|raw+sess|norm|norm+sess] [--w-emo=8] [--tau=1]
                                        [--seq-pool=ALPHA] [--base-sub=submissions/sub12_nv.csv]
                                        [--weights=src/best_nv_final_W.json]

emotion: sub12's fused score (option prior + VLM letters + IMU RandomForest) rebuilt on test.
  "norm" z-scores IMU features within recording sessions (train: training HAU sessions; test:
  test sessions). "+sess" decodes each test session with Viterbi under "consecutive clips never
  share the adverb".
sequence: sub12's permutation score (slot model * w_seq + precedence * w_prec) rebuilt on test;
  --seq-pool decodes each run of consecutive same-option-set clips once, from the summed slot
  evidence times ALPHA plus the prior (identical order 74/74 in training).
Sessions: test units sorted by recording time (features/unit_times.csv), split at gaps >= 30
min. Regression: with --emo=raw and no --seq-pool, the output must equal sub12 exactly. The
leak guard allows changes only in the categories rebuilt with an option switched on.
"""
import sys, os, json, collections, itertools, hashlib, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from solver import load, lo, opts
from joint import Joint
from fuse import _centred, load_vlm, DEFAULT_W
from feats import nonvisual, task_features
from seq_model import load_segments, SeqModel
from emo_decode import emissions as emo_emissions, viterbi1 as emo_v1, viterbi2 as emo_v2

A = sys.argv[1:]
OUT = A[0]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in A if a.startswith('--%s=' % k)), d)


EMO, SEQ_POOL = arg('emo', 'raw'), arg('seq-pool')
BASE_SUB, WFILE = arg('base-sub', 'submissions/sub12_nv.csv'), arg('weights', 'src/best_nv_final_W.json')
W0 = dict(DEFAULT_W); W0.update(json.load(open(WFILE)))
W_EMO, TAU = float(arg('w-emo', W0.get('w_mot_emotion', 8))), float(arg('tau', '1'))
tr, te = load()
base = pd.read_csv(BASE_SUB, dtype=str).set_index('qa_id').prediction
new = base.copy()
U = pd.read_csv('features/unit_times.csv', parse_dates=['t0', 't1']).set_index('unit')
te['unit'] = te.path.str.replace(chr(92), '/', regex=False).map(lambda q: '/'.join(q.split('/')[:2]))


def norm_txt(v):
    return ' '.join(str(v).strip().lower().split())


def chains(df, key=None):
    out, cur = [], []
    for r in df.sort_values('t0').itertuples():
        if pd.isna(r.t0):
            out.append([r]); continue
        if cur and ((r.t0 - cur[-1].t1).total_seconds() >= 1800 or (key and getattr(r, key) != getattr(cur[-1], key))):
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def session_norm(items_by_path, table):
    items = sorted([(t0, t1, p) for p, (t0, t1) in items_by_path.items() if p in table and pd.notna(t0)])
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


# ---------------- emotion ----------------
QT = 'emotion|HAU'
M = Joint(tr)
VT = load_vlm('vlm/vlm_scores_test.csv')
XKEYS = set(task_features('motion+clip')['X'])
IMU, _ = nonvisual('imu')
T = IMU
if EMO.startswith('norm'):
    hau_train = tr[tr.source == 'HAU'].path.unique()
    Ttr = session_norm({p: (U.t0.get(p), U.t1.get(p)) for p in hau_train}, IMU)
    tp = te[te.source == 'HAU'][['path', 'unit']].drop_duplicates()
    Tte = session_norm({r.path: (U.t0.get(r.unit), U.t1.get(r.unit)) for r in tp.itertuples()}, IMU)
    T = {**Ttr, **Tte}
e = tr[tr.qt == QT].copy()
e = e[e.path.isin(XKEYS)]
e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
keep = {k for k, v in collections.Counter(e.lab).items() if v >= 5}
e2 = e[e.lab.isin(keep) & e.path.isin(T)]
clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
clf.fit(np.array([T[p] for p in e2.path]), e2.lab.to_numpy())
rows = []
for _, r in te[te.qt == QT].iterrows():
    mot_e = {}
    if r.path in T:
        p = clf.predict_proba(T[r.path].reshape(1, -1))[0]
        mot_e = {c: float(np.log(max(v, 1e-9))) for c, v in zip(clf.classes_, p)}
    O = opts(r)
    keys = [str(r[L]).strip() for L in O]
    c = _centred(VT.get(r.qa_id, {}).get('letter', {}), O)
    mc = _centred({k: mot_e[k] for k in keys if k in mot_e}, keys) if mot_e else None
    prior = np.array([lo(M.p(QT, r[L])) for L in O])
    vlm = np.array([c[L] for L in O])
    imu = np.array([mc[k] if mc else 0.0 for k in keys])
    sc = prior + W0.get('w_vlm_emotion', 0.0) * vlm + W_EMO * imu
    rows.append(dict(qa_id=r.qa_id, O=O, adv=[' '.join(k.lower().split()) for k in keys], sc=sc,
                     prior=prior, vlm=vlm, imu=imu, t0=U.t0.get(r.unit), t1=U.t1.get(r.unit)))
E = pd.DataFrame(rows)
ind = {r.qa_id: r.O[int(np.argmax(r.sc))] for r in E.itertuples()}
if EMO == 'raw' and arg('w-emo') is None:
    mm = sum(ind[q] != base[q] for q in ind)
    print('REGRESSION emotion (raw, independent) vs %s: %d / %d differ' % (BASE_SUB, mm, len(ind)), flush=True)


def viterbi(chain, tau):
    em = []
    for r in chain:
        z = r.sc / tau
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
    return {chain[i].qa_id: chain[i].O[k] for i, k in enumerate(path)}


touched = set()
# --order2=GAMMA2 / --nbopt=BETA: the refined decoder validated in src/emo_fast_validate2.py
# (shared code in src/emo_decode.py). Unset, the original first-order decoder below runs unchanged.
# --lagw=REPORT:VARIANT (09-14): viterbiL with the lag weights of an ACCEPTED variant of
# src/emo_fast_validate3.py. They include lag 2, so --order2 is ignored. --beam=512.
ORDER2, NBOPT, LAGW = arg('order2'), arg('nbopt'), arg('lagw')
if LAGW:
    from emo_decode import viterbiL as emo_vL
    rep_f, var = LAGW.rsplit(':', 1)
    rep3 = json.load(open(rep_f))[var]
    assert rep3['accepted'] and all(g > 0 for g in rep3['held_out_gains']), 'lagw variant not accepted'
    LAGW = {int(k): float(v) for k, v in rep3['lagw'].items()}
    assert 2 in LAGW
    BEAM = int(arg('beam', '512'))
    print('emotion decoder: viterbiL %s (beam %d) from %s' % (LAGW, BEAM, rep_f), flush=True)
if EMO != 'raw' or arg('w-emo') is not None:
    ans = dict(ind)
    if EMO.endswith('+sess'):
        n_lag = n_self = n_tot = 0
        for ch in chains(E):
            if ORDER2 is None and NBOPT is None and not LAGW:
                ans.update(viterbi(ch, TAU))
            else:
                em = emo_emissions(ch, W_EMO, TAU, float(NBOPT or 0.0), W0.get('w_vlm_emotion', 0.0))
                if LAGW:
                    path, p2 = emo_vL(ch, em, LAGW, BEAM), emo_v2(ch, em, LAGW[2])
                    n_self += sum(a != b for a, b in zip(emo_vL(ch, em, {2: LAGW[2]}, BEAM), p2))
                    n_lag += sum(a != b for a, b in zip(path, p2)); n_tot += len(ch)
                else:
                    path = emo_v1(ch, em) if ORDER2 is None else emo_v2(ch, em, float(ORDER2))
                ans.update({r.qa_id: r.O[k] for r, k in zip(ch, path)})
        if LAGW:
            print('REGRESSION viterbiL({2: g2}) vs viterbi2 on test: %d / %d differ | lag terms change %d answers'
                  % (n_self, n_tot, n_lag), flush=True)
    for q, a in ans.items():
        new[q] = a
    touched.add(QT)
    rep = 0
    Et = E.sort_values('t0').reset_index(drop=True)
    for a_, b_ in zip(Et.itertuples(), list(Et.itertuples())[1:]):
        if pd.notna(a_.t0) and pd.notna(b_.t0) and (b_.t0 - a_.t1).total_seconds() < 1800:
            rep += a_.adv[a_.O.index(ans[a_.qa_id])] == b_.adv[b_.O.index(ans[b_.qa_id])]
    print('emotion %s (w %.2f, tau %.2f): %d answers changed vs base | consecutive repeats now %d' % (
        EMO, W_EMO, TAU, sum(new[q] != base[q] for q in ind), rep), flush=True)

# ---------------- sequence ----------------
QS = 'sequence|HAU'
SQ = SeqModel(tr, load_segments())
ws, wp = W0.get('w_seq', 1.0), W0.get('w_prec', 0.0)
srows = []
for _, r in te[te.qt == QS].iterrows():
    O = opts(r)
    f, h = SQ.perm_scorer(r.path, r, O), SQ.prec_scorer(r, O)
    perms = list(itertools.permutations(O))
    tot = [(ws * f(p) if f else 0.0) + (wp * h(p) if (h and wp) else 0.0) for p in perms]
    txt = {L: norm_txt(r[L]) for L in O}
    srows.append(dict(qa_id=r.qa_id, txt=txt, set=frozenset(txt.values()), ind=''.join(perms[int(np.argmax(tot))]),
                      slot={tuple(txt[L] for L in p): (f(p) if f else 0.0) for p in perms},
                      pr={tuple(txt[L] for L in p): (h(p) if h else 0.0) for p in perms},
                      t0=U.t0.get(r.unit), t1=U.t1.get(r.unit)))
S = pd.DataFrame(srows)
if SEQ_POOL is None:
    mm = sum(r.ind != base[r.qa_id] for r in S.itertuples())
    print('REGRESSION sequence (independent) vs %s: %d / %d differ' % (BASE_SUB, mm, len(S)), flush=True)
else:
    alpha, nrun = float(SEQ_POOL), 0
    for run in chains(S, key='set'):
        if len(run) == 1:
            new[run[0].qa_id] = run[0].ind
            continue
        nrun += 1
        orders = list(run[0].pr)
        best = max(orders, key=lambda o: wp * run[0].pr[o] + alpha * sum(ws * r.slot.get(o, 0.0) for r in run))
        for r in run:
            inv = {t: L for L, t in r.txt.items()}
            new[r.qa_id] = ''.join(inv[t] for t in best)
    touched.add(QS)
    print('sequence pooled (alpha %.2f): %d runs, %d answers changed vs base' % (
        alpha, nrun, sum(new[q] != base[q] for q in S.qa_id)), flush=True)

# ---------------- write + leak guard ----------------
out = pd.DataFrame({'qa_id': te.qa_id, 'prediction': te.qa_id.map(new)})
assert out.prediction.notna().all() and len(out) == 682
qt = te.set_index('qa_id').qt
changed = qt[out.set_index('qa_id').prediction != base.reindex(out.qa_id).values]
ok = set(changed) <= touched
print('LEAK GUARD vs base: %d changed %s | allowed %s -> %s' % (
    len(changed), changed.value_counts().to_dict(), sorted(touched), 'OK' if ok else 'LEAK'))
if ok:
    out.to_csv(OUT, index=False)
    print('wrote %s md5 %s' % (OUT, hashlib.md5(open(OUT, 'rb').read()).hexdigest()))
