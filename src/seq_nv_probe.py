"""Standalone probe (09-12 night): do the organisers' IMU / Skeleton time-slot features help the
`sequence` slot model? Same 6-fold cross-subject protocol as `seq_model.cv()`.

Per slot k (of 4) it appends: IMU acc / gyro magnitude means per wearable (`<dev>_am_seg k`,
`<dev>_gm_segk`), Skeleton motion energy (`e_segk`) and per-joint speed (`j<j>_vsegk`). Each is
also given relative to the clip's mean over the 4 slots (slot k / mean), so the model sees
where in the clip the movement happens rather than how much there is. Missing units -> zeros.

    python src/seq_nv_probe.py
"""
import sys, os, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seq_model as SM
from solver import load, opts

NSEG = SM.NSEG


def nv_slots(kinds):
    out = {}
    for f, pre in (('features/imu_feats.csv', 'imu'), ('features/skel_feats.csv', 'skel')):
        if pre not in kinds:
            continue
        d = pd.read_csv(f)
        cols = [[c for c in d.columns if re.search(r'(^|_)v?seg%d$' % k, c)] for k in range(NSEG)]
        base = [re.sub(r'seg0$', '', c) for c in cols[0]]
        cols = [[b + 'seg%d' % k for b in base] for k in range(NSEG)]
        A = np.stack([d[c].values.astype(float) for c in cols], 1)      # units x slots x feat
        A = np.nan_to_num(A, nan=0.0)
        rel = A / (np.abs(A).mean(1, keepdims=True) + 1e-6)
        V = np.concatenate([np.log1p(np.abs(A)), rel], 2)
        for u, v in zip(d.unit, V):
            out.setdefault(u, []).append(v)
    return {u: np.concatenate(v, 1) for u, v in out.items()}


def augment(S, N, paths):
    dim = next(iter(N.values())).shape[1]
    out = {}
    for p in paths:
        if p not in S:
            continue
        key = p if p in N else p.split('/')[-1] if p.split('/')[-1] in N else None
        nv = N[key] if key else np.zeros((NSEG, dim))
        out[p] = np.concatenate([S[p], nv], 1)
    return out


def run(tr, S, label, Cs=(0.1, 0.3, 1.0)):
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    halves = [set(users[0::2]), set(users[1::2])]
    for C in Cs:
        hit = {0: [0, 0], 1: [0, 0]}; pair_ok = pair_tot = 0
        for f in folds:
            sc, clf = SM.fit(tr[~tr.user.isin(f)], S, C)
            val = tr[(tr.user.isin(f)) & (tr.category == 'sequence')]
            for _, r in val.iterrows():
                O = opts(r)
                p = SM.predict(sc, clf, S, r.path, r, O)
                h = 0 if r.user in halves[0] else 1
                hit[h][1] += 1
                if p is None:
                    continue
                t = str(r.answer)
                hit[h][0] += p == t
                for i in range(4):
                    for j in range(i + 1, 4):
                        pair_tot += 1; pair_ok += p.index(t[i]) < p.index(t[j])
        print('%-14s C=%.1f exact half0=%.4f half1=%.4f all=%.4f  pairwise=%.4f' % (
            label, C, hit[0][0] / hit[0][1], hit[1][0] / hit[1][1],
            (hit[0][0] + hit[1][0]) / (hit[0][1] + hit[1][1]), pair_ok / max(1, pair_tot)), flush=True)


if __name__ == '__main__':
    tr, te = load()
    S = SM.load_segments()
    paths = set(tr.path) | set(te.path)
    N_all = nv_slots({'imu', 'skel'})
    cov = np.mean([p in N_all or p.split('/')[-1] in N_all for p in tr[tr.category == 'sequence'].path])
    print('segment clips %d, nv units %d, sequence-clip nv coverage %.3f' % (len(S), len(N_all), cov))
    run(tr, S, 'motion')
    for kinds in (('imu',), ('skel',), ('imu', 'skel')):
        run(tr, augment(S, nv_slots(set(kinds)), paths), '+' + '+'.join(kinds))
    # non-visual only (no motion segments)
    Z = {p: np.zeros((NSEG, 0)) for p in S}
    run(tr, augment(Z, nv_slots({'imu', 'skel'}), paths), 'nv-only')
