"""Temporal-order model for the `sequence` questions.

Train: a clip's sequence answer IS the chronological order, so under an equal-duration
assumption time slot k holds the k-th action. Learn P(action | slot descriptor).
Test: score all 4x4 (action, slot) pairs and solve the assignment; reading the assigned
actions off in slot order gives the predicted permutation.

Exact-match is what the competition scores, so that is the headline number; pairwise
order accuracy is reported as a diagnostic (0.5 = no signal).
"""
import pandas as pd, numpy as np, collections, itertools, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

NSEG = 4


def _ntxt(v):
    return ' '.join(str(v).strip().lower().split())


def load_segments():
    parts = []
    for s in ('train', 'test'):
        f = 'features/seg_%s.csv' % s
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
    if not parts:
        return {}
    d = pd.concat(parts, ignore_index=True)
    fc = [c for c in d.columns if c.startswith('f')]
    S = {}
    for p, g in d.groupby('path'):
        g = g.sort_values('seg')
        if len(g) == NSEG:
            S[p] = g[fc].values.astype(float)
    # 09-12: CUHKX_SEQ=skel | imu | imu+skel appends the organisers' non-visual time-slot
    # features (src/seq_nv_probe.py). Unset, S is exactly the motion segments above.
    kinds = os.environ.get('CUHKX_SEQ')
    if kinds:
        N = nv_slots(set(kinds.split('+')))
        dim = next(iter(N.values())).shape[1]
        for p in S:
            k = p if p in N else '/'.join(p.split('/')[:2])
            S[p] = np.concatenate([S[p], N.get(k, np.zeros((NSEG, dim)))], 1)
    return S


def nv_slots(kinds):
    """unit -> (NSEG, d): IMU acc/gyro magnitude per wearable and Skeleton energy / joint speed
    per time slot, as log1p(|x|) and relative to the unit's mean over the slots."""
    import re
    out = {}
    for f, kind in (('features/imu_feats.csv', 'imu'), ('features/skel_feats.csv', 'skel')):
        if kind not in kinds:
            continue
        d = pd.read_csv(f)
        base = [re.sub(r'seg0$', '', c) for c in d.columns if re.search(r'(^|_)v?seg0$', c)]
        A = np.stack([d[[b + 'seg%d' % k for b in base]].values.astype(float)
                      for k in range(NSEG)], 1)
        A = np.nan_to_num(A, nan=0.0)
        V = np.concatenate([np.log1p(np.abs(A)), A / (np.abs(A).mean(1, keepdims=True) + 1e-6)], 2)
        for u, v in zip(d.unit, V):
            out.setdefault(u, []).append(v)
    return {u: np.concatenate(v, 1) for u, v in out.items()}


def build_training(tr, S):
    """-> X (n,dim+1), y (action text); one row per (clip, slot)."""
    xs, ys = [], []
    for path, g in tr[tr.category == 'sequence'].groupby('path'):
        if path not in S:
            continue
        r = g.iloc[0]
        order = [str(r[L]).strip() for L in str(r.answer)]   # chronological
        for k in range(NSEG):
            xs.append(np.concatenate([S[path][k], [k / (NSEG - 1.0)]]))
            ys.append(order[k])
    return np.array(xs), np.array(ys)


def fit(tr, S, C=0.5, min_n=6):
    X, y = build_training(tr, S)
    if len(X) == 0:
        return None, None
    keep = {k for k, v in collections.Counter(y).items() if v >= min_n}
    m = np.array([v in keep for v in y])
    if m.sum() < 40:
        return None, None
    sc = StandardScaler().fit(X[m])
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(sc.transform(X[m]), y[m])
    return sc, clf


def predict(sc, clf, S, path, r, O):
    """Assignment decode -> permutation string."""
    if sc is None or path not in S:
        return None
    cls = list(clf.classes_)
    idx = {c: i for i, c in enumerate(cls)}
    Z = np.array([np.concatenate([S[path][k], [k / (NSEG - 1.0)]]) for k in range(NSEG)])
    LP = clf.predict_log_proba(sc.transform(Z))          # slots x classes
    cost = np.zeros((NSEG, len(O)))
    for k in range(NSEG):
        for j, L in enumerate(O):
            t = str(r[L]).strip()
            cost[k, j] = -LP[k, idx[t]] if t in idx else 12.0
    rr, cc = linear_sum_assignment(cost)
    slot_to_letter = {int(a): O[int(b)] for a, b in zip(rr, cc)}
    return ''.join(slot_to_letter[k] for k in range(NSEG))


def cv():
    tr, te = load()
    S = load_segments()
    print('segment descriptors for %d clips' % len(S))
    if not S:
        return
    users = sorted(tr.user.unique()); folds = [users[i::6] for i in range(6)]
    for C in (0.1, 0.3, 1.0, 3.0):
        exact = tot = 0; pair_ok = pair_tot = 0; fell_back = 0
        for f in folds:
            sc, clf = fit(tr[~tr.user.isin(f)], S, C)
            val = tr[(tr.user.isin(f)) & (tr.category == 'sequence')]
            for _, r in val.iterrows():
                O = opts(r)
                p = predict(sc, clf, S, r.path, r, O)
                tot += 1
                if p is None:
                    fell_back += 1
                    continue
                exact += (p == str(r.answer))
                truth = str(r.answer)
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = truth[i], truth[j]
                        pair_tot += 1
                        pair_ok += (p.index(a) < p.index(b))
        print('  C=%.1f  exact=%.4f (n=%d, chance 0.0417)   pairwise=%.4f (chance 0.5)   no-feats=%d'
              % (C, exact / tot, tot, pair_ok / max(1, pair_tot), fell_back))


if __name__ == '__main__':
    cv()


class SeqModel:
    """Packaged temporal-order model: gives a log-likelihood for any candidate permutation."""
    def __init__(self, tr, S, C=None):
        if C is None:                      # 09-12: CUHKX_C_SEQ, set with CUHKX_SEQ; unset = 1.0
            C = float(os.environ.get('CUHKX_C_SEQ', 1.0))
        self.S = S
        self.sc, self.clf = fit(tr, S, C)
        self.idx = {c: i for i, c in enumerate(self.clf.classes_)} if self.clf is not None else {}
        # Pairwise precedence between action texts, counted from training sequence answers.
        # The activities follow a largely scripted order (64% of action pairs occur one way
        # round >= 90% of the time); see src/eval_compact_decoder.py, src/prec_validate.py.
        self.before = collections.Counter()
        for _, r in tr[tr.category == 'sequence'].iterrows():
            o = [_ntxt(r[L]) for L in str(r.answer).strip()]
            for i in range(len(o)):
                for j in range(i + 1, len(o)):
                    self.before[(o[i], o[j])] += 1

    def slot_logp(self, path):
        if self.sc is None or path not in self.S:
            return None
        Z = np.array([np.concatenate([self.S[path][k], [k / (NSEG - 1.0)]]) for k in range(NSEG)])
        return self.clf.predict_log_proba(self.sc.transform(Z))

    def perm_scorer(self, path, r, O):
        """-> f(perm_tuple_of_letters) = sum_k log P(action at slot k), or None."""
        LP = self.slot_logp(path)
        if LP is None:
            return None
        txt = {L: str(r[L]).strip() for L in O}
        def score(perm):
            s = 0.0
            for k, L in enumerate(perm[:NSEG]):
                j = self.idx.get(txt.get(L))
                s += LP[k, j] if j is not None else -12.0
            return s
        return score

    def prec_scorer(self, r, O):
        """-> f(perm) = sum over ordered pairs of log P(a before b), smoothed (x+1.5)/(n+3)."""
        txt = {L: _ntxt(r[L]) for L in O}
        def score(perm):
            s = 0.0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    x = self.before[(txt[perm[i]], txt[perm[j]])]
                    y = self.before[(txt[perm[j]], txt[perm[i]])]
                    s += math.log((x + 1.5) / (x + y + 3.0))
            return s
        return score
