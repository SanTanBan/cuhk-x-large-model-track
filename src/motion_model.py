"""Fitted motion models, packaged for use inside the joint solver.

  harn_action   44-way classifier over HARn actions (labels come free from the training
                path `HARn/<action>/<user>/<trial>`)
  emotion       classifier over manner adverbs
  object        derived from harn_action via P(object | action), which is also free:
                every training `object_interaction` question sits on a clip whose path
                names the action.
"""
import pandas as pd, numpy as np, collections, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import opts
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DROP = {'path', 'n_frames'}


def motion_table(prefix='motion2'):
    parts = []
    for s in ('train', 'test'):
        f = 'features/%s_%s.csv' % (prefix, s)
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
    if not parts:
        return {}, []
    m = pd.concat(parts, ignore_index=True)
    cols = [c for c in m.columns if c not in DROP]
    for c in ('dur', 'motion', 'motion_sd', 'motion_p90', 'motion_p10'):
        if c in m: m[c] = np.log1p(m[c])
    m[cols] = m[cols].replace([np.inf, -np.inf], np.nan).fillna(m[cols].median())
    return {r.path: np.array([getattr(r, c) for c in cols], dtype=float)
            for r in m.itertuples(index=False)}, cols


_CLIP_FULL = None


def _clip_full():
    """Full-dim pooled CLIP table, loaded once (for CUHKX_OBJ_CLIP)."""
    global _CLIP_FULL
    if _CLIP_FULL is None:
        from feats import clip_pooled
        _CLIP_FULL = clip_pooled(0)
    return _CLIP_FULL


class MotionModel:
    """Different sub-tasks want different features: CLIP semantics identify *what* an
    action or object is, frame-difference dynamics identify *how* someone moves. Pass
    X_harn / X_emo separately to use the right one for each."""
    def __init__(self, tr, X, C_harn=1.0, C_emo=1.0, min_n=5, X_harn=None, X_emo=None):
        # 09-11 switches for the non-visual features, all off by default: CUHKX_C_HARN /
        # CUHKX_C_EMO override C; CUHKX_CLF_HARN / CUHKX_CLF_EMO = 'rf' use a random forest;
        # CUHKX_HARN_UNITS=1 also trains the HARn action classifier on every HARn unit in
        # X_harn whose user is in this training set (the action is the folder name), not only
        # on the clips that carry a question (src/nv_probe2.py).
        import os
        env = os.environ
        C_harn = float(env.get('CUHKX_C_HARN', C_harn))
        C_emo = float(env.get('CUHKX_C_EMO', C_emo))
        kind_h, kind_e = env.get('CUHKX_CLF_HARN', 'lr'), env.get('CUHKX_CLF_EMO', 'lr')
        self.X = X
        self.X_h = X_harn if X_harn is not None else X
        self.X_e = X_emo if X_emo is not None else X
        # ---- HARn action classifier ----
        h = tr[tr.qt == 'single|HARn'].copy()
        h = h[h.path.isin(X)]
        h['action'] = h.path.str.split('/').str[1]
        self.act2txt = {}
        for _, r in h.iterrows():
            self.act2txt[r.action] = str(r[str(r.answer)]).strip()
        h = h[h.path.isin(self.X_h)]
        paths, acts = list(h.path), list(h.action)
        if env.get('CUHKX_HARN_UNITS') == '1':
            users, have = set(tr.user), set(paths)
            for p in self.X_h:
                q = p.split('/')
                if (len(q) == 4 and q[0] == 'HARn' and q[2].startswith('user') and p not in have
                        and q[2][4:].isdigit() and int(q[2][4:]) in users):
                    paths.append(p); acts.append(q[1])
        self.h_sc, self.h_clf = self._fit(paths, acts, C_harn, self.X_h, kind_h)
        # CUHKX_OBJ_CLIP=1: objects keep today's CLIP action classifier (question clips, C=1).
        # CLIP sees the object in the scene; a pose only says what the body does. In
        # src/nv_probe.py objects score 0.865 via CLIP vs 0.835 via Skeleton.
        self.o_sc = self.o_clf = None
        if env.get('CUHKX_OBJ_CLIP') == '1':
            self.X_o = _clip_full()
            ho = tr[tr.qt == 'single|HARn'].copy()
            ho = ho[ho.path.isin(X) & ho.path.isin(self.X_o)]
            self.o_sc, self.o_clf = self._fit(list(ho.path), list(ho.path.str.split('/').str[1]),
                                              1.0, self.X_o)
        # ---- emotion adverb classifier ----
        e = tr[tr.qt == 'emotion|HAU'].copy()
        e = e[e.path.isin(X)]
        e['lab'] = [str(r[str(r.answer)]).strip() for _, r in e.iterrows()]
        keep = {k for k, v in collections.Counter(e.lab).items() if v >= min_n}
        e2 = e[e.lab.isin(keep)]
        e2 = e2[e2.path.isin(self.X_e)]
        self.e_sc, self.e_clf = self._fit(list(e2.path), list(e2.lab), C_emo, self.X_e, kind_e)
        # ---- P(object | action), read off the HARn paths ----
        oi = tr[tr.qt == 'object_interaction|HARn'].copy()
        oi['action'] = oi.path.str.split('/').str[1]
        self.obj_given_act = collections.defaultdict(collections.Counter)
        for _, r in oi.iterrows():
            self.obj_given_act[r.action][str(r[str(r.answer)]).strip()] += 1

    def _fit(self, paths, labels, C, X=None, kind='lr'):
        X = X if X is not None else self.X
        if not paths:
            return None, None
        xs = np.array([X[p] for p in paths]); ys = np.array(labels)
        if kind == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                         random_state=0)
            return None, clf.fit(xs, ys)
        sc = StandardScaler().fit(xs)
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(sc.transform(xs), ys)
        return sc, clf

    def _lp(self, sc, clf, path, X=None):
        X = X if X is not None else self.X
        if clf is None or path not in X:
            return {}
        x = X[path].reshape(1, -1)
        if sc is None:        # random forest: its probabilities can be exactly 0
            lp = np.log(np.maximum(clf.predict_proba(x)[0], 1e-9))
        else:
            lp = clf.predict_log_proba(sc.transform(x))[0]
        return {c: float(lp[i]) for i, c in enumerate(clf.classes_)}

    def harn_action_logp(self, path):
        """-> {option text: log P}, collapsing action folders that share an option string."""
        lp = self._lp(self.h_sc, self.h_clf, path, self.X_h)
        out = {}
        for a, v in lp.items():
            t = self.act2txt.get(a)
            if t is not None:
                out[t] = max(out.get(t, -1e9), v)
        return out

    def emotion_logp(self, path):
        return self._lp(self.e_sc, self.e_clf, path, self.X_e)

    def object_logp(self, path):
        """P(object) = sum_action P(action | video) P(object | action)."""
        if self.o_clf is not None:
            lp = self._lp(self.o_sc, self.o_clf, path, self.X_o)
        else:
            lp = self._lp(self.h_sc, self.h_clf, path, self.X_h)
        if not lp:
            return {}
        acc = collections.defaultdict(float)
        for a, v in lp.items():
            c = self.obj_given_act.get(a)
            if not c:
                continue
            tot = sum(c.values())
            for o, n in c.items():
                acc[o] += math.exp(v) * (n / tot)
        return {o: math.log(max(p, 1e-9)) for o, p in acc.items()}


class HAUPresence:
    """One-vs-rest P(action occurs in this clip | motion features), for HAU clips.

    Labels are free: a clip's `combination` answer names its action set, and its
    `sequence` options are four more known-present actions. Negatives are "not in that
    set", which is noisy but serviceable.
    """
    def __init__(self, tr, X, C=1.0, min_pos=10):
        from solver import acts_of
        C = float(os.environ.get('CUHKX_C_PRES', C))     # 09-13, set with CUHKX_PRES; unset = 1.0
        clip_acts = {}
        for path, g in tr[tr.source == 'HAU'].groupby('path'):
            if path not in X:
                continue
            S = set()
            cb = g[g.category == 'combination']
            if len(cb):
                S |= set(a.strip() for a in acts_of(cb.iloc[0][str(cb.iloc[0].answer)]))
            sq = g[g.category == 'sequence']
            if len(sq):
                S |= {str(sq.iloc[0][L]).strip() for L in opts(sq.iloc[0])}
            if S:
                clip_acts[path] = S
        self.X = X
        paths = sorted(clip_acts)
        if not paths:
            self.sc, self.models = None, {}
            return
        Xm = np.array([X[p] for p in paths])
        self.sc = StandardScaler().fit(Xm)
        Z = self.sc.transform(Xm)
        vocab = sorted({a for S in clip_acts.values() for a in S})
        self.models = {}
        for a in vocab:
            y = np.array([1 if a in clip_acts[p] else 0 for p in paths])
            if y.sum() < min_pos or (len(y) - y.sum()) < min_pos:
                continue
            clf = LogisticRegression(max_iter=2000, C=C)
            clf.fit(Z, y)
            self.models[a] = clf
        self.base = {a: float(np.mean([1 if a in clip_acts[p] else 0 for p in paths]))
                     for a in vocab}

    def logodds(self, path):
        """-> {action: log p/(1-p) - log base/(1-base)}  (centred on the action's prior)"""
        if self.sc is None or path not in self.X or not self.models:
            return {}
        z = self.sc.transform(self.X[path].reshape(1, -1))
        out = {}
        for a, clf in self.models.items():
            p = float(clf.predict_proba(z)[0, 1])
            p = min(max(p, 1e-4), 1 - 1e-4)
            b = min(max(self.base.get(a, 0.25), 1e-4), 1 - 1e-4)
            out[a] = math.log(p / (1 - p)) - math.log(b / (1 - b))
        return out
