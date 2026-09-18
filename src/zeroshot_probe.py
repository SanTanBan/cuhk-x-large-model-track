"""Does CLIP zero-shot add to the supervised models?

Zero-shot uses no training labels at all, so its errors are uncorrelated with the
supervised classifier's -- which is exactly what makes an ensemble worth testing even
though it is individually weaker (0.564 vs 0.690 on HARn).

Also tests zero-shot as HAU action-presence evidence, which is the larger prize: presence
drives `combination`, `multi` and `single|HAU` (427 test questions).
"""
import numpy as np, collections, sys, math
sys.path.insert(0, 'src')
from solver import load, opts, lo
from joint import Joint, clip_features, sm
from feats import load_clip_emb, task_features
from motion_model import MotionModel
from clip_zeroshot import text_embeddings

tr, te = load()
C = load_clip_emb()
TF = task_features()

harn_phrases = sorted({str(r[L]).strip() for df in (tr, te)
                       for _, r in df[df.qt == 'single|HARn'].iterrows() for L in opts(r)})
hau_actions = sorted({a.strip() for _, r in tr[tr.category == 'combination'].iterrows()
                      for L in opts(r) for a in str(r[L]).split(',')})
print('encoding %d HARn phrases + %d HAU actions' % (len(harn_phrases), len(hau_actions)),
      flush=True)
T = text_embeddings(harn_phrases + [a.lower() for a in hau_actions])
TA = {a: T[a.lower()] for a in hau_actions}


def zs(path, phrase_vec):
    E = C.get(path)
    return float((E @ phrase_vec).mean()) if E is not None else 0.0


users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]

# ---------- 1. HARn: supervised + zero-shot ----------
REC = []
for f in folds:
    trn = tr[~tr.user.isin(f)]
    MM = MotionModel(trn, TF['X'], X_harn=TF['X_harn'], X_emo=TF['X_emo'])
    for _, r in tr[(tr.user.isin(f)) & (tr.qt == 'single|HARn')].iterrows():
        if r.path not in C:
            continue
        O = opts(r)
        sup = MM.harn_action_logp(r.path)
        texts = [str(r[L]).strip() for L in O]
        m = float(np.mean([sup.get(t, -9) for t in texts]))
        REC.append(([sup.get(t, -9) - m for t in texts],
                    [zs(r.path, T[t]) for t in texts], O.index(str(r.answer))))
print('\nHARn: supervised + zero-shot (n=%d)' % len(REC))
for w in (0, 2, 5, 10, 20, 40, 80):
    ok = sum(1 for s, z, ci in REC
             if int(np.argmax([s[i] + w * z[i] for i in range(len(s))])) == ci)
    print('   w_zs=%3d  acc=%.4f' % (w, ok / len(REC)))

# ---------- 2. HAU presence: zero-shot as combination evidence ----------
REC2 = []
for f in folds:
    trn = tr[~tr.user.isin(f)]
    M = Joint(trn)
    for p, g in tr[tr.user.isin(f)].groupby('path'):
        cb = g[g.category == 'combination']
        if len(cb) == 0 or p not in C:
            continue
        K, comb = clip_features(g, M)
        base = [comb['marg'][i] + 3.0 * comb['pmi'][i] + 25.0 * comb['ovl'][i]
                for i in range(len(comb['O']))]
        zsc = [float(np.mean([zs(p, TA[a]) for a in comb['sets'][i] if a in TA] or [0.0]))
               for i in range(len(comb['O']))]
        REC2.append((base, zsc, comb['O'].index(str(cb.iloc[0].answer)),
                     (g.category == 'sequence').any()))
print('\nHAU combination: structural + zero-shot presence (n=%d)' % len(REC2))
for w in (0, 5, 10, 20, 40, 80, 150):
    res = collections.defaultdict(lambda: [0, 0])
    for base, z, ci, hs in REC2:
        k = int(np.argmax([base[i] + w * z[i] for i in range(len(base))]))
        res[hs][1] += 1; res[hs][0] += (k == ci)
    ns, sq = res[False], res[True]
    print('   w_zs=%3d  no-seq=%.4f (n=%d)  seq=%.4f (n=%d)'
          % (w, ns[0] / max(1, ns[1]), ns[1], sq[0] / max(1, sq[1]), sq[1]))
