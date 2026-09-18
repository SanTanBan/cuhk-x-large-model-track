"""Combined clip descriptors: frame-difference motion + CLIP semantics.

MotionModel / HAUPresence / SeqModel all take a plain {path: vector} (or {path: (4,d)})
mapping, so richer features drop straight in.

CLIP embeddings are 512-d per frame; mean+std pooling gives 1024, which badly overfits
~800 training clips, so they are PCA-reduced (fit on all clips -- unsupervised, so no
label leakage across the CV split) before being concatenated with the motion features.
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_model import motion_table
from seq_model import load_segments
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NSEG = 4


def load_clip_emb():
    """-> {path: (nframes, 512)} or {}"""
    out = {}
    for s in ('train', 'test'):
        f = 'features/clip_%s.npz' % s
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        for p, e in zip(z['paths'], z['emb']):
            out[str(p)] = e.astype(np.float32)
    return out


def _pca(mat, n):
    n = min(n, mat.shape[0], mat.shape[1])
    p = PCA(n_components=n, random_state=0).fit(mat)
    return p


def combined_X(n_pca=96, use_clip=True, use_motion=True):
    """Clip-level feature vector per path."""
    M, _ = motion_table('motion2') if use_motion else ({}, [])
    C = load_clip_emb() if use_clip else {}
    if not C:
        return M
    paths = sorted(C)
    pooled = np.array([np.concatenate([C[p].mean(0), C[p].std(0)]) for p in paths])
    sc = StandardScaler().fit(pooled)
    Z = sc.transform(pooled)
    R = _pca(Z, n_pca).transform(Z)
    clip_of = {path: R[i] for i, path in enumerate(paths)}
    # Every clip gets the SAME layout [motion | clip-pca], zero-padded where a modality is
    # missing -- otherwise sklearn sees ragged rows.
    mdim = len(next(iter(M.values()))) if M else 0
    cdim = R.shape[1]
    out = {}
    for path in set(clip_of) | set(M):
        out[path] = np.concatenate([M.get(path, np.zeros(mdim)),
                                    clip_of.get(path, np.zeros(cdim))])
    return out


def combined_S(n_pca=48, use_clip=True, use_motion=True):
    """Per-time-slot features: (NSEG, d) per path, for the temporal-order model."""
    S = load_segments() if use_motion else {}
    C = load_clip_emb() if use_clip else {}
    if not C:
        return S
    paths = sorted(C)
    # split the 8 sampled frames into NSEG contiguous slots and mean-pool each
    slots = {}
    for path in paths:
        e = C[path]
        b = np.linspace(0, len(e), NSEG + 1).astype(int)
        slots[path] = np.stack([e[b[k]:max(b[k] + 1, b[k + 1])].mean(0) for k in range(NSEG)])
    flat = np.concatenate([slots[p] for p in paths])
    sc = StandardScaler().fit(flat)
    pc = _pca(sc.transform(flat), n_pca)
    sdim = next(iter(S.values())).shape[1] if S else 0
    cdim = pc.n_components_
    out = {}
    for path in set(paths) | set(S or {}):
        m = S[path] if (S and path in S) else np.zeros((NSEG, sdim))
        c = (pc.transform(sc.transform(slots[path])) if path in slots
             else np.zeros((NSEG, cdim)))
        out[path] = np.concatenate([m, c], axis=1)
    return out


if __name__ == '__main__':
    X = combined_X(); S = combined_S()
    print('combined_X: %d clips x %d dims' % (len(X), len(next(iter(X.values())))))
    print('combined_S: %d clips x %s' % (len(S), (next(iter(S.values())).shape,)))


def clip_pooled(n_pca=64):
    """{path: mean|std CLIP}, optionally PCA-reduced.

    `n_pca=0` keeps the full 1024 dims. That matters: PCA to 64 costs the HARn action
    classifier 0.12 accuracy (0.571 vs 0.690), because the discriminative directions for
    44 fine-grained actions are not the top-variance ones. Reduce only where the consumer
    is dimension-sensitive (the presence model, fit per action on ~800 clips).
    """
    C = load_clip_emb()
    if not C:
        return {}
    paths = sorted(C)
    Z = StandardScaler().fit_transform(
        np.array([np.concatenate([C[p].mean(0), C[p].std(0)]) for p in paths]))
    if n_pca:
        Z = _pca(Z, n_pca).transform(Z)
    return {p: Z[i] for i, p in enumerate(paths)}


def nonvisual(name):
    """{path: vector} from features/<name>_feats.csv (`imu` / `skel`, the organisers' non-visual
    supplement, src/nonvisual_feats.py), keyed like the other tables: training clip dirs,
    test Depth paths. NaNs -> column medians; returns (table, median vector)."""
    import pandas as pd
    d = pd.read_csv(os.path.join('features', '%s_feats.csv' % name), index_col=0)
    d = d.loc[:, d.notna().any()]
    med = d.median()
    d = d.fillna(med)
    te = pd.read_csv(os.path.join('data', 'test_qa.csv'), usecols=['path'])
    unit2test = {}
    for p in te.path.unique():
        q = str(p).replace(chr(92), '/')
        unit2test['/'.join(q.split('/')[:2])] = p
    return {unit2test.get(k, k): v for k, v in zip(d.index, d.to_numpy(float))}, med.to_numpy(float)


def _concat(parts):
    """parts = [(table, fill)]: a key missing from a table with fill=None drops the clip;
    with a fill vector (non-visual tables next to a visual one) the fill stands in."""
    out = {}
    for k in set().union(*[set(t) for t, _ in parts]):
        vs = []
        for t, fill in parts:
            if k in t:
                vs.append(t[k])
            elif fill is not None:
                vs.append(fill)
            else:
                break
        else:
            out[k] = np.concatenate(vs)
    return out


def _table_set(spec, M, Cfull):
    """'motion', 'clip', 'imu', 'skel' or '+'-joined; non-visual parts are median-filled only
    when a visual part is present (alone, a clip without the sensor simply has no evidence)."""
    names = spec.split('+')
    visual = any(n in ('motion', 'clip') for n in names)
    parts = []
    for n in names:
        if n == 'motion':
            parts.append((M, None))
        elif n == 'clip':
            parts.append((Cfull, None))
        else:
            t, med = nonvisual(n)
            parts.append((t, med if visual else None))
    return _concat(parts)


def task_features(presence='motion+clip'):
    """Per-task feature tables, chosen by measured accuracy (see src/clip_probe.py,
    src/presence_probe.py):

        single|HARn / object_interaction -> full-dim CLIP (0.690 / 0.827 vs 0.624 / 0.774)
        emotion                          -> motion (0.394 vs 0.318 for CLIP)
        sequence                         -> motion segments (0.276 vs 0.214 for CLIP)
        HAU action presence              -> motion + PCA-64 CLIP (0.873 combination)

    09-11: the environment variables CUHKX_EMO and CUHKX_HARN (e.g. `imu+motion`,
    `clip+skel`; see src/nv_probe.py) swap in the non-visual tables for those two tasks.
    Unset, the tables are exactly the ones above.
    """
    M, _ = motion_table('motion2')
    Cfull = clip_pooled(0)      # 1024-d, for the fine-grained action / object classifier
    Cp = clip_pooled(64)        # reduced, for the per-action presence models
    both = {}
    mdim = len(next(iter(M.values()))) if M else 0
    cdim = len(next(iter(Cp.values()))) if Cp else 0
    for p in set(M) | set(Cp):
        both[p] = np.concatenate([M.get(p, np.zeros(mdim)), Cp.get(p, np.zeros(cdim))])
    pres = {'motion': M, 'clip': Cp, 'motion+clip': both}[presence]
    # 09-13: CUHKX_PRES=imu | skel | imu+skel appends the non-visual tables (median-filled) to
    # the HAU presence table (src/pres_nv_validate.py). Unset, the table is unchanged.
    if os.environ.get('CUHKX_PRES'):
        pres = _concat([(pres, None)] + [nonvisual(n) for n in os.environ['CUHKX_PRES'].split('+')])
    X_harn, X_emo = Cfull or M, M or Cfull
    emo, harn = os.environ.get('CUHKX_EMO'), os.environ.get('CUHKX_HARN')
    if emo:
        X_emo = _table_set(emo, M, Cfull)
    if harn:
        X_harn = _table_set(harn, M, Cfull)
    return dict(X=pres, X_harn=X_harn, X_emo=X_emo, SEG=load_segments())
