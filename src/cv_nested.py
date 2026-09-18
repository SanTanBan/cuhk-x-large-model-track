"""Nested cross-subject CV.

Plain CV re-uses the same folds to fit models AND to pick ~13 fusion weights by
coordinate descent, so its number is optimistically biased. Here the subjects are split
in two: weights are tuned on one half (by inner CV) and scored on the other half, whose
subjects the tuner never saw. The outer number is what should be believed.

Usage:  python src/cv_nested.py [--clip] [--no-motion]
"""
import pandas as pd, numpy as np, collections, sys, os, json, argparse
sys.path.insert(0, 'src')
from solver import load, opts
from joint import Joint, clip_features
from fuse import answer_clip_fused, load_vlm, DEFAULT_W, clip_evidence
from motion_model import motion_table, MotionModel, HAUPresence
from seq_model import load_segments, SeqModel
from zeroshot import build as build_zs
from feats import load_clip_emb

ap = argparse.ArgumentParser()
ap.add_argument('--clip', action='store_true', help='add CLIP features')
ap.add_argument('--no-motion', action='store_true')
ap.add_argument('--passes', type=int, default=2)
A = ap.parse_args()

tr, te = load()
if A.clip:
    from feats import task_features
    TF = task_features()
    X, X_HARN, X_EMO, SEG = TF['X'], TF['X_harn'], TF['X_emo'], TF['SEG']
    ZS = build_zs(tr, te, load_clip_emb())
else:
    X, _ = motion_table('motion2')
    X_HARN = X_EMO = ZS = None
    SEG = load_segments()
print('features: %d clips x %d dims | segments: %d clips'
      % (len(X), len(next(iter(X.values()))) if X else 0, len(SEG)))
VLM = load_vlm('vlm/vlm_scores_train.csv')

TESTC = collections.Counter()
for p, g in te.groupby('path'):
    hs = (g.category == 'sequence').any()
    for _, r in g.iterrows():
        TESTC[(r['qt'], hs)] += 1


def make_folds(df, user_groups):
    """Fit models on everything except each group; return (M,MM,HP,SQ,clips) per group."""
    out = []
    for f in user_groups:
        trn = df[~df.user.isin(f)]
        M = Joint(trn)
        MM = MotionModel(trn, X, X_harn=X_HARN, X_emo=X_EMO) if X else None
        HP = HAUPresence(trn, X) if X else None
        SQ = SeqModel(trn, SEG) if SEG else None
        val = df[df.user.isin(f)]
        clips = [(g, clip_features(g, M), (g.category == 'sequence').any(),
                  clip_evidence(g, MM, HP, SQ, ZS)) for _, g in val.groupby('path')]
        out.append((M, clips))
    return out


def evaluate(folds, W):
    res = collections.defaultdict(lambda: [0, 0])
    for M, clips in folds:
        for g, feats, hs, ev in clips:
            a = answer_clip_fused(g, M, W, VLM, feats, None, None, None, ev)
            for _, r in g.iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1
                res[k][0] += (a[r.qa_id] == str(r.answer))
    num = den = 0
    for k, n in TESTC.items():
        acc = res[k][0] / res[k][1] if (k in res and res[k][1] >= 5) else .25
        num += n * acc; den += n
    return num / den, res


GRID = dict(
    w_mot_harn=[0, .25, .5, 1, 2, 3], w_mot_emotion=[0, .5, 1, 2, 3],
    w_mot_oi=[0, .25, .5, 1, 2], w_mot_comb=[0, .25, .5, 1, 2, 3],
    w_mot_bel=[0, .1, .25, .5, 1], w_seq=[0, .5, 1, 2],
    w_zs_harn=[0, .25, .5, 1, 2, 3], w_zs_comb=[0, .5, 1, 2, 3], w_zs_bel=[0, .1, .25, .5],
    w_pmi=[1, 2, 3, 5], w_ovl=[10, 16, 25, 40], w_sharp=[.25, .5, 1, 2],
    w_bel=[2, 3, 5, 8], w_bel_m=[.5, 1, 2], thr_m=[-2, -1.5, -1, -.5, 0],
)


def tune(folds, passes):
    W = dict(DEFAULT_W)
    cur = evaluate(folds, W)[0]
    for _ in range(passes):
        for k, vals in GRID.items():
            bv, bs = W[k], cur
            for v in vals:
                W2 = dict(W); W2[k] = v
                s = evaluate(folds, W2)[0]
                if s > bs + 1e-9:
                    bs, bv = s, v
            W[k] = bv; cur = bs
    return W, cur


users = sorted(tr.user.unique())
half = [users[0::2], users[1::2]]
outer_scores, defaults = [], []
for i in (0, 1):
    inner_users, outer_users = half[1 - i], half[i]
    inner_df = tr[tr.user.isin(inner_users)]
    inner_groups = [inner_users[j::3] for j in range(3)]
    inner_folds = make_folds(inner_df, inner_groups)
    W, inner_score = tune(inner_folds, A.passes)

    outer_folds = make_folds(tr, [outer_users])   # models fit on all other subjects
    out_tuned = evaluate(outer_folds, W)[0]
    out_default = evaluate(outer_folds, dict(DEFAULT_W))[0]
    outer_scores.append(out_tuned); defaults.append(out_default)
    print('half %d: inner(tuned)=%.4f   OUTER tuned=%.4f   OUTER untuned=%.4f'
          % (i, inner_score, out_tuned, out_default))
    print('        W = %s' % json.dumps({k: v for k, v in W.items() if not k.startswith('w_vlm')},
                                        sort_keys=True))

print('\nHONEST (nested) estimate: tuned=%.4f   untuned=%.4f   tuning gain=%+.4f'
      % (np.mean(outer_scores), np.mean(defaults),
         np.mean(outer_scores) - np.mean(defaults)))
