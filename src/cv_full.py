"""Full cross-subject CV: structural model + motion models (+ VLM evidence if present).

Tunes every weight by coordinate descent against the test set's own stratum mix.
"""
import pandas as pd, numpy as np, collections, sys, os, json
sys.path.insert(0, 'src')
from solver import load, opts
from joint import Joint, clip_features
from fuse import answer_clip_fused, load_vlm, DEFAULT_W, clip_evidence
from motion_model import motion_table, MotionModel, HAUPresence
from seq_model import load_segments, SeqModel
from zeroshot import build as build_zs
from feats import load_clip_emb

tr, te = load()
USE_CLIP = '--clip' in sys.argv
PRESENCE = next((a.split('=')[1] for a in sys.argv if a.startswith('--presence=')),
                'motion+clip')
if USE_CLIP:
    from feats import task_features
    TF = task_features(PRESENCE)
    X, X_HARN, X_EMO = TF['X'], TF['X_harn'], TF['X_emo']
    print('per-task features: presence=%s(%d) harn=CLIP(%d) emotion=motion(%d)'
          % (PRESENCE, len(next(iter(X.values()))), len(next(iter(X_HARN.values()))),
             len(next(iter(X_EMO.values())))))
else:
    X, cols = motion_table('motion2')
    if not X:
        X, cols = motion_table('motion')
    X_HARN = X_EMO = None
    print('motion features: %d clips x %d dims' % (len(X), len(cols)))
SEG = load_segments()
print('segment descriptors for %d clips' % len(SEG))
ZS = build_zs(tr, te, load_clip_emb()) if USE_CLIP else None
VLM_DIR = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--vlm-dir=')), 'vlm')
VLM = load_vlm(os.path.join(VLM_DIR, 'vlm_scores_train.csv'))
# The VLM scores only a sample of training clips. With --only-vlm the objective (and the
# report) is restricted to those clips, which is the condition the test set is in -- every
# test clip has VLM evidence. Without it the gain is diluted by ~1000 clips the VLM never saw.
ONLY_VLM = '--only-vlm' in sys.argv
COVERED = set(VLM)
COVERED_PATHS = set(tr[tr.qa_id.isin(COVERED)].path)
print('VLM training evidence for %d questions' % len(VLM))

TESTC = collections.Counter()
for p, g in te.groupby('path'):
    hs = (g.category == 'sequence').any()
    for _, r in g.iterrows():
        TESTC[(r['qt'], hs)] += 1

users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]
FOLD = []
for f in folds:
    trn = tr[~tr.user.isin(f)]
    M = Joint(trn)
    MM = MotionModel(trn, X, X_harn=X_HARN, X_emo=X_EMO) if X else None
    HP = HAUPresence(trn, X) if X else None
    SQ = SeqModel(trn, SEG) if SEG else None
    val = tr[tr.user.isin(f)]
    if ONLY_VLM:   # ev() skips uncovered clips anyway -- don't pay to prepare them
        val = val[val.path.isin(COVERED_PATHS)]
    clips = [(g, clip_features(g, M), (g.category == 'sequence').any(),
              clip_evidence(g, MM, HP, SQ, ZS))
             for _, g in val.groupby('path')]
    FOLD.append((M, MM, HP, SQ, clips))
print('prepared %d folds' % len(FOLD))


def ev(W, verbose=False):
    res = collections.defaultdict(lambda: [0, 0])
    for M, MM, HP, SQ, clips in FOLD:
        for g, feats, hs, ev in clips:
            if ONLY_VLM and not any(q in COVERED for q in g.qa_id):
                continue
            a = answer_clip_fused(g, M, W, VLM, feats, MM, HP, SQ, ev)
            for _, r in g.iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1
                res[k][0] += (a[r.qa_id] == str(r.answer))
    num = den = 0
    for k, n in TESTC.items():
        acc = res[k][0] / res[k][1] if (k in res and res[k][1] >= 5) else .25
        num += n * acc; den += n
    if verbose:
        for k, v in sorted(res.items()):
            print('   %-40s acc=%.4f n_val=%5d n_test=%d'
                  % (str(k), v[0] / v[1], v[1], TESTC.get(k, 0)))
    return num / den, res


GRID = dict(
    w_mot_harn=[0, .25, .5, 1, 1.5, 2, 3, 5],
    w_mot_emotion=[0, .25, .5, .75, 1, 1.5, 2, 3],
    w_mot_oi=[0, .25, .5, 1, 2, 3],
    w_mot_comb=[0, .25, .5, 1, 2, 3, 5],
    w_mot_bel=[0, .1, .25, .5, 1, 2],
    w_seq=[0, .5, 1, 2],
    w_zs_harn=[0, .25, .5, 1, 2, 3, 5], w_zs_comb=[0, .25, .5, 1, 2, 3],
    w_zs_bel=[0, .1, .25, .5, 1],
    w_pmi=[1, 2, 3, 5, 8], w_ovl=[10, 16, 25, 40], w_sharp=[.25, .5, 1, 2],
    w_bel=[2, 3, 5, 8], w_bel_m=[.25, .5, 1, 2, 3],
    thr_m=[-5, -4, -3, -2.5, -2, -1.5, -1, -.5, 0],
)
if VLM:
    GRID.update(w_vlm_single=[0, .5, 1, 2, 3, 5], w_vlm_single_harn=[0, .5, 1, 2, 3, 5, 8],
                w_vlm_emotion=[0, .5, 1, 2, 3, 5], w_vlm_oi=[0, .5, 1, 2, 3],
                w_vlm_comb=[0, .5, 1, 2, 3], w_vlm_pres=[0, .1, .25, .5, 1, 2],
                w_gen=[0, 1, 2, 4], w_vlm_pair=[0, .25, .5, 1, 2, 4])

if __name__ == '__main__':
    W = dict(DEFAULT_W)
    VLM_ONLY_TUNE = '--tune-vlm-only' in sys.argv
    if VLM_ONLY_TUNE:
        # The VLM scored only ~320 of 1333 training clips. Keep every non-VLM weight at its
        # optimum on ALL clips (4x the data) and search only the VLM weights, so the small
        # covered subset can't drag the proven weights around.
        # Exception: thr_m. Relative weights are safe to freeze (tuning a VLM weight against
        # fixed others tunes the ratio), but thr_m is an ABSOLUTE cut-off on a score the VLM
        # presence term is now added to -- its optimum moves when that term appears.
        W.update(json.load(open('src/best_full_W.json')))
        W['w_vlm_pair'] = 0.0; W['w_gen'] = 0.0      # start exactly at the current model
        for k in list(GRID):
            if not (k.startswith('w_vlm') or k in ('w_gen', 'thr_m')):
                del GRID[k]
        print('tuning only:', sorted(GRID))
    cur = ev(W)[0]
    print('\nstart (VLM weights = 0, i.e. the current model): %.4f' % cur)
    for it in range(4):
        improved = False
        for k, vals in GRID.items():
            bv, bs = W[k], cur
            for v in vals:
                W2 = dict(W); W2[k] = v
                s = ev(W2)[0]
                if s > bs + 1e-9:
                    bs, bv = s, v
            if bv != W[k]:
                improved = True
            W[k] = bv; cur = bs
        print('  pass %d: %.4f' % (it + 1, cur))
        if not improved:
            break
    print('\nEXPECTED PRIVATE LB = %.4f' % cur)
    print('W = %s\n' % json.dumps(W, sort_keys=True))
    ev(W, verbose=True)
    out_w = 'src/best_vlm_W.json' if VLM_ONLY_TUNE else 'src/best_full_W.json'
    if '--dry-run' in sys.argv:          # exercise the code path without leaving a weights file
        print('dry run -- NOT writing', out_w)
    else:
        json.dump(W, open(out_w, 'w'), indent=1)
        print('wrote', out_w)
