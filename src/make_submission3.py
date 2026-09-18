"""Final submission: joint structural model + motion models (+ VLM evidence if present)."""
import pandas as pd, numpy as np, collections, sys, os, json
sys.path.insert(0, 'src')
from solver import load, opts
from joint import Joint, clip_features
from fuse import answer_clip_fused, load_vlm, DEFAULT_W
from motion_model import motion_table, MotionModel, HAUPresence
from seq_model import load_segments, SeqModel
from zeroshot import build as build_zs
from feats import load_clip_emb

OUT = sys.argv[1] if len(sys.argv) > 1 else 'submissions/sub03_motion.csv'
USE_CLIP = '--clip' in sys.argv
# score directory, default vlm/ -- e.g. --vlm-dir=vlm_tta for option-order-TTA scores
VLM_DIR = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--vlm-dir=')), 'vlm')
# explicit weights file (e.g. src/best_vlm_W_tta.json); overrides the default preference order
WEIGHTS = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--weights=')), None)
assert WEIGHTS is None or os.path.exists(WEIGHTS), 'weights file not found: %s' % WEIGHTS

tr, te = load()
if USE_CLIP:
    from feats import task_features
    TF = task_features()
    X, X_HARN, X_EMO = TF['X'], TF['X_harn'], TF['X_emo']
    print('per-task features: presence(%d) harn(%d) emotion(%d)'
          % (len(next(iter(X.values()))), len(next(iter(X_HARN.values()))),
             len(next(iter(X_EMO.values())))))
else:
    X_HARN = X_EMO = None
    X, cols = motion_table('motion2')
    if not X:
        X, cols = motion_table('motion')
    print('motion features: %d clips x %d dims' % (len(X), len(cols)))

W = dict(DEFAULT_W)
# VLM-tuned weights only make sense when VLM test evidence is actually present
_have_vlm = os.path.exists(os.path.join(VLM_DIR, 'vlm_scores_test.csv'))
_cands = ((WEIGHTS,) if WEIGHTS else
          (('src/best_vlm_W.json',) if _have_vlm else ()) + ('src/best_full_W.json',
                                                              'src/best_joint_W.json'))
for f in _cands:
    if os.path.exists(f):
        W.update(json.load(open(f)))
        print('weights from', f)
        break
print('W =', json.dumps(W, sort_keys=True))

SEG = load_segments()
print('segment descriptors for %d clips' % len(SEG))
ZS = build_zs(tr, te, load_clip_emb()) if USE_CLIP else None
VLM = load_vlm(os.path.join(VLM_DIR, 'vlm_scores_test.csv'))
print('VLM test evidence for %d questions' % len(VLM))

M = Joint(tr)
MM = MotionModel(tr, X, X_harn=X_HARN, X_emo=X_EMO) if X else None
HP = HAUPresence(tr, X) if X else None
SQ = SeqModel(tr, SEG) if SEG else None
print('models fitted (HAUPresence covers %d actions)' % (len(HP.models) if HP else 0))

pred = {}
missing_motion = 0
# 09-13: CUHKX_NB=1 attaches time-neighbour option evidence (src/neighbours.py); unset = identical
NB = None
if os.environ.get('CUHKX_NB') == '1':
    from neighbours import nb_pool
    from fuse import clip_evidence
    NB = nb_pool(te)
    print('neighbour pools for %d test HAU clips (%d non-empty)' % (len(NB), sum(1 for v in NB.values() if v)))
for p, g in te.groupby('path'):
    if X and p not in X:
        missing_motion += 1
    ev_ = None
    if NB is not None:
        ev_ = clip_evidence(g, MM, HP, SQ, ZS); ev_['nb_p'] = NB.get(p, {})
    pred.update(answer_clip_fused(g, M, W, VLM, clip_features(g, M), MM, HP, SQ,
                                  ev_, ZS))
if missing_motion:
    print('WARNING: %d test clips have no motion features (they fall back to text-only)'
          % missing_motion)

sub = pd.read_csv('data/sample_submission.csv')
sub['prediction'] = sub.qa_id.map(pred)
assert sub.prediction.notna().all(), 'missing predictions'
assert len(sub) == 682 and list(sub.columns) == ['qa_id', 'prediction']

cat = dict(zip(te.qa_id, te.category))
nopt = {r.qa_id: set(opts(r)) for _, r in te.iterrows()}
bad = []
for _, r in sub.iterrows():
    c, a, O = cat[r.qa_id], str(r.prediction), nopt[r.qa_id]
    if not set(a) <= O:
        bad.append((r.qa_id, 'option not offered', a))
    elif c == 'sequence' and (len(a) != 4 or set(a) != O):
        bad.append((r.qa_id, 'not a full permutation', a))
    elif c == 'multi' and list(a) != sorted(a):
        bad.append((r.qa_id, 'multi not sorted', a))
    elif c in ('single', 'combination', 'emotion', 'object_interaction') and len(a) != 1:
        bad.append((r.qa_id, 'not a single letter', a))
print('format violations:', len(bad), bad[:5])
assert not bad

os.makedirs('submissions', exist_ok=True)
sub.to_csv(OUT, index=False)
print('wrote', OUT)

prev = 'submissions/sub02_joint.csv'
if os.path.exists(prev):
    old = pd.read_csv(prev)
    m = old.merge(sub, on='qa_id', suffixes=('_old', '_new')).merge(
        te[['qa_id', 'category']], on='qa_id')
    print('\nchanged vs %s: %d/682' % (prev, (m.prediction_old != m.prediction_new).sum()))
    print(m.groupby('category').apply(
        lambda d: (d.prediction_old != d.prediction_new).mean().round(3),
        include_groups=False).to_string())
