"""Per-segment motion descriptors, for the `sequence` (temporal order) questions.

Each HAU clip is split into 4 equal time slots and described separately. Training labels
are free: a clip's `sequence` answer *is* the chronological order of its four actions, so
under an equal-duration assumption slot k holds the k-th action of the answer. That gives
~1200 (slot, action) pairs to learn from, and at test time the order is recovered by
solving the 4x4 assignment between offered actions and time slots.
"""
import cv2, numpy as np, pandas as pd, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load
from motion import clip_video

W, H, NSEG = 160, 120, 4


def seg_feats(A):
    """Descriptor for one contiguous block of frames (T,H,W float32)."""
    if len(A) < 3:
        return [0.0] * 11
    D = np.abs(np.diff(A, axis=0))
    m = D.mean(axis=(1, 2))
    E = D.mean(axis=0) + 1e-9
    ys, xs = np.mgrid[0:H, 0:W]
    tot = E.sum()
    cy = float((E * ys).sum() / tot) / H
    cx = float((E * xs).sum() / tot) / W
    thirds = [float(E[i * H // 3:(i + 1) * H // 3].sum() / tot) for i in range(3)]
    x = m - m.mean()
    per = 0.0
    if x.std() > 1e-6 and len(x) >= 10:
        ac = np.correlate(x, x, mode='full')[len(x) - 1:]
        ac = ac / (ac[0] + 1e-9)
        hi = min(len(ac) - 1, 25)
        if hi > 3:
            per = float(ac[3:hi].max())
    fg = A > np.percentile(A, 60)
    rows = fg.mean(axis=(0, 2))
    return [float(np.log1p(m.mean())), float(np.log1p(m.std())),
            float(np.log1p(np.percentile(m, 90))), cy, cx,
            thirds[0], thirds[1], thirds[2], per,
            float((rows > rows.max() * 0.3).sum()) / H, float(A.mean()) / 255.0]


def clip_segments(video, max_frames=400):
    c = cv2.VideoCapture(video)
    if not c.isOpened():
        return None
    fr = []
    while len(fr) < max_frames:
        ok, f = c.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    c.release()
    if len(fr) < NSEG * 3:
        return None
    A = np.stack(fr)
    bounds = np.linspace(0, len(A), NSEG + 1).astype(int)
    return [seg_feats(A[bounds[k]:bounds[k + 1]]) for k in range(NSEG)]


def run(paths, out):
    rows = []
    for i, p in enumerate(paths):
        v = clip_video(p)
        segs = clip_segments(v) if os.path.exists(v) else None
        if segs:
            for k, f in enumerate(segs):
                rows.append(dict(path=p, seg=k, **{('f%02d' % j): v2 for j, v2 in enumerate(f)}))
        if (i + 1) % 250 == 0:
            print('  %d/%d' % (i + 1, len(paths)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print('wrote %s: %d rows (%d clips)' % (out, len(df), df.path.nunique()), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['train', 'test', 'both'])
    a = ap.parse_args()
    tr, te = load()
    os.makedirs('features', exist_ok=True)
    if a.which in ('train', 'both'):
        run(sorted(tr[tr.source == 'HAU'].path.unique()), 'features/seg_train.csv')
    if a.which in ('test', 'both'):
        run(sorted(te[te.source == 'HAU'].path.unique()), 'features/seg_test.csv')
