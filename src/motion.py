"""Cheap CPU-only motion features.

The `emotion` questions are manner adverbs -- Quickly / Slowly / Hurriedly / Leisurely /
Steadily -- which describe *how fast and how evenly* the person moves. That is measurable
from frame differencing alone, no VLM required.

Per clip we compute, from the Depth stream (most robust to lighting):
    dur        clip duration in seconds
    motion     mean per-second absolute frame difference   (how fast)
    motion_sd  std of the per-frame difference             (how bursty / uneven)
    motion_p90 90th percentile difference                  (peak effort)
    busy       fraction of frames with above-median motion
"""
import cv2, numpy as np, pandas as pd, os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load

FPS = 10.0
SMALL = (160, 120)


def clip_video(path, root='data', modality='Depth'):
    p = str(path)
    if p.startswith('large_model_track_test'):
        d = os.path.dirname(os.path.dirname(p))
        return os.path.join(root, d, modality, modality + '.mp4')
    return os.path.join(root, p, modality, modality + '.mp4')


def features(video):
    c = cv2.VideoCapture(video)
    if not c.isOpened():
        return None
    prev, diffs = None, []
    while True:
        ok, f = c.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, SMALL), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
        prev = g
    c.release()
    n = len(diffs) + 1
    if len(diffs) < 2:
        return None
    d = np.array(diffs)
    med = float(np.median(d))
    return dict(n_frames=n, dur=n / FPS, motion=float(d.mean()) * FPS,
                motion_sd=float(d.std()) * FPS, motion_p90=float(np.percentile(d, 90)) * FPS,
                busy=float((d > med).mean()))


def run(paths, out):
    rows = []
    for i, p in enumerate(paths):
        v = clip_video(p)
        f = features(v) if os.path.exists(v) else None
        if f:
            f['path'] = p
            rows.append(f)
        if (i + 1) % 200 == 0:
            print('  %d/%d' % (i + 1, len(paths)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print('wrote %s: %d clips' % (out, len(df)))
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['train', 'test', 'both'])
    a = ap.parse_args()
    tr, te = load()
    os.makedirs('features', exist_ok=True)
    if a.which in ('train', 'both'):
        run(sorted(tr.path.unique()), 'features/motion_train.csv')
    if a.which in ('test', 'both'):
        run(sorted(te.path.unique()), 'features/motion_test.csv')
