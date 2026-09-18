"""Richer CPU-only clip descriptors (still no VLM, no GPU).

Beyond raw motion energy, three things distinguish these actions cheaply:

  periodicity   jumping jacks / squats / lunges / jogging are strongly repetitive;
                pouring or reading are not. Autocorrelation of the motion signal finds it.
  where         motion high in frame = hair / face / teeth; low = floor / feet;
                the vertical centroid of motion energy separates them.
  body posture  depth statistics say how far away and how tall/wide the person is,
                which separates lying down / sitting / standing.
"""
import cv2, numpy as np, pandas as pd, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load
from motion import clip_video

FPS = 10.0
W, H = 160, 120


def features(video, max_frames=400):
    c = cv2.VideoCapture(video)
    if not c.isOpened():
        return None
    frames = []
    while len(frames) < max_frames:
        ok, f = c.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    c.release()
    if len(frames) < 4:
        return None
    A = np.stack(frames)                       # T,H,W
    D = np.abs(np.diff(A, axis=0))             # T-1,H,W
    m = D.mean(axis=(1, 2))                    # motion signal over time
    T = len(A)

    # --- periodicity of the motion signal ---
    x = m - m.mean()
    per_peak, per_lag = 0.0, 0.0
    if x.std() > 1e-6 and len(x) >= 12:
        ac = np.correlate(x, x, mode='full')[len(x) - 1:]
        ac = ac / (ac[0] + 1e-9)
        hi = min(len(ac) - 1, 40)
        if hi > 3:
            seg = ac[3:hi]
            per_peak = float(seg.max()); per_lag = float(np.argmax(seg) + 3)

    # --- where the motion happens (vertical / horizontal centroid + spread) ---
    E = D.mean(axis=0) + 1e-9                  # H,W energy map
    ys, xs = np.mgrid[0:H, 0:W]
    tot = E.sum()
    cy = float((E * ys).sum() / tot) / H
    cx = float((E * xs).sum() / tot) / W
    sy = float(np.sqrt((E * (ys - cy * H) ** 2).sum() / tot)) / H
    thirds = [float(E[i * H // 3:(i + 1) * H // 3].sum() / tot) for i in range(3)]

    # --- depth / posture proxies from the frame content itself ---
    fg = A > np.percentile(A, 60)
    frac = float(fg.mean())
    rows = fg.mean(axis=(0, 2)); cols = fg.mean(axis=(0, 1))
    ext_y = float((rows > rows.max() * 0.3).sum()) / H
    ext_x = float((cols > cols.max() * 0.3).sum()) / W

    # --- temporal profile ---
    t3 = [float(m[i * len(m) // 3:(i + 1) * len(m) // 3].mean()) for i in range(3)]
    mm = m.mean() + 1e-9
    return dict(n_frames=T, dur=T / FPS,
                motion=float(m.mean()) * FPS, motion_sd=float(m.std()) * FPS,
                motion_p90=float(np.percentile(m, 90)) * FPS,
                motion_p10=float(np.percentile(m, 10)) * FPS,
                burst=float(np.percentile(m, 90) / mm),
                per_peak=per_peak, per_lag=per_lag,
                cy=cy, cx=cx, sy=sy,
                e_top=thirds[0], e_mid=thirds[1], e_bot=thirds[2],
                fg_frac=frac, ext_y=ext_y, ext_x=ext_x,
                t_first=t3[0] / mm, t_mid=t3[1] / mm, t_last=t3[2] / mm,
                mean_lvl=float(A.mean()), std_lvl=float(A.std()))


def run(paths, out):
    rows = []
    for i, p in enumerate(paths):
        v = clip_video(p)
        f = features(v) if os.path.exists(v) else None
        if f:
            f['path'] = p
            rows.append(f)
        if (i + 1) % 250 == 0:
            print('  %d/%d' % (i + 1, len(paths)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print('wrote %s: %d clips, %d features' % (out, len(df), df.shape[1] - 1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['train', 'test', 'both'])
    a = ap.parse_args()
    tr, te = load()
    os.makedirs('features', exist_ok=True)
    if a.which in ('train', 'both'):
        run(sorted(tr.path.unique()), 'features/motion2_train.csv')
    if a.which in ('test', 'both'):
        run(sorted(te.path.unique()), 'features/motion2_test.csv')
