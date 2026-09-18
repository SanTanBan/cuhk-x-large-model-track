"""Per-clip features from the organisers' non-visual supplement (IMU, Skeleton), read straight
from data/LMT_IMU_Radar_Skeleton.zip; nothing is extracted to disk. Written 09-11.

    python src/nonvisual_feats.py      -> features/imu_feats.csv, features/skel_feats.csv

Rows are keyed by `unit`, following the main release's paths: HAU/<user>/<trial>,
HARn/<action>/<user>/<trial>, large_model_track_test/LM_test_XXXX.

IMU: five wearables (WTC chest, WTLA/WTRA arms, WTLL/WTRL legs) at ~10 Hz, stored in
packet-arrival order, so each device is sorted by timestamp and resampled to a 10 Hz grid.
Columns are read by position, since headers are Chinese or English in the same order:
time, device, acc xyz (g), gyro xyz (deg/s), angle xyz (deg), ...
Features follow the public notebook's recipe (band power, spectral entropy, dominant
frequency, autocorrelation), plus distribution stats and 4 time-slot means.

Skeleton: one JSON per video frame with 17 keypoints x 3 (root-centred 3D pose
predictions) and per-keypoint scores. Features: root-relative joint position mean/std, joint
speed stats, motion energy per time slot, a few joint-pair distances, and periodicity.
Radar is skipped: the public notebook reports it lost on subject-disjoint validation.
"""
import sys, io, json, zipfile, collections, time
import numpy as np, pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ZIP = 'data/LMT_IMU_Radar_Skeleton.zip'
ROOT = 'LMT_(IMU,Radar,Skeleton)/'
DEVS = ['WTC', 'WTLA', 'WTRA', 'WTLL', 'WTRL']
FS = 10.0
NAN = float('nan')
SPEC = ('b0', 'b1', 'b2', 'b3', 'ent', 'dom', 'per')


def unit_of(name):
    """zip member -> (unit key, modality) or (None, None)."""
    p = name[len(ROOT):].split('/')
    if len(p) > 4 and p[0] == 'Training' and p[1] == 'HAU':
        return '/'.join(p[1:4]), p[4]
    if len(p) > 5 and p[0] == 'Training' and p[1] == 'HARn':
        return '/'.join(p[1:5]), p[5]
    if len(p) > 3 and p[0] == 'Testing':
        return '/'.join(p[1:3]), p[3]
    return None, None


def spec_feats(x):
    """Band power shares (0-0.5, 0.5-1.5, 1.5-3, 3-5 Hz), spectral entropy, dominant
    frequency, and the autocorrelation peak over lags 0.2-2 s (periodicity)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or not np.isfinite(x).all():
        return [NAN] * len(SPEC)
    x = x - x.mean()
    f = np.fft.rfftfreq(n, 1 / FS)[1:]
    p = (np.abs(np.fft.rfft(x)) ** 2)[1:]
    tot = p.sum() + 1e-12
    bands = [p[(f >= lo) & (f < hi)].sum() / tot for lo, hi in ((0, .5), (.5, 1.5), (1.5, 3), (3, 5.01))]
    q = p / tot
    ent = float(-(q * np.log(q + 1e-12)).sum() / np.log(max(len(q), 2)))
    dom = float(f[int(np.argmax(p))])
    ac = np.correlate(x, x, 'full')[n - 1:] / (x.var() * n + 1e-12)
    per = float(ac[2:min(21, n)].max()) if n > 3 else NAN
    return bands + [ent, dom, per]


def seg_means(x, k=4):
    return [float(s.mean()) if len(s) else NAN for s in np.array_split(np.asarray(x, float), k)]


def imu_feats(tables):
    df = pd.concat(tables, ignore_index=True)
    t = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    dev = df.iloc[:, 1].astype(str).str.extract(r'^(WT[A-Z]+)', expand=False)
    V = df.iloc[:, 2:11].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    out = {}
    ok = t.notna().to_numpy()
    t0 = t[ok].min() if ok.any() else None
    out['imu_dur'] = (t[ok].max() - t0).total_seconds() if t0 is not None else NAN
    names = ('ax', 'ay', 'az', 'am', 'gx', 'gy', 'gz', 'gm', 'nx', 'ny', 'nz')
    for d in DEVS:
        m = ok & (dev == d).to_numpy() & np.isfinite(V).all(1)
        out['%s_n' % d] = int(m.sum())
        if m.sum() < 8:
            for k in names:
                for s in ('mean', 'std', 'p10', 'p90'):
                    out['%s_%s_%s' % (d, k, s)] = NAN
            for k in ('am', 'gm'):
                for s in ('jerk',) + SPEC + ('seg0', 'seg1', 'seg2', 'seg3'):
                    out['%s_%s_%s' % (d, k, s)] = NAN
            continue
        td = (t[m] - t0).dt.total_seconds().to_numpy()
        X = V[m]
        o = np.argsort(td, kind='stable'); td, X = td[o], X[o]
        keep = np.r_[True, np.diff(td) > 0]; td, X = td[keep], X[keep]
        grid = np.arange(td[0], td[-1] + 1e-9, 1 / FS)
        if len(grid) < 8:
            grid = td
        G = np.column_stack([np.interp(grid, td, X[:, j]) for j in range(9)])
        acc, gyr, ang = G[:, 0:3], G[:, 3:6], G[:, 6:9]
        sig = dict(ax=acc[:, 0], ay=acc[:, 1], az=acc[:, 2], am=np.linalg.norm(acc, axis=1),
                   gx=gyr[:, 0], gy=gyr[:, 1], gz=gyr[:, 2], gm=np.linalg.norm(gyr, axis=1),
                   nx=ang[:, 0], ny=ang[:, 1], nz=ang[:, 2])
        for k in names:
            x = sig[k]
            out['%s_%s_mean' % (d, k)] = float(x.mean())
            out['%s_%s_std' % (d, k)] = float(x.std())
            out['%s_%s_p10' % (d, k)] = float(np.percentile(x, 10))
            out['%s_%s_p90' % (d, k)] = float(np.percentile(x, 90))
        for k in ('am', 'gm'):
            x = sig[k]
            out['%s_%s_jerk' % (d, k)] = float(np.abs(np.diff(x)).mean()) if len(x) > 1 else NAN
            for s, v in zip(SPEC, spec_feats(x)):
                out['%s_%s_%s' % (d, k, s)] = v
            for i, v in enumerate(seg_means(x)):
                out['%s_%s_seg%d' % (d, k, i)] = v
    return out


# H36M-style 17-joint order (joint 0 = pelvis / root, 10 = head, 13/16 = wrists, 3/6 = ankles)
PAIRS = {'wrists': (13, 16), 'lwr_head': (13, 10), 'rwr_head': (16, 10), 'ankles': (3, 6),
         'lwr_root': (13, 0), 'rwr_root': (16, 0), 'head_root': (10, 0)}


def skel_feats(frames):
    P = np.stack([k for _, k in sorted(frames, key=lambda x: x[0])])     # T x 17 x 3
    T = len(P)
    out = {'sk_frames': T}
    R = P - P[:, :1, :]
    mu, sd = R.mean(0), R.std(0)
    for j in range(17):
        for c, a in enumerate('xyz'):
            out['j%d_%s_mean' % (j, a)] = float(mu[j, c])
            out['j%d_%s_std' % (j, a)] = float(sd[j, c])
    if T > 2:
        S = np.linalg.norm(np.diff(P, axis=0), axis=2)                     # (T-1) x 17 speeds
        for j in range(17):
            out['j%d_v_mean' % j] = float(S[:, j].mean())
            out['j%d_v_std' % j] = float(S[:, j].std())
            out['j%d_v_p90' % j] = float(np.percentile(S[:, j], 90))
        E = S.mean(1)
        for i, v in enumerate(seg_means(E)):
            out['e_seg%d' % i] = v
        for j in (3, 6, 13, 16):
            for i, v in enumerate(seg_means(S[:, j])):
                out['j%d_vseg%d' % (j, i)] = v
        for s, v in zip(SPEC, spec_feats(E)):
            out['e_%s' % s] = v
    for k, (a, b) in PAIRS.items():
        dist = np.linalg.norm(P[:, a] - P[:, b], axis=1)
        out['d_%s_mean' % k] = float(dist.mean())
        out['d_%s_std' % k] = float(dist.std())
    return out


def main():
    z = zipfile.ZipFile(ZIP)
    units = collections.defaultdict(lambda: {'IMU': [], 'Skeleton': []})
    for n in z.namelist():
        if n.startswith('__MACOSX') or n.endswith('/'):
            continue
        u, mod = unit_of(n)
        if u is None:
            continue
        if mod == 'IMU' and n.endswith('.csv'):
            units[u]['IMU'].append(n)
        elif mod == 'Skeleton' and n.endswith('.json'):
            units[u]['Skeleton'].append(n)
    print('units: %d (with IMU %d, with Skeleton %d)' % (
        len(units), sum(bool(v['IMU']) for v in units.values()),
        sum(bool(v['Skeleton']) for v in units.values())))
    imu_rows, sk_rows, fail, headers = [], [], collections.Counter(), collections.Counter()
    t_start = time.time()
    for i, (u, v) in enumerate(sorted(units.items())):
        if v['IMU']:
            try:
                tabs = []
                for n in v['IMU']:
                    b = z.read(n)
                    headers[b[:12].decode('utf-8', 'replace').split(',')[0]] += 1
                    tabs.append(pd.read_csv(io.BytesIO(b), header=0, encoding='utf-8-sig',
                                            on_bad_lines='skip', low_memory=False))
                r = imu_feats(tabs); r['unit'] = u; imu_rows.append(r)
            except Exception as e:
                fail['imu:' + type(e).__name__] += 1
        if v['Skeleton']:
            try:
                frames = []
                for n in v['Skeleton']:
                    fr = json.loads(z.read(n))
                    if not fr:
                        continue
                    best = max(fr, key=lambda d: float(np.mean(d.get('keypoint_scores') or [0])))
                    k = np.asarray(best['keypoints'], float)
                    if k.shape != (17, 3):
                        continue
                    idx = int(n.rsplit('_', 1)[-1].split('.')[0])
                    frames.append((idx, k))
                if len(frames) >= 3:
                    r = skel_feats(frames); r['unit'] = u; sk_rows.append(r)
                else:
                    fail['skel:too_few_frames'] += 1
            except Exception as e:
                fail['skel:' + type(e).__name__] += 1
        if (i + 1) % 500 == 0:
            print('  %d / %d units, %.0f s' % (i + 1, len(units), time.time() - t_start), flush=True)
    imu = pd.DataFrame(imu_rows).set_index('unit')
    sk = pd.DataFrame(sk_rows).set_index('unit')
    imu.to_csv('features/imu_feats.csv'); sk.to_csv('features/skel_feats.csv')
    print('IMU: %d units x %d features -> features/imu_feats.csv' % imu.shape)
    print('Skeleton: %d units x %d features -> features/skel_feats.csv' % sk.shape)
    print('IMU header first tokens:', dict(headers.most_common(6)))
    print('failures:', dict(fail) or 'none')
    for name, df in (('IMU', imu), ('Skeleton', sk)):
        g = collections.Counter(k.split('/')[0] for k in df.index)
        print('%s coverage by source: %s' % (name, dict(g)))


if __name__ == '__main__':
    main()
