"""Recording-session timeline: the first and last Skeleton frame time of every unit (09-18).

    python src/unit_times.py [--zip=data/LMT_IMU_Radar_Skeleton.zip] [--out=features/unit_times.csv]
                             [--check=features/unit_times.csv]

The organisers' Skeleton frames are named `Color_<YYYY-MM-DD>_<HH-MM-SS.mmm>_<frame>.json`, so each
unit's first and last frame give its recording window (seconds resolution). `src/neighbours.py` and
`src/build_session.py` sort clips by t0 and start a new recording session wherever the gap reaches
30 minutes.

This rebuilds `features/unit_times.csv`, which the emotion session decoder and the time-neighbour
option pooling both read. `--check` compares against an existing table and writes nothing.
"""
import sys, os, re, zipfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nonvisual_feats import unit_of                      # same member -> unit mapping as the features


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--%s=' % k)), d)


ZIP = arg('zip', 'data/LMT_IMU_Radar_Skeleton.zip')
OUT, CHECK = arg('out', 'features/unit_times.csv'), arg('check')
PAT = re.compile(r'_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})[._]')

lo, hi, seen = {}, {}, 0
for n in zipfile.ZipFile(ZIP).namelist():
    if not n.endswith('.json'):
        continue
    u, mod = unit_of(n)
    if u is None or mod != 'Skeleton':
        continue
    m = PAT.search(os.path.basename(n))
    if not m:
        continue
    t = '%s %s:%s:%s' % (m.group(1), m.group(2), m.group(3), m.group(4))
    seen += 1
    if u not in lo or t < lo[u]:
        lo[u] = t
    if u not in hi or t > hi[u]:
        hi[u] = t
print('%d Skeleton frames -> %d units' % (seen, len(lo)))
out = pd.DataFrame({'unit': list(lo), 't0': [lo[u] for u in lo], 't1': [hi[u] for u in lo]})

if CHECK:
    old = pd.read_csv(CHECK, dtype=str).set_index('unit')
    new = out.set_index('unit')
    missing, extra = sorted(set(old.index) - set(new.index)), sorted(set(new.index) - set(old.index))
    both = [u for u in old.index if u in new.index]
    diff = [u for u in both if (old.t0[u], old.t1[u]) != (new.t0[u], new.t1[u])]
    print('CHECK vs %s: %d units in both, %d differ, %d missing here, %d extra here'
          % (CHECK, len(both), len(diff), len(missing), len(extra)))
    for u in diff[:5]:
        print('   %-40s old %s..%s | new %s..%s' % (u, old.t0[u], old.t1[u], new.t0[u], new.t1[u]))
    print('MATCH' if not (diff or missing or extra) else 'MISMATCH')
else:
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    out.to_csv(OUT, index=False)
    print('wrote %s (%d rows)' % (OUT, len(out)))
