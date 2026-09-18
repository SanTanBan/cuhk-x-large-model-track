"""Inspect the organisers' LMT_(IMU,Radar,Skeleton).zip once it is in data/ (written 09-11).

Prints the folder structure, the file types, a peek at one file of each type, and how the
file paths map onto our clip ids (HAU/<user>/<trial>, HARn/<action>/<user>/<trial>,
large_model_track_test/LM_test_XXXX). Read-only: nothing is extracted, and .npy files are
loaded with allow_pickle=False.

    python src/lmt_inspect.py [path-to-zip]
"""
import sys, os, io, zipfile, collections

if hasattr(sys.stdout, 'reconfigure'):       # Windows consoles default to cp1252 (BOMs etc.)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

Z = sys.argv[1] if len(sys.argv) > 1 else 'data/LMT_IMU_Radar_Skeleton.zip'
z = zipfile.ZipFile(Z)
infos = z.infolist()
files = [i for i in infos if not i.is_dir()]
print('%s: %d entries, %.0f MB uncompressed' % (Z, len(infos), sum(i.file_size for i in infos) / 1e6))
print('file types:', collections.Counter(os.path.splitext(i.filename)[1].lower() for i in files).most_common())
print('\ntop of the tree (first 3 path levels):')
for k, c in collections.Counter('/'.join(i.filename.split('/')[:3]) for i in files).most_common(30):
    print('%7d  %s' % (c, k))

seen = set()
for i in files:
    e = os.path.splitext(i.filename)[1].lower()
    if e in seen:
        continue
    seen.add(e)
    print('\n--- sample %s (%d bytes): %s' % (e or '(no ext)', i.file_size, i.filename))
    if e in ('.csv', '.txt', '.json', '.md', '.tsv'):
        print(z.read(i.filename)[:700].decode('utf-8', 'replace'))
    elif e == '.npy':
        import numpy as np
        a = np.load(io.BytesIO(z.read(i.filename)), allow_pickle=False)
        print('npy shape %s dtype %s first values %s' % (a.shape, a.dtype, a.ravel()[:12]))
    elif e == '.zip':
        inner = zipfile.ZipFile(io.BytesIO(z.read(i.filename)))
        print('nested zip with %d entries, e.g. %s' % (len(inner.namelist()), inner.namelist()[:5]))
    else:
        print(repr(z.read(i.filename)[:160]))

names = [i.filename for i in files]
print('\nmapping onto our clip ids:')
for key in ('HAU/', 'HARn/', 'large_model_track_test', 'LM_test_', 'IMU', 'Skeleton', 'Radar'):
    hits = [n for n in names if key.lower() in n.lower()]
    print('  %-24s %6d files, e.g. %s' % (key, len(hits), hits[:2]))
