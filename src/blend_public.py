"""Category-level blend of our submission with the public 0.77777 release (team Fususu).

    python src/blend_public.py OURS OUT --theirs=G1[,G2,...|,multi|HAU,...] [--ref=...]

`--theirs` takes group names and/or single question types (e.g. `multi|HAU`), so one
category can be swapped at a time.

Answers for every question in the listed groups come from the public release; all other
answers come from OURS. Groups are whole question types -- decisions are made per group,
refereed by the public LB, never per question (user decision 2026-09-11, FEEDBACK.md).

  G1  emotion|HAU, single|HARn, object_interaction|HARn
      where the release's non-visual specialists (IMU emotion, Skeleton HARn action and
      object) produced its gains over a structural base
  G2  multi|HAU, sequence|HAU
      their evidence here is structural only; ours adds motion, CLIP and the VLM
  G3  combination|HAU, single|HAU
      where our structural leaks + VLM are strongest (CV 0.90-0.97)
"""
import sys, hashlib, collections
import pandas as pd
sys.path.insert(0, 'src')
from solver import load

REF_MD5 = '8f997f262e7be646296442148dc8e64f'
GROUPS = {
    'G1': {'emotion|HAU', 'single|HARn', 'object_interaction|HARn'},
    'G2': {'multi|HAU', 'sequence|HAU'},
    'G3': {'combination|HAU', 'single|HAU'},
}


def valid(pred, cat):
    p = str(pred).strip().upper()
    if not p or any(c not in 'ABCD' for c in p) or len(set(p)) != len(p):
        return False
    if cat == 'multi':
        return p == ''.join(sorted(p))
    if cat == 'sequence':
        return len(p) == 4
    return len(p) == 1


def blend(ours, ref, groups):
    _, te = load()
    assert set(te.qt) == set().union(*GROUPS.values()), set(te.qt)
    take = set().union(*(GROUPS[g] if g in GROUPS else {g} for g in groups))
    o = pd.read_csv(ours, dtype=str).set_index('qa_id')['prediction']
    t = pd.read_csv(ref, dtype=str).set_index('qa_id')['prediction']
    assert set(o.index) == set(t.index) == set(te.qa_id)
    hs = te.groupby('path').category.apply(lambda c: (c == 'sequence').any())
    rows, stat = [], collections.defaultdict(lambda: [0, 0, 0])
    for _, r in te.iterrows():
        use_t = r.qt in take
        p = t[r.qa_id] if use_t else o[r.qa_id]
        assert valid(p, r.category), (r.qa_id, p)
        rows.append((r.qa_id, p))
        s = stat[(r.qt, 'seq' if hs[r.path] else 'noseq', 'THEIRS' if use_t else 'ours')]
        s[0] += 1; s[1] += o[r.qa_id] == t[r.qa_id]; s[2] += p != o[r.qa_id]
    return pd.DataFrame(rows, columns=['qa_id', 'prediction']), stat


if __name__ == '__main__':
    ours, out = sys.argv[1], sys.argv[2]
    groups = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--theirs=')).split(',')
    ref = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--ref=')),
               'public_ref/fususu_lb0.77777.csv')
    assert hashlib.md5(open(ref, 'rb').read()).hexdigest() == REF_MD5, 'reference file changed'
    assert all(g in GROUPS or g in set().union(*GROUPS.values()) for g in groups), groups
    sub, stat = blend(ours, ref, groups)
    sub.to_csv(out, index=False)
    print('%-26s %-6s %-7s %4s %6s %8s' % ('stratum', 'seq?', 'source', 'n', 'agree', 'changed'))
    for k, (n, ag, ch) in sorted(stat.items()):
        print('%-26s %-6s %-7s %4d %6.3f %8d' % (k[0], k[1], k[2], n, ag / n, ch))
    print('%s: theirs on %s | %d of %d answers differ from %s' % (
        out, '+'.join(groups), sum(v[2] for v in stat.values()), len(sub), ours))
