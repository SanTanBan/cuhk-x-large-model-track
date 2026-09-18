"""Sanity-check VLM score files pulled from the Kaggle kernel.

    kaggle kernels output santanubanerjee9/cuhk-x-vlm-evidence -p vlm
    python src/check_vlm.py [vlm_dir]

Reports, before any fusion:
  * structure   rows per kind, and whether every question got the evidence it should
  * test cover  fraction of the 682 test questions with usable VLM evidence
  * signal      the VLM's *standalone* accuracy per category on the training clips it
                scored, next to the chance rate -- if it can't beat chance on its own there
                is nothing for the fusion to find
"""
import pandas as pd, numpy as np, sys, os, json, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts
from fuse import load_vlm, best_perm

D = sys.argv[1] if len(sys.argv) > 1 else 'vlm'
tr, te = load()
CHANCE = {'single|HAU': .25, 'single|HARn': 1 / 3, 'combination|HAU': .25, 'emotion|HAU': .25,
          'object_interaction|HARn': .25, 'multi|HAU': 1 / 15, 'sequence|HAU': 1 / 24}

info = os.path.join(D, 'run_info.json')
if os.path.exists(info):
    print('run_info:', json.dumps(json.load(open(info)), indent=1)[:1500])

for split, df in (('test', te), ('train', tr)):
    f = os.path.join(D, 'vlm_scores_%s.csv' % split)
    if not os.path.exists(f):
        print('\n[%s] no file %s' % (split, f)); continue
    raw = pd.read_csv(f)
    V = load_vlm(f)
    print('\n[%s] %d rows, %d questions, kinds=%s'
          % (split, len(raw), raw.qa_id.nunique(), raw.kind.value_counts().to_dict()))
    sub = df[df.qa_id.isin(V)]
    need = {'single': 'letter', 'combination': 'letter', 'emotion': 'letter',
            'object_interaction': 'letter', 'multi': 'present', 'sequence': 'pair'}
    bad = [q for q, c in zip(sub.qa_id, sub.category) if not V[q].get(need[c])]
    print('  questions missing their expected evidence kind: %d' % len(bad), bad[:5])
    if split == 'test':
        print('  test coverage: %d / %d questions' % (len(sub), len(te)))
        continue

    # standalone VLM accuracy on scored training questions
    res = collections.defaultdict(lambda: [0, 0])
    for _, r in sub.iterrows():
        e, O, qt = V[r.qa_id], opts(r), r['qt']
        if r.category in ('single', 'combination', 'emotion', 'object_interaction'):
            if not e['letter']: continue
            pred = max(O, key=lambda L: e['letter'].get(L, -1e9))
        elif r.category == 'multi':
            if not e['present']: continue
            pred = ''.join(sorted(L for L in O if e['present'].get(L, -1) > 0)) or \
                   max(O, key=lambda L: e['present'].get(L, -1e9))
        elif r.category == 'sequence':
            pred = best_perm(O, e['pair'], e['perm'], 2.0) if (e['pair'] or e['perm']) else None
            if pred is None: continue
            res['sequence|generated-only'][1] += 1
            res['sequence|generated-only'][0] += (e['perm'] == str(r.answer))
        res[qt][1] += 1
        res[qt][0] += (pred == str(r.answer))
    print('  VLM ALONE on scored training questions:')
    for k, (ok, n) in sorted(res.items()):
        ch = CHANCE.get(k)
        print('    %-28s acc=%.3f  n=%4d  chance=%s' % (k, ok / max(1, n), n,
              '%.3f' % ch if ch else '-'))
