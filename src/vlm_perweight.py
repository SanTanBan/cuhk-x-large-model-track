"""Per-weight nested validation of the VLM fusion weights (the 09-12 step).

Tuning all nine VLM weights together on 320 clips looked +0.017 in-sample and was -0.003 on
held-out subjects. So here each VLM weight -- or a tightly coupled pair -- is tuned ALONE on
one half of the covered training subjects and scored on the other half, and is accepted
only if the held-out gain is positive on BOTH halves. The accepted groups are then checked
together the same way. Only what survives is written to src/best_vlm_W.json, along with a
report naming the categories each accepted group may legitimately change (the leak guard).

    python src/vlm_perweight.py [--dry-run]
"""
import sys, json, collections, itertools

DRY = '--dry-run' in sys.argv
_tag = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--out-tag=')), '')
SUF = '_' + _tag if _tag else ''          # --out-tag=tta -> *_tta.json, never clobbering the main files
_vd = [a for a in sys.argv if a.startswith('--vlm-dir=')]     # e.g. a TTA-merged score dir
sys.argv = ['cv_full.py', '--clip', '--only-vlm'] + _vd
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, te, FOLD, VLM, COVERED, TESTC, DEFAULT_W, answer_clip_fused, ...

BASE = json.load(open('src/best_full_W.json'))
START = dict(DEFAULT_W); START.update(BASE)
for k in list(START):
    if k.startswith('w_vlm'):
        START[k] = 0.0
START['w_gen'] = 0.0

# group -> (keys tuned together, grids, categories the group may change)
# combination feeds the action beliefs, so it legitimately moves single|HAU and multi too;
# VLM presence feeds the same beliefs, so it moves single|HAU as well as multi.
GROUPS = collections.OrderedDict([
    ('harn',       (['w_vlm_single_harn'], {'w_vlm_single_harn': [0.5, 1, 2, 3, 5, 8]},
                    ['single|HARn'])),
    ('single_hau', (['w_vlm_single'], {'w_vlm_single': [0.25, 0.5, 1, 2, 3]},
                    ['single|HAU'])),
    ('comb_noseq', (['w_vlm_comb_noseq'], {'w_vlm_comb_noseq': [0.25, 0.5, 1, 2, 3]},
                    ['combination|HAU', 'single|HAU', 'multi|HAU'])),
    ('comb_seq',   (['w_vlm_comb_seq'], {'w_vlm_comb_seq': [0.25, 0.5, 1, 2, 3]},
                    ['combination|HAU', 'single|HAU', 'multi|HAU'])),
    ('oi',         (['w_vlm_oi'], {'w_vlm_oi': [0.25, 0.5, 1, 2, 3]},
                    ['object_interaction|HARn'])),
    ('emotion',    (['w_vlm_emotion'], {'w_vlm_emotion': [0.25, 0.5, 1, 2, 3]},
                    ['emotion|HAU'])),
    ('presence',   (['w_vlm_pres', 'thr_m'], {'w_vlm_pres': [0.1, 0.25, 0.5, 1, 2],
                                              'thr_m': [-5, -4, -3, -2.5, -2, -1.5, -1]},
                    ['multi|HAU', 'single|HAU'])),
    ('sequence',   (['w_vlm_pair', 'w_gen'], {'w_vlm_pair': [0, 0.25, 0.5, 1, 2],
                                              'w_gen': [0, 1, 2]},
                    ['sequence|HAU'])),
])

cov_users = sorted({int(u) for u in tr[tr.qa_id.isin(COVERED)].user})
HALVES = [set(cov_users[0::2]), set(cov_users[1::2])]


def ev_sub(W, keep_users=None):
    """Test-weighted accuracy over covered clips of the given subjects (all if None)."""
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in FOLD:
        for g, feats, hs, e_ in clips:
            if keep_users is not None and g.iloc[0].user not in keep_users:
                continue
            if not any(q in COVERED for q in g.qa_id):
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats, MM_, HP_, SQ_, e_)
            for _, r in g.iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1
                res[k][0] += (a[r.qa_id] == str(r.answer))
    num = den = 0
    for k, n in TESTC.items():
        if k in res and res[k][1] >= 5:
            num += n * res[k][0] / res[k][1]; den += n
    return num / max(den, 1)


def tune(keys, grids, keep_users):
    best_v, best_s = None, -1.0
    for combo in itertools.product(*[grids[k] for k in keys]):
        W = dict(START); W.update(dict(zip(keys, combo)))
        s = ev_sub(W, keep_users)
        if s > best_s + 1e-12:
            best_s, best_v = s, dict(zip(keys, combo))
    return best_v


print('')
print('covered subjects: %s  |  VLM-scored training questions: %d' % (cov_users, len(COVERED)))
base_out = [ev_sub(START, HALVES[i]) for i in (0, 1)]
report = {'groups': {}, 'n_questions': len(COVERED)}
fitted = {}                                    # (group, half) -> values tuned on the OTHER half
print('')
print('%-11s %22s %22s  %s' % ('group', 'held-out half 0', 'held-out half 1', 'verdict'))
for name, (keys, grids, cats) in GROUPS.items():
    gains = []
    for i in (0, 1):
        v = tune(keys, grids, HALVES[1 - i])
        fitted[(name, i)] = v
        W = dict(START); W.update(v)
        gains.append(ev_sub(W, HALVES[i]) - base_out[i])
    ok = all(x > 1e-9 for x in gains)
    report['groups'][name] = {'keys': keys, 'held_out_gains': gains, 'accepted': ok,
                              'categories': cats,
                              'fitted_per_half': [fitted[(name, 0)], fitted[(name, 1)]]}
    print('%-11s %+22.4f %+22.4f  %s' % (name, gains[0], gains[1], 'ACCEPT' if ok else 'reject'))

accepted = [n for n, r in report['groups'].items() if r['accepted']]
print('')
print('accepted individually:', accepted or 'none')

final_groups = []
if accepted:
    # the accepted groups together, each at the values tuned on the fit half
    comb_gains = []
    for i in (0, 1):
        W = dict(START)
        for n in accepted:
            W.update(fitted[(n, i)])
        comb_gains.append(ev_sub(W, HALVES[i]) - base_out[i])
    print('accepted groups together, held-out: %+.4f / %+.4f' % tuple(comb_gains))
    if all(x > 1e-9 for x in comb_gains):
        final_groups = accepted
    else:            # interactions spoil the combination: keep the single best group
        best = max(accepted, key=lambda n: sum(report['groups'][n]['held_out_gains']))
        print('combination fails on a half -> keeping only the best single group:', best)
        final_groups = [best]
    report['combined_held_out_gains'] = comb_gains

# final values: tuned on ALL covered clips, only for the groups that survived
W_final = dict(START)
for n in final_groups:
    keys, grids, _ = GROUPS[n]
    W_final.update(tune(keys, grids, None))
allowed = sorted({c for n in final_groups for c in GROUPS[n][2]} | {'single|HARn'})
report.update(final_groups=final_groups, allowed_categories=allowed,
              final_vlm_weights={k: v for k, v in W_final.items()
                                 if (k.startswith('w_vlm') or k in ('w_gen', 'thr_m'))
                                 and v != START.get(k)},
              score_all_covered={'current': ev_sub(START), 'final': ev_sub(W_final)})
print('final groups:', final_groups or 'none', '| weights changed:', report['final_vlm_weights'])
print('all covered clips (in-sample, for reference): %.4f -> %.4f'
      % (report['score_all_covered']['current'], report['score_all_covered']['final']))

if DRY:
    print('dry run -- NOT writing src/best_vlm_W%s.json or the report' % SUF)
elif final_groups:
    json.dump(W_final, open('src/best_vlm_W%s.json' % SUF, 'w'), indent=1, sort_keys=True)
    json.dump(report, open('src/vlm_perweight_report%s.json' % SUF, 'w'), indent=1)
    print('wrote src/best_vlm_W%s.json and src/vlm_perweight_report%s.json' % (SUF, SUF))
else:
    json.dump(report, open('src/vlm_perweight_report%s.json' % SUF, 'w'), indent=1)
    print('nothing validated -- src/best_vlm_W%s.json not written; report written' % SUF)
