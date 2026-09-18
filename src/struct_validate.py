"""Held-out-subject validation of structural weights found 09-11 (see README).

  prec   w_prec, w_seq          sequence|HAU  pairwise precedence between action texts
  rec_m  w_rec_m, w_sgl_m, thr_m multi|HAU    recurrence of an option in the combination
                                               options / the single question's options
  rec_s  w_rec_s                single|HAU   the same recurrence for single answers
  vseq   w_vlm_pair, w_gen      sequence|HAU VLM pairwise order + generated order (re-check)
  vpres  w_vlm_pres, thr_m      multi, single|HAU  VLM presence votes (re-check)
  route_m w_sans_m, thr_m       multi|HAU    the clip's predicted single answer is present
  route_c w_ovl_m, w_ovl_s      combination (+ single, multi via beliefs): overlap with the
                                clip's multi / single options

Each group is tuned ALONE on one half of the subjects and scored on the other, starting from
the base configuration (sub08 by default), and accepted only if the held-out gain is positive
on BOTH halves; the accepted groups are then checked together the same way. Only the
categories a group can change are evaluated -- all other answers are identical by
construction -- and gains are reported as overall test-weighted accuracy.

    python src/struct_validate.py prec,rec_m,rec_s [--base=src/best_vlm_W.json]
           [--out=src/best_struct_W.json] [--dry-run]
"""
import sys, json, collections, itertools

ARGS = sys.argv[1:]
DRY = '--dry-run' in ARGS
NAMES = [a for a in ARGS if not a.startswith('--')][0].split(',')
BASE_F = next((a.split('=', 1)[1] for a in ARGS if a.startswith('--base=')), 'src/best_vlm_W.json')
OUT_F = next((a.split('=', 1)[1] for a in ARGS if a.startswith('--out=')), 'src/best_struct_W.json')
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, FOLD, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

GROUPS = {
    'prec':  (['w_prec', 'w_seq'], {'w_prec': [0, 0.25, 0.5, 1, 2, 3, 5, 8],
                                    'w_seq': [0, 0.25, 0.5, 1, 2]}, ['sequence|HAU']),
    'rec_m': (['w_rec_m', 'w_sgl_m', 'thr_m'], {'w_rec_m': [0, 0.5, 1, 2, 3, 5],
                                                'w_sgl_m': [0, 0.5, 1, 2],
                                                'thr_m': [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0]},
              ['multi|HAU']),
    'rec_s': (['w_rec_s'], {'w_rec_s': [0, 0.25, 0.5, 1, 2, 3]}, ['single|HAU']),
    # re-checks of VLM groups rejected on sub08's base, on top of the structural weights
    'vseq':  (['w_vlm_pair', 'w_gen'], {'w_vlm_pair': [0, 0.25, 0.5, 1, 2],
                                        'w_gen': [0, 0.5, 1, 2]}, ['sequence|HAU']),
    'vpres': (['w_vlm_pres', 'thr_m'], {'w_vlm_pres': [0, 0.1, 0.25, 0.5, 1],
                                        'thr_m': [-3, -2.5, -2, -1.5, -1]},
              ['multi|HAU', 'single|HAU']),
    # the clip's predicted single answer is always a present action (67/67 in training)
    'route_m': (['w_sans_m', 'thr_m'], {'w_sans_m': [0, 1, 2, 3, 5, 10],
                                        'thr_m': [-3, -2.5, -2, -1.5]}, ['multi|HAU']),
    # combination: its true actions recur among the clip's multi / single options
    'route_c': (['w_ovl_m', 'w_ovl_s'], {'w_ovl_m': [0, 1, 2, 4, 8, 16],
                                         'w_ovl_s': [0, 1, 2, 4, 8]},
                ['combination|HAU', 'single|HAU', 'multi|HAU']),
    # route_c refinements: scale it down on sequence clips; keep it out of the beliefs
    'route_c2': (['w_ovl_seq_scale', 'w_ovl_bel', 'w_ovl_s'],
                 {'w_ovl_seq_scale': [0, 0.25, 1], 'w_ovl_bel': [0, 1], 'w_ovl_s': [2, 8]},
                 ['combination|HAU', 'single|HAU', 'multi|HAU']),
}
START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]


def score(W, keep, cats):
    """Overall test-weighted accuracy contributed by the strata in `cats`."""
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in FOLD:
        for g, feats, hs, e_ in clips:
            if (keep is not None and g.iloc[0].user not in keep) or not g.qt.isin(cats).any():
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats, MM_, HP_, SQ_, e_)
            for _, r in g[g.qt.isin(cats)].iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1; res[k][0] += a[r.qa_id] == str(r.answer)
    return sum(TESTC[k] * ok / n for k, (ok, n) in res.items() if n >= 5) / N_TEST


def tune(keys, grids, keep, cats, W0):
    best_v, best_s = None, -1.0
    for combo in itertools.product(*[grids[k] for k in keys]):
        W = dict(W0); W.update(dict(zip(keys, combo)))
        s = score(W, keep, cats)
        if s > best_s + 1e-12:
            best_s, best_v = s, dict(zip(keys, combo))
    return best_v


report, fitted = {'base': BASE_F, 'groups': {}}, {}
print('\n%-6s %16s %16s  %s' % ('group', 'held-out half 0', 'held-out half 1', 'verdict'))
for n in NAMES:
    keys, grids, cats = GROUPS[n]
    gains = []
    for i in (0, 1):
        v = tune(keys, grids, HALVES[1 - i], cats, START); fitted[(n, i)] = v
        W = dict(START); W.update(v)
        gains.append(score(W, HALVES[i], cats) - score(START, HALVES[i], cats))
    ok = all(x > 1e-9 for x in gains)
    report['groups'][n] = dict(keys=keys, categories=cats, held_out_gains=gains, accepted=ok,
                               fitted_per_half=[fitted[(n, 0)], fitted[(n, 1)]])
    print('%-6s %+16.4f %+16.4f  %s   (fits: %s | %s)' % (n, gains[0], gains[1],
          'ACCEPT' if ok else 'reject', fitted[(n, 0)], fitted[(n, 1)]))

acc = [n for n in NAMES if report['groups'][n]['accepted']]
cats_all = sorted({c for n in acc for c in GROUPS[n][2]})
final = []
if acc:
    cg = []
    for i in (0, 1):
        W = dict(START)
        for n in acc:
            W.update(fitted[(n, i)])
        cg.append(score(W, HALVES[i], cats_all) - score(START, HALVES[i], cats_all))
    print('accepted together, held-out: %+.4f / %+.4f' % tuple(cg))
    final = acc if all(x > 1e-9 for x in cg) else [max(acc, key=lambda n: sum(report['groups'][n]['held_out_gains']))]
    report['combined_held_out_gains'] = cg
W_final = dict(START)
for n in final:
    keys, grids, cats = GROUPS[n]
    W_final.update(tune(keys, grids, None, cats, START))
allowed = sorted({c for n in final for c in GROUPS[n][2]})
report.update(final_groups=final, allowed_categories=allowed,
              final_weights={k: W_final[k] for n in final for k in GROUPS[n][0]})
if final:
    print('final groups %s -> %s | all subjects (in-sample) %+.4f'
          % (final, report['final_weights'], score(W_final, None, allowed) - score(START, None, allowed)))
else:
    print('nothing accepted')
if DRY:
    print('dry run -- nothing written')
else:
    if final:
        json.dump(W_final, open(OUT_F, 'w'), indent=1, sort_keys=True)
    json.dump(report, open(OUT_F.replace('.json', '_report.json'), 'w'), indent=1)
    print('wrote', (OUT_F + ' and ') if final else '', OUT_F.replace('.json', '_report.json'))
