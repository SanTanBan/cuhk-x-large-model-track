"""WIDE-GRID re-run (09-13 night) of src/nb_validate.py: the accepted weights sat at the top of the grid
(w_nb_comb 4, w_nb_bel 2), so the grid is extended; everything else is identical.

Head-to-head validation (09-13): neighbouring-clip option evidence (src/neighbours.py) for
`combination`, `single|HAU` and `multi`, inside the full pipeline, on held-out subjects.

The CV prep runs once (CUHKX_* cleared). Each clip's evidence gets `nb_p` from its time
neighbours' option texts (no answers used; neighbours share the user, so they sit in the same
fold). fuse.py adds w_nb_comb * mean(nb_p) to the combination option scores and
w_nb_bel * nb_p to the action beliefs. For each half of the subjects both versions are tuned
on the other half (old: thr_m only; new: w_nb_comb, w_nb_bel, thr_m) and scored on this half.
Accepted only if the new one wins on both halves.

    python src/nb_validate.py [--base=src/best_nv_final_W.json] [--out=src/best_nb_W.json] [--dry-run]
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in ARGS if a.startswith('--%s=' % k)), d)


BASE_F, OUT_F, DRY = arg('base', 'src/best_nv_final_W.json'), arg('out', 'src/best_nb_wide_W.json'), '--dry-run' in ARGS
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]

sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

from neighbours import nb_pool
NB = nb_pool(tr)
print('neighbour pools for %d HAU clips (%d non-empty)' % (len(NB), sum(1 for v in NB.values() if v)), flush=True)
CATS = ['combination|HAU', 'single|HAU', 'multi|HAU']
FOLD_NEW = []
for M_, MM_, HP_, SQ_, clips in FOLD:
    new = []
    for g, feats_, hs, e_ in clips:
        e2 = dict(e_); e2['nb_p'] = NB.get(g.iloc[0].path, {})
        new.append((g, feats_, hs, e2))
    FOLD_NEW.append((M_, MM_, HP_, SQ_, new))

START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
G_OLD = {'thr_m': [-2.5, -2, -1.5]}
G_NEW = {'w_nb_comb': [1, 2, 4, 8, 16], 'w_nb_bel': [0.5, 1, 2, 4, 8], 'thr_m': [-2.5, -2, -1.5]}


def score(FL, W, keep):
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in FL:
        for g, feats_, hs, e_ in clips:
            if (keep is not None and g.iloc[0].user not in keep) or not g.qt.isin(CATS).any():
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats_, MM_, HP_, SQ_, e_)
            for _, r in g[g.qt.isin(CATS)].iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1; res[k][0] += a[r.qa_id] == str(r.answer)
    by = {'%s|%s' % (k[0], 'seq' if k[1] else 'noseq'): o / m for k, (o, m) in res.items()}
    return sum(TESTC[k] * o / m for k, (o, m) in res.items() if m >= 5) / N_TEST, by


def tune(FL, grid, keep):
    best, best_s = None, -1.0
    for combo in itertools.product(*grid.values()):
        W = dict(START); W.update(dict(zip(grid, combo)))
        s = score(FL, W, keep)[0]
        if s > best_s + 1e-12:
            best_s, best = s, dict(zip(grid, combo))
    return best


def with_(v):
    W = dict(START); W.update(v)
    return W


gains, rows = [], []
for i in (0, 1):
    vo, vn = tune(FOLD, G_OLD, HALVES[1 - i]), tune(FOLD_NEW, G_NEW, HALVES[1 - i])
    so, sn = score(FOLD, with_(vo), HALVES[i])[0], score(FOLD_NEW, with_(vn), HALVES[i])[0]
    gains.append(sn - so)
    rows.append('half %d: old %s -> %.4f | new %s -> %.4f | gain %+.4f' % (i, vo, so, vn, sn, sn - so))
    print(rows[-1], flush=True)
ok = all(x > 1e-9 for x in gains)
vo, vn = tune(FOLD, G_OLD, None), tune(FOLD_NEW, G_NEW, None)
ao, an = score(FOLD, with_(vo), None)[1], score(FOLD_NEW, with_(vn), None)[1]
print('held-out gains %+.4f / %+.4f -> %s' % (gains[0], gains[1], 'ACCEPT' if ok else 'reject'))
for k in sorted(an):
    print('   %-26s all subjects (in-sample): %.4f -> %.4f' % (k, ao.get(k, float('nan')), an[k]))
print('weights old %s | new %s' % (vo, vn))
report = dict(base=BASE_F, held_out_gains=gains, accepted=ok, rows=rows, weights_old=vo,
              weights_new=vn, acc_all_old=ao, acc_all_new=an, allowed_categories=CATS)
if DRY:
    print('dry run -- nothing written')
else:
    if ok:
        json.dump(with_(vn), open(OUT_F, 'w'), indent=1, sort_keys=True)
        cfg = json.load(open(BASE_F.replace('.json', '_config.json'))) if os.path.exists(
            BASE_F.replace('.json', '_config.json')) else {'build_env': {}}
        env = dict(cfg.get('build_env', {})); env.update(CUHKX_NB='1')
        json.dump(dict(build_env=env, allowed_categories=CATS, base=BASE_F),
                  open(OUT_F.replace('.json', '_config.json'), 'w'), indent=1)
    json.dump(report, open(OUT_F.replace('.json', '_report.json'), 'w'), indent=1)
    print('wrote', OUT_F.replace('.json', '_report.json'), '(and weights + config)' if ok else '')
