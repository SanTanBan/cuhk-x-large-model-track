"""Head-to-head validation (09-13): the organisers' IMU / Skeleton features appended to the HAU
action-presence model (motion + PCA-64 CLIP today), inside the full pipeline, on held-out subjects.

The CV prep runs once with the current features (CUHKX_* cleared). Then each fold's HAUPresence
is refit on the new table (CUHKX_PRES) and only its log-odds (`ev['mot_p']`) are swapped. That
evidence feeds `combination` (w_mot_comb) and the action beliefs behind `single|HAU` and `multi`
(w_mot_bel, with the absolute cut-off thr_m). For each half of the subjects BOTH versions have
those weights tuned on the other half -- the new one also its regularisation C -- and are scored
on this half, over the three categories. Accepted only if the new one wins on both halves.

    python src/pres_nv_validate.py --pres=skel [--base=src/best_nv_final_W.json]
                                   [--out=src/best_nv_pres_W.json] [--dry-run]
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in ARGS if a.startswith('--%s=' % k)), d)


PRES = arg('pres', 'skel')
BASE_F, OUT_F, DRY = arg('base', 'src/best_nv_final_W.json'), arg('out', 'src/best_nv_pres_W.json'), '--dry-run' in ARGS
CS = [float(c) for c in arg('cs', '0.1,1').split(',')]
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]                     # the prep must see the current (old) features

sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, X, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

os.environ['CUHKX_PRES'] = PRES
X_NEW = task_features(PRESENCE)['X']
print('new presence table: CUHKX_PRES=%s, %d clips x %d' % (PRES, len(X_NEW), len(next(iter(X_NEW.values())))))
CATS = ['combination|HAU', 'single|HAU', 'multi|HAU']


def refit(C):
    os.environ['CUHKX_C_PRES'] = str(C)
    FL = []
    for (M_, MM_, HP_, SQ_, clips), f in zip(FOLD, folds):
        HPn = HAUPresence(tr[~tr.user.isin(f)], X_NEW)
        new = []
        for g, feats_, hs, e_ in clips:
            e2 = dict(e_)
            if g.iloc[0].source == 'HAU':
                e2['mot_p'] = HPn.logodds(g.iloc[0].path)
            new.append((g, feats_, hs, e2))
        FL.append((M_, MM_, HPn, SQ_, new))
    del os.environ['CUHKX_C_PRES']
    return FL


FOLD_NEW = {}
for C in CS:
    FOLD_NEW[C] = refit(C)
    print('refit C=%s' % C, flush=True)
START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
GRID = {'w_mot_comb': [0, 0.25, 0.5, 1, 2], 'w_mot_bel': [0, 0.1, 0.25, 0.5],
        'thr_m': [-2.5, -2, -1.5]}


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


def tune(options, keep):
    """options: {C: FOLD list}; -> (C, weights) maximising the score on `keep`."""
    best, best_s = None, -1.0
    for C, FL in options.items():
        for combo in itertools.product(*GRID.values()):
            W = dict(START); W.update(dict(zip(GRID, combo)))
            s = score(FL, W, keep)[0]
            if s > best_s + 1e-12:
                best_s, best = s, (C, dict(zip(GRID, combo)))
    return best


def with_(v):
    W = dict(START); W.update(v)
    return W


OLD = {None: FOLD}
gains, rows = [], []
for i in (0, 1):
    co, vo = tune(OLD, HALVES[1 - i])
    cn, vn = tune(FOLD_NEW, HALVES[1 - i])
    so = score(FOLD, with_(vo), HALVES[i])[0]
    sn = score(FOLD_NEW[cn], with_(vn), HALVES[i])[0]
    gains.append(sn - so)
    rows.append('half %d: old %s -> %.4f | new C=%s %s -> %.4f | gain %+.4f' % (i, vo, so, cn, vn, sn, sn - so))
    print(rows[-1], flush=True)
ok = all(x > 1e-9 for x in gains)
co, vo = tune(OLD, None); cn, vn = tune(FOLD_NEW, None)
ao, an = score(FOLD, with_(vo), None)[1], score(FOLD_NEW[cn], with_(vn), None)[1]
print('held-out gains %+.4f / %+.4f -> %s' % (gains[0], gains[1], 'ACCEPT' if ok else 'reject'))
for k in sorted(an):
    print('   %-26s all subjects (in-sample): %.4f -> %.4f' % (k, ao.get(k, float('nan')), an[k]))
print('weights old %s | new C=%s %s' % (vo, cn, vn))
report = dict(config={'CUHKX_PRES': PRES, 'CUHKX_C_PRES': cn}, base=BASE_F, held_out_gains=gains,
              accepted=ok, rows=rows, weights_old=vo, weights_new=vn, acc_all_old=ao,
              acc_all_new=an, allowed_categories=CATS)
if DRY:
    print('dry run -- nothing written')
else:
    tag = OUT_F.replace('.json', '_%s' % PRES.replace('+', ''))
    if ok:
        json.dump(with_(vn), open(tag + '.json', 'w'), indent=1, sort_keys=True)
        cfg = json.load(open(BASE_F.replace('.json', '_config.json'))) if os.path.exists(
            BASE_F.replace('.json', '_config.json')) else {'build_env': {}}
        env = dict(cfg.get('build_env', {})); env.update(CUHKX_PRES=PRES, CUHKX_C_PRES=str(cn))
        json.dump(dict(build_env=env, allowed_categories=CATS, base=BASE_F),
                  open(tag + '_config.json', 'w'), indent=1)
    json.dump(report, open(tag + '_report.json', 'w'), indent=1)
    print('wrote', tag + '_report.json', '(and weights + config)' if ok else '')
