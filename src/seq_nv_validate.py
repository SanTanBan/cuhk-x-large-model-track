"""Head-to-head validation (09-12 night): non-visual time-slot features for the `sequence` slot
model vs the current motion segments, inside the full pipeline, on held-out subjects.

The CV prep runs once with the current features (CUHKX_* cleared). Then each fold's SeqModel
is refit on the new slot table (CUHKX_SEQ) and only the slot log-likelihoods (`ev['seq']`)
are swapped; the precedence prior is unchanged. For each half of the subjects BOTH versions
have (w_seq, w_prec) tuned on the other half -- the new one also its regularisation C -- and
are scored on this half. Accepted only if the new one wins on both halves.

    python src/seq_nv_validate.py --seq=skel [--base=src/best_nv_final_W.json]
                                  [--out=src/best_nv_seq_W.json] [--dry-run]
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in ARGS if a.startswith('--%s=' % k)), d)


SEQ = arg('seq', 'skel')
BASE_F, OUT_F, DRY = arg('base', 'src/best_nv_final_W.json'), arg('out', 'src/best_nv_seq_W.json'), '--dry-run' in ARGS
CS = [0.1, 0.3, 1.0]
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]                     # the prep must see the current (old) features

sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, SEG, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

os.environ['CUHKX_SEQ'] = SEQ
SEG_NEW = load_segments()
print('new slot table: CUHKX_SEQ=%s, %d clips x %s' % (SEQ, len(SEG_NEW), next(iter(SEG_NEW.values())).shape))
CATS = ['sequence|HAU']


def refit(C):
    FL = []
    for (M_, MM_, HP_, SQ_, clips), f in zip(FOLD, folds):
        SQn = SeqModel(tr[~tr.user.isin(f)], SEG_NEW, C=C)
        new = []
        for g, feats_, hs, e_ in clips:
            e2 = dict(e_)
            if (g.category == 'sequence').any():
                e2['seq'] = clip_evidence(g, None, None, SQn, None)['seq']
            new.append((g, feats_, hs, e2))
        FL.append((M_, MM_, HP_, SQn, new))
    return FL


FOLD_NEW = {C: refit(C) for C in CS}
START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
GRID = {'w_seq': [0, 0.25, 0.5, 1, 2], 'w_prec': [0, 0.25, 0.5, 1, 2, 3, 5]}


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
    ok = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())
    return sum(TESTC[k] * o / m for k, (o, m) in res.items() if m >= 5) / N_TEST, ok / max(1, n)


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
    so, eo = score(FOLD, with_(vo), HALVES[i])
    sn, en = score(FOLD_NEW[cn], with_(vn), HALVES[i])
    gains.append(sn - so)
    rows.append('half %d: old %s -> overall %.4f (seq exact %.3f) | new C=%s %s -> %.4f (seq exact %.3f) | gain %+.4f'
                % (i, vo, so, eo, cn, vn, sn, en, sn - so))
    print(rows[-1], flush=True)
ok = all(x > 1e-9 for x in gains)
co, vo = tune(OLD, None); cn, vn = tune(FOLD_NEW, None)
ao, an = score(FOLD, with_(vo), None)[1], score(FOLD_NEW[cn], with_(vn), None)[1]
print('held-out gains %+.4f / %+.4f -> %s' % (gains[0], gains[1], 'ACCEPT' if ok else 'reject'))
print('all subjects (in-sample) sequence exact: %.3f %s -> %.3f C=%s %s' % (ao, vo, an, cn, vn))
report = dict(config={'CUHKX_SEQ': SEQ, 'CUHKX_C_SEQ': cn}, base=BASE_F, held_out_gains=gains,
              accepted=ok, rows=rows, weights_old=vo, weights_new=vn, seq_exact_all_old=ao,
              seq_exact_all_new=an, allowed_categories=CATS)
if DRY:
    print('dry run -- nothing written')
else:
    if ok:
        W = with_(vn)
        json.dump(W, open(OUT_F, 'w'), indent=1, sort_keys=True)
        cfg = json.load(open(BASE_F.replace('.json', '_config.json'))) if os.path.exists(
            BASE_F.replace('.json', '_config.json')) else {'build_env': {}}
        env = dict(cfg.get('build_env', {})); env.update(CUHKX_SEQ=SEQ, CUHKX_C_SEQ=str(cn))
        json.dump(dict(build_env=env, allowed_categories=CATS, base=BASE_F),
                  open(OUT_F.replace('.json', '_config.json'), 'w'), indent=1)
    json.dump(report, open(OUT_F.replace('.json', '_report.json'), 'w'), indent=1)
    print('wrote', OUT_F.replace('.json', '_report.json'), '(and weights + config)' if ok else '')
