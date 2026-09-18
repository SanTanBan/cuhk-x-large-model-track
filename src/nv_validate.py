"""Head-to-head validation of the non-visual features (09-11): new feature configuration vs the
current one, on held-out subjects.

The CV prep runs once with the current features (all CUHKX_* switches cleared). Then each
fold's MotionModel is refit with the new configuration, and only the evidence it feeds is
swapped in: `mot_e` (emotion) and `mot_h` / `mot_o` (HARn action, object via action). For
each half of the subjects, BOTH versions have their fusion weights tuned on the other half
and are scored on this one. The new configuration is accepted only if it beats the current
one on both halves.

    python src/nv_validate.py [--emo=imu] [--c-emo=0.1] [--clf-emo=rf]
                              [--harn=skel] [--c-harn=1] [--clf-harn=rf] [--harn-units] [--obj-clip]
                              [--base=src/best_struct4_W.json] [--out=src/best_nv_W.json] [--dry-run]

On acceptance it writes the weights file plus `<out>_config.json` holding the CUHKX_*
environment the build needs, e.g.:
    CUHKX_EMO=imu CUHKX_C_EMO=0.1 python src/make_submission3.py OUT --clip --weights=src/best_nv_W.json
"""
import sys, os, json, collections, itertools

ARGS = sys.argv[1:]


def arg(k, d=None):
    return next((a.split('=', 1)[1] for a in ARGS if a.startswith('--%s=' % k)), d)


EMO, HARN = arg('emo'), arg('harn')
NEWENV = {}
if EMO:
    NEWENV['CUHKX_EMO'] = EMO
if HARN:
    NEWENV['CUHKX_HARN'] = HARN
for k, e in (('c-emo', 'CUHKX_C_EMO'), ('c-harn', 'CUHKX_C_HARN'),
             ('clf-emo', 'CUHKX_CLF_EMO'), ('clf-harn', 'CUHKX_CLF_HARN')):
    if arg(k):
        NEWENV[e] = arg(k)
if '--harn-units' in ARGS:
    NEWENV['CUHKX_HARN_UNITS'] = '1'
if '--obj-clip' in ARGS:          # objects stay on the CLIP classifier (motion_model.py)
    NEWENV['CUHKX_OBJ_CLIP'] = '1'
BASE_F, OUT_F, DRY = arg('base', 'src/best_struct4_W.json'), arg('out', 'src/best_nv_W.json'), '--dry-run' in ARGS
for k in [k for k in os.environ if k.startswith('CUHKX_')]:
    del os.environ[k]                     # the prep must see the current (old) features

sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, folds, FOLD, X, PRESENCE, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

os.environ.update(NEWENV)
from feats import task_features as _tf
from motion_model import MotionModel as _MM
TFn = _tf(PRESENCE)
FOLD_NEW = []
for (M_, MM_, HP_, SQ_, clips), f in zip(FOLD, folds):
    MMn = _MM(tr[~tr.user.isin(f)], X, X_harn=TFn['X_harn'], X_emo=TFn['X_emo'])
    new = []
    for g, feats_, hs, e_ in clips:
        path, e2 = g.iloc[0].path, dict(e_)
        if HARN:
            e2['mot_h'], e2['mot_o'] = MMn.harn_action_logp(path), MMn.object_logp(path)
        if EMO:
            e2['mot_e'] = MMn.emotion_logp(path)
        new.append((g, feats_, hs, e2))
    FOLD_NEW.append((M_, MMn, HP_, SQ_, new))
print('new configuration: %s' % NEWENV)

GROUPS = {}
if EMO:
    GROUPS['emotion'] = (['w_mot_emotion'], {'w_mot_emotion': [0, 0.5, 1, 2, 3, 5, 8]},
                         ['emotion|HAU'])
if HARN:
    GROUPS['harn'] = (['w_mot_harn', 'w_mot_oi'], {'w_mot_harn': [0, 0.25, 0.5, 1, 2, 3, 5],
                                                   'w_mot_oi': [0, 1, 2, 3, 5]},
                      ['single|HARn', 'object_interaction|HARn'])
START = dict(DEFAULT_W); START.update(json.load(open(BASE_F)))
N_TEST = sum(TESTC.values())
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]


def score(FL, W, keep, cats):
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in FL:
        for g, feats_, hs, e_ in clips:
            if (keep is not None and g.iloc[0].user not in keep) or not g.qt.isin(cats).any():
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats_, MM_, HP_, SQ_, e_)
            for _, r in g[g.qt.isin(cats)].iterrows():
                k = (r['qt'], hs)
                res[k][1] += 1; res[k][0] += a[r.qa_id] == str(r.answer)
    s = sum(TESTC[k] * ok / n for k, (ok, n) in res.items() if n >= 5) / N_TEST
    return s, {('%s|%s' % (k[0], 'seq' if k[1] else 'noseq')): ok / n for k, (ok, n) in res.items()}


def tune(FL, keys, grids, keep, cats):
    best_v, best_s = None, -1.0
    for combo in itertools.product(*[grids[k] for k in keys]):
        W = dict(START); W.update(dict(zip(keys, combo)))
        s = score(FL, W, keep, cats)[0]
        if s > best_s + 1e-12:
            best_s, best_v = s, dict(zip(keys, combo))
    return best_v


def with_(v):
    W = dict(START); W.update(v)
    return W


report = {'config': NEWENV, 'base': BASE_F, 'groups': {}}
accepted = {}
print('\n%-8s %30s %30s  %s' % ('group', 'held-out half 0 (old -> new)', 'held-out half 1 (old -> new)', 'verdict'))
for name, (keys, grids, cats) in GROUPS.items():
    row, gains = [], []
    for i in (0, 1):
        vo = tune(FOLD, keys, grids, HALVES[1 - i], cats)
        vn = tune(FOLD_NEW, keys, grids, HALVES[1 - i], cats)
        so = score(FOLD, with_(vo), HALVES[i], cats)[0]
        sn = score(FOLD_NEW, with_(vn), HALVES[i], cats)[0]
        gains.append(sn - so); row.append('%.4f -> %.4f (%+.4f)' % (so, sn, sn - so))
    ok = all(x > 1e-9 for x in gains)
    vo_all = tune(FOLD, keys, grids, None, cats); vn_all = tune(FOLD_NEW, keys, grids, None, cats)
    acc_old = score(FOLD, with_(vo_all), None, cats)[1]; acc_new = score(FOLD_NEW, with_(vn_all), None, cats)[1]
    report['groups'][name] = dict(keys=keys, categories=cats, held_out_gains=gains, accepted=ok,
                                  weights_old=vo_all, weights_new=vn_all,
                                  acc_all_old=acc_old, acc_all_new=acc_new)
    print('%-8s %30s %30s  %s' % (name, row[0], row[1], 'ACCEPT' if ok else 'reject'))
    for k in sorted(acc_new):
        print('           %-34s all subjects (in-sample): %.4f -> %.4f' % (k, acc_old.get(k, float('nan')), acc_new[k]))
    if ok:
        accepted[name] = vn_all

if DRY:
    print('dry run -- nothing written')
elif accepted:
    W = dict(START)
    for v in accepted.values():
        W.update(v)
    json.dump(W, open(OUT_F, 'w'), indent=1, sort_keys=True)
    report['final_weights'] = {k: v for vv in accepted.values() for k, v in vv.items()}
    report['allowed_categories'] = sorted({c for n in accepted for c in GROUPS[n][2]})
    report['build_env'] = {k: v for k, v in NEWENV.items()
                           if (k in ('CUHKX_EMO', 'CUHKX_C_EMO', 'CUHKX_CLF_EMO') and 'emotion' in accepted)
                           or (k not in ('CUHKX_EMO', 'CUHKX_C_EMO', 'CUHKX_CLF_EMO') and 'harn' in accepted)}
    json.dump(report, open(OUT_F.replace('.json', '_config.json'), 'w'), indent=1)
    env = ' '.join('%s=%s' % kv for kv in report['build_env'].items())
    print('\naccepted %s -> wrote %s and %s' % (sorted(accepted), OUT_F, OUT_F.replace('.json', '_config.json')))
    print('build: %s python src/make_submission3.py submissions/<name>.csv --clip --weights=%s' % (env, OUT_F))
else:
    json.dump(report, open(OUT_F.replace('.json', '_config.json'), 'w'), indent=1)
    print('\nnothing accepted -- report only')
