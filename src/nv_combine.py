"""Combine non-visual groups validated in separate `src/nv_validate.py` runs into one weights
file plus the CUHKX_* environment its build needs (09-11).

Emotion evidence feeds only `emotion|HAU`; HARn evidence feeds only `single|HARn` and
`object_interaction|HARn`. So groups validated in separate runs combine safely. Each group's
weights come from the run where that group was accepted, never from another run's weights
file: e.g. the IMU-logistic run's file carries an emotion weight tuned for a different model.

    python src/nv_combine.py --emo=src/best_nv_emorf_W_config.json --harn=<run>_config.json
                             [--base=src/best_struct4_W.json] [--out=src/best_nv_final_W.json]
"""
import sys, json

args = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
BASE = args.get('base', 'src/best_struct4_W.json')
OUT = args.get('out', 'src/best_nv_final_W.json')
EMO_KEYS = ('CUHKX_EMO', 'CUHKX_C_EMO', 'CUHKX_CLF_EMO')

W = json.load(open(BASE))
env, cats, used = {}, set(), {}
for grp, key in (('emotion', 'emo'), ('harn', 'harn')):
    if key not in args:
        continue
    rep = json.load(open(args[key]))
    g = rep['groups'][grp]
    assert g['accepted'], '%s group was not accepted in %s' % (grp, args[key])
    W.update(g['weights_new'])
    used[grp] = g['weights_new']
    cats |= set(g['categories'])
    for k, v in rep['config'].items():
        if (k in EMO_KEYS) == (grp == 'emotion'):
            env[k] = v
json.dump(W, open(OUT, 'w'), indent=1, sort_keys=True)
cfg = {'build_env': env, 'allowed_categories': sorted(cats), 'weights': used,
       'sources': {k: args[k] for k in ('emo', 'harn') if k in args}, 'base': BASE}
json.dump(cfg, open(OUT.replace('.json', '_config.json'), 'w'), indent=1)
print('wrote %s | weights %s' % (OUT, used))
print('build: %s python src/make_submission3.py submissions/<name>.csv --clip --weights=%s'
      % (' '.join('%s=%s' % kv for kv in env.items()), OUT))
print('leak guard: changes allowed only in', sorted(cats))
