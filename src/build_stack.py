"""Build one validated action-category group on top of sub14e (09-13 handover). Never submits.

    python src/build_stack.py nb|nbwide|pres|presimuskel OUT.csv [--c-pres=C]

  nb    src/best_nb_W_report.json, src/best_nb_W.json, src/best_nb_W_config.json
  pres  src/best_nv_pres_W_skel_report.json, src/best_nv_pres_W_skel.json, src/best_nv_pres_W_skel_config.json

Refuses unless the report is accepted AND its held-out gains are positive on both halves. For
`pres` the build needs CUHKX_C_PRES: from the config's build_env, else --c-pres (the default 1.0
would silently build the wrong model), else it refuses. Steps:
  1. make_submission3.py with the config's build_env and weights -> OUT_base.csv; leak guard vs
     sub12 to the config's allowed categories.
  2. build_session.py adds sub14e's emotion (norm+sess, w 5, tau 0.5, order2 -4, nbopt 0.25)
     -> OUT.csv; its own leak guard allows emotion only vs OUT_base.csv.
  3. OUT.csv vs sub14e may differ only in the allowed categories.
"""
import sys, os, json, subprocess, hashlib
import pandas as pd

GROUP, OUT = sys.argv[1], sys.argv[2]
C_PRES = next((a.split('=', 1)[1] for a in sys.argv[3:] if a.startswith('--c-pres=')), None)
STEM = {'nb': 'src/best_nb_W', 'nbwide': 'src/best_nb_wide_W', 'pres': 'src/best_nv_pres_W_skel', 'presimuskel': 'src/best_nv_pres_W_imuskel',
        'nbpres': 'src/best_nbpres_W'}[GROUP]
REPORT, WEIGHTS, CONFIG = STEM + '_report.json', STEM + '.json', STEM + '_config.json'
SUB12, SUB14E = 'submissions/sub12_nv.csv', 'submissions/sub14e_emo-order2nb.csv'
EMO_ARGS = ['--emo=norm+sess', '--w-emo=5', '--tau=0.5', '--order2=-4', '--nbopt=0.25']


def md5(f):
    return hashlib.md5(open(f, 'rb').read()).hexdigest()


def diff_cats(a, b):
    te = pd.read_csv('data/test_qa.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    qt = (te.category + '|' + te.source).set_axis(te.qa_id)
    x = pd.read_csv(a, dtype=str).set_index('qa_id').prediction.reindex(qt.index)
    y = pd.read_csv(b, dtype=str).set_index('qa_id').prediction.reindex(qt.index)
    return qt[x != y].value_counts().to_dict()


rep = json.load(open(REPORT))
gains = rep.get('held_out_gains') or []
print('report %s: accepted=%s held_out_gains=%s' % (REPORT, rep.get('accepted'), gains))
if not (rep.get('accepted') and len(gains) == 2 and all(g > 0 for g in gains)):
    sys.exit('REFUSED: not positive on both halves')
cfg = json.load(open(CONFIG))
env_add, allowed = dict(cfg.get('build_env', {})), set(cfg.get('allowed_categories', []))
if GROUP.startswith('pres') and 'CUHKX_C_PRES' not in env_add:
    if C_PRES is None:
        sys.exit('REFUSED: CUHKX_C_PRES not in the config; pass --c-pres=<C from the report>')
    env_add['CUHKX_C_PRES'] = C_PRES
print('build_env %s | allowed %s' % (env_add, sorted(allowed)))

env = {k: v for k, v in os.environ.items() if not k.startswith('CUHKX_')}
env.update(env_add)
env['PYTHONIOENCODING'] = 'utf-8'
base = OUT.replace('.csv', '_base.csv')
r = subprocess.run([sys.executable, 'src/make_submission3.py', base, '--clip', '--weights=' + WEIGHTS],
                   env=env, capture_output=True, text=True)
print('\n'.join(l for l in r.stdout.splitlines()[-6:]))
if r.returncode != 0 or not os.path.exists(base):
    sys.exit('make_submission3 failed:\n' + r.stderr[-2000:])
d1 = diff_cats(base, SUB12)
print('step 1 leak guard: %s vs sub12 changed %s -> %s' % (base, d1, 'OK' if set(d1) <= allowed else 'LEAK'))
if not set(d1) <= allowed:
    sys.exit('REFUSED: step-1 changes outside the allowed categories')

env2 = {k: v for k, v in os.environ.items() if not k.startswith('CUHKX_')}
env2['PYTHONIOENCODING'] = 'utf-8'
r = subprocess.run([sys.executable, 'src/build_session.py', OUT] + EMO_ARGS + ['--base-sub=' + base],
                   env=env2, capture_output=True, text=True)
print('\n'.join(l for l in r.stdout.splitlines() if any(k in l for k in ('REGRESSION', 'LEAK', 'emotion', 'wrote'))))
if r.returncode != 0 or not os.path.exists(OUT):
    sys.exit('build_session failed:\n' + r.stderr[-2000:])

d3 = diff_cats(OUT, SUB14E)
ok = set(d3) <= allowed
print('step 3: %s vs sub14e changed %s -> %s | md5 %s' % (OUT, d3, 'OK' if ok else 'LEAK', md5(OUT)))
if not ok:
    sys.exit('REFUSED: final changes outside the allowed categories')
print('READY (not submitted): count(%s) - 278 isolates the %s change' % (OUT, GROUP))
