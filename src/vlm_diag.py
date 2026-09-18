"""Is the VLM fusion gain real, and how much of it is the thr_m move?

The VLM-only tuner reported 0.7046 -> 0.7219 on the 320 covered training clips, but it
also moved thr_m from -1.5 to -5 (the grid edge) even though the VLM presence weight came
out at 0 -- so that move is the small subsample re-fitting a non-VLM parameter, not an
effect of the VLM. This separates the two, then measures the VLM weights honestly by
tuning on half the covered subjects and scoring on the other half.

Reuses cv_full.py's module-level preparation (folds, fitted models, cached evidence).
"""
import sys, json, collections
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # defines tr, te, FOLD, VLM, COVERED, TESTC, GRID, ev, answer_clip_fused ...

TUNED = json.load(open('src/best_vlm_W.json'))
BASE = json.load(open('src/best_full_W.json'))
START = dict(DEFAULT_W); START.update(BASE); START['w_vlm_pair'] = 0.0; START['w_gen'] = 0.0
TUNED_THR = dict(TUNED); TUNED_THR['thr_m'] = BASE['thr_m']
VLM_KEYS = [k for k in GRID if k.startswith('w_vlm') or k == 'w_gen']

print('')
print('=== 1) where does the gain come from? (all 320 covered clips) ===')
runs = {}
for name, W in (('current model', START), ('tuned (thr_m=%s)' % TUNED['thr_m'], TUNED),
                ('tuned, thr_m back to %s' % BASE['thr_m'], TUNED_THR)):
    s, res = ev(W)
    runs[name] = res
    print('  %-34s %.4f' % (name, s))
names = list(runs)
print('')
print('  %-40s %8s %8s %8s   n' % ('stratum', 'current', 'tuned', 'tuned-thr'))
for k in sorted(runs[names[0]]):
    row = [runs[n][k][0] / runs[n][k][1] for n in names]
    print('  %-40s %8.3f %8.3f %8.3f  %3d' % (str(k), row[0], row[1], row[2], runs[names[0]][k][1]))


def ev_sub(W, keep_users):
    """Test-weighted score on covered clips of the given subjects only."""
    res = collections.defaultdict(lambda: [0, 0])
    for M_, MM_, HP_, SQ_, clips in FOLD:
        for g, feats, hs, e_ in clips:
            if g.iloc[0].user not in keep_users:
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
        if k in res and res[k][1] >= 5:          # same strata for every config compared
            num += n * res[k][0] / res[k][1]; den += n
    return num / max(den, 1)


def tune_on(keep_users, keys):
    W = dict(START)
    cur = ev_sub(W, keep_users)
    for _ in range(3):
        moved = False
        for k in keys:
            best_v, best_s = W[k], cur
            for v in GRID[k]:
                W2 = dict(W); W2[k] = v
                s = ev_sub(W2, keep_users)
                if s > best_s + 1e-9:
                    best_s, best_v = s, v
            moved = moved or best_v != W[k]
            W[k] = best_v; cur = best_s
        if not moved:
            break
    return W


cov_users = sorted({int(u) for u in tr[tr.qa_id.isin(COVERED)].user})
halves = [set(cov_users[0::2]), set(cov_users[1::2])]
print('')
print('=== 2) nested: tune on one half of the covered subjects, score on the other ===')
print('  covered subjects:', cov_users)
gains = collections.defaultdict(list)
for i in (0, 1):
    fit_u, out_u = halves[1 - i], halves[i]
    base_out = ev_sub(START, out_u)
    for label, keys in (('VLM weights only', VLM_KEYS), ('VLM weights + thr_m', VLM_KEYS + ['thr_m'])):
        W = tune_on(fit_u, keys)
        s = ev_sub(W, out_u)
        gains[label].append(s - base_out)
        shown = {k: W[k] for k in keys if W[k] != START.get(k)}
        print('  half %d  %-22s held-out %.4f -> %.4f  (%+.4f)   changed: %s'
              % (i, label, base_out, s, s - base_out, json.dumps(shown, sort_keys=True)))
print('')
for label, g in gains.items():
    print('  HONEST held-out gain, %-22s %+.4f  (halves: %s)'
          % (label + ':', sum(g) / len(g), ', '.join('%+.4f' % x for x in g)))
