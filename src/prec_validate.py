"""Held-out-subject validation of the pairwise-precedence prior for `sequence` (w_prec).

The activities are largely scripted: 64% of action pairs in training sequence answers occur
in one order >= 90% of the time. The public compact decoder's text-only precedence gets
0.409 exact on `sequence` in our cross-subject CV (src/eval_compact_decoder.py) against 0.276
for our motion-based slot model. Here w_prec -- with w_seq re-tuned alongside, since both
feed one permutation score -- is tuned on one half of the subjects and scored on the other,
starting from the sub08 configuration, and accepted only if positive on BOTH halves.
Only sequence answers can change, so only clips with a sequence question are evaluated.

    python src/prec_validate.py [--dry-run]      -> src/best_prec_W.json, src/prec_report.json
"""
import sys, json, collections, itertools

DRY = '--dry-run' in sys.argv
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)          # tr, te, FOLD, VLM, TESTC, DEFAULT_W, answer_clip_fused, ...

START = dict(DEFAULT_W); START.update(json.load(open('src/best_vlm_W.json')))
START['w_prec'] = 0.0
KEYS = ['w_prec', 'w_seq']
GRID = {'w_prec': [0, 0.25, 0.5, 1, 2, 3, 5, 8], 'w_seq': [0, 0.25, 0.5, 1, 2]}
users = sorted(tr.user.unique())
HALVES = [set(users[0::2]), set(users[1::2])]
N_SEQ_TEST = TESTC[('sequence|HAU', True)]
N_TEST = sum(TESTC.values())


def seq_acc(W, keep=None):
    ok = n = 0
    for M_, MM_, HP_, SQ_, clips in FOLD:
        for g, feats, hs, e_ in clips:
            if not hs or (keep is not None and g.iloc[0].user not in keep):
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats, MM_, HP_, SQ_, e_)
            for _, r in g[g.category == 'sequence'].iterrows():
                n += 1; ok += a[r.qa_id] == str(r.answer)
    return ok / max(n, 1), n


def tune(keep):
    best_v, best_s = None, -1.0
    for combo in itertools.product(*[GRID[k] for k in KEYS]):
        W = dict(START); W.update(dict(zip(KEYS, combo)))
        s = seq_acc(W, keep)[0]
        if s > best_s + 1e-12:
            best_s, best_v = s, dict(zip(KEYS, combo))
    return best_v


base = [seq_acc(START, HALVES[i])[0] for i in (0, 1)]
gains, fitted = [], []
for i in (0, 1):
    v = tune(HALVES[1 - i]); fitted.append(v)
    W = dict(START); W.update(v)
    gains.append(seq_acc(W, HALVES[i])[0] - base[i])
    print('held-out half %d: sequence exact %.4f -> %.4f (%+.4f) with %s' % (
        i, base[i], base[i] + gains[-1], gains[-1], v))
ok = all(x > 1e-9 for x in gains)
W_final = dict(START)
if ok:
    W_final.update(tune(None))
a0, n = seq_acc(START); a1, _ = seq_acc(W_final)
print('all subjects (in-sample): sequence exact %.4f -> %.4f (n=%d) | overall test-weighted %+.4f'
      % (a0, a1, n, (a1 - a0) * N_SEQ_TEST / N_TEST))
print('verdict:', 'ACCEPT' if ok else 'reject', '| final', {k: W_final[k] for k in KEYS})
report = dict(held_out_gains=gains, fitted_per_half=fitted, accepted=ok,
              final={k: W_final[k] for k in KEYS}, seq_exact_all=[a0, a1],
              allowed_categories=['sequence|HAU'])
if DRY:
    print('dry run -- nothing written')
elif ok:
    json.dump(W_final, open('src/best_prec_W.json', 'w'), indent=1, sort_keys=True)
    json.dump(report, open('src/prec_report.json', 'w'), indent=1)
    print('wrote src/best_prec_W.json and src/prec_report.json')
else:
    json.dump(report, open('src/prec_report.json', 'w'), indent=1)
    print('not accepted -- report only')
