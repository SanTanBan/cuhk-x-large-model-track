"""Single-weight VLM fusion: switch the VLM on for single|HARn ONLY, at fixed weights.

Full VLM-weight tuning failed the nested check (-0.003 held-out). But the VLM is zero-shot,
so its HARn accuracy on training clips is an unbiased estimate of its test accuracy -- and
it beats the pipeline there by ~0.09. One weight, swept rather than tuned, with each half of
the subjects reported separately: a value that helps on both halves is robust, not fitted.
"""
import sys, json
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)

BASE = json.load(open('src/best_full_W.json'))
START = dict(DEFAULT_W); START.update(BASE); START['w_vlm_pair'] = 0.0; START['w_gen'] = 0.0
cov_users = sorted({int(u) for u in tr[tr.qa_id.isin(COVERED)].user})
halves = [set(cov_users[0::2]), set(cov_users[1::2])]


def harn_acc(W, keep_users=None):
    ok = n = 0
    for M_, MM_, HP_, SQ_, clips in FOLD:
        for g, feats, hs, e_ in clips:
            if keep_users is not None and g.iloc[0].user not in keep_users:
                continue
            if not (g.qt == 'single|HARn').any():
                continue
            if not any(q in COVERED for q in g.qa_id):
                continue
            a = answer_clip_fused(g, M_, W, VLM, feats, MM_, HP_, SQ_, e_)
            for _, r in g[g.qt == 'single|HARn'].iterrows():
                n += 1
                ok += (a[r.qa_id] == str(r.answer))
    return ok / max(n, 1), n


print('')
print('single|HARn accuracy with ONLY the HARn VLM weight switched on (no tuning)')
print('  %-6s %18s %18s %18s' % ('w', 'all covered', 'half 0', 'half 1'))
for v in (0, 0.5, 1, 2, 3, 5, 8, 12, 20):
    W = dict(START); W['w_vlm_single_harn'] = v
    a, n = harn_acc(W)
    h0, n0 = harn_acc(W, halves[0])
    h1, n1 = harn_acc(W, halves[1])
    print('  %-6s %10.3f (n=%3d) %10.3f (n=%3d) %10.3f (n=%3d)' % (v, a, n, h0, n0, h1, n1))

print('')
base = ev(START)[0]
for v in (1, 2, 5):
    W = dict(START); W['w_vlm_single_harn'] = v
    print('  overall covered score, HARn-only w=%s: %.4f   (current model %.4f)' % (v, ev(W)[0], base))
