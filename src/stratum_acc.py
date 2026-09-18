"""Per-stratum cross-subject accuracy of a given weights file, on every VLM-scored clip.

    python src/stratum_acc.py [weights.json]      (default src/best_vlm_W.json -> sub08)

Used to anchor label-free comparisons against other submissions: where our accuracy is
near-perfect, agreement with another file estimates that file's accuracy directly.
"""
import sys, os, json, collections
W_FILES = sys.argv[1:] or ['src/best_vlm_W.json']     # several files share one (slow) prep
sys.argv = ['cv_full.py', '--clip', '--only-vlm']
SRC = open('src/cv_full.py', encoding='utf-8').read().split("if __name__ == '__main__':")[0]
exec(SRC)

for W_FILE in W_FILES:
    W = dict(DEFAULT_W); W.update(json.load(open(W_FILE)))
    s, res = ev(W)
    print('')
    print('weights: %s | test-weighted CV over %d VLM-scored questions: %.4f' % (W_FILE, len(COVERED), s))
    out = {}
    for k, (ok, n) in sorted(res.items()):
        out['%s|%s' % (k[0], 'seq' if k[1] else 'noseq')] = [ok / n, n]
        print('   %-40s acc=%.4f  n=%d' % (str(k), ok / n, n))
    # the sub08 file keeps its old name (src/eval_compact_decoder.py reads it)
    out_f = ('src/stratum_acc_sub08.json' if W_FILE == 'src/best_vlm_W.json' else
             'src/stratum_acc_%s.json' % os.path.basename(W_FILE).replace('best_', '').replace('.json', ''))
    json.dump(out, open(out_f, 'w'), indent=1)
    print('wrote', out_f)
