"""Who is right when our sequence model and precedence-only disagree? (label-based, CV)

The public 0.77777 file's sequence answers agree with our precedence+slot model on 77% of
test questions, so theirs look precedence-based. On training clips (same 6 subject folds as
cv_full) this compares our model -- w_seq * slot log-lik + w_prec * precedence, weights from
src/best_prec_W.json -- against precedence alone, on the questions where the two differ.
That is the evidence for swapping our `sequence` answers into their file.

    python src/seq_disagree.py
"""
import sys, json, itertools, collections
sys.path.insert(0, 'src')
from solver import load, opts
from seq_model import load_segments, SeqModel

W = json.load(open('src/best_prec_W.json'))
ws, wp = W['w_seq'], W['w_prec']
tr, _ = load()
SEG = load_segments()
users = sorted(tr.user.unique())
folds = [users[i::6] for i in range(6)]
c = collections.Counter()
for f in folds:
    SQ = SeqModel(tr[~tr.user.isin(f)], SEG)
    for _, r in tr[tr.user.isin(f) & (tr.category == 'sequence')].iterrows():
        O = opts(r)
        slot, prec = SQ.perm_scorer(r.path, r, O), SQ.prec_scorer(r, O)
        perms = list(itertools.permutations(O))
        ours = max(perms, key=lambda p: (ws * slot(p) if slot else 0.0) + wp * prec(p))
        pre = max(perms, key=lambda p: prec(p))
        ours, pre, y = ''.join(ours), ''.join(pre), str(r.answer)
        c['n'] += 1; c['ours'] += ours == y; c['prec'] += pre == y
        if ours != pre:
            c['disagree'] += 1; c['ours_right'] += ours == y; c['prec_right'] += pre == y
n = c['n']
print('sequence exact (CV, n=%d): ours %.3f | precedence-only %.3f' % (n, c['ours'] / n, c['prec'] / n))
print('disagreements: %d (%.0f%%) -> ours right %d, precedence-only right %d, neither %d'
      % (c['disagree'], 100 * c['disagree'] / n, c['ours_right'], c['prec_right'],
         c['disagree'] - c['ours_right'] - c['prec_right']))
