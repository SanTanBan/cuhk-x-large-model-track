"""Merge option-order TTA scores into a VLM score file.

The original run gives shift 0 of every letter question (kind 'letter'); a TTA run adds
shifts 1..m-1 (kinds 'letter_s1', 'letter_s2', ...), keyed by the ORIGINAL option letter.
Within each (question, shift) the letter log-probs are renormalised over the offered options
(log-softmax), then averaged per original option. Every option has then appeared in every
position exactly once, so the VLM's letter-position bias cancels.

Non-letter rows (present / pair / perm) pass through unchanged. Letter questions without TTA
shifts keep only shift 0, renormalised -- a per-question constant, which the fusion's
centring removes anyway.

    python src/tta_merge.py <orig.csv> <tta.csv> [<tta2.csv> ...] -o <out.csv>
"""
import sys
import numpy as np
import pandas as pd


def _lse(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))


def merge(orig, ttas):
    d = pd.concat([orig] + list(ttas), ignore_index=True)
    let = d[d.kind.str.startswith('letter')].copy()
    let['shift'] = let.kind.map(lambda k: 0 if k == 'letter' else int(k.split('_s')[1]))
    let = let.drop_duplicates(['qa_id', 'key', 'shift'])
    let['lp'] = let.groupby(['qa_id', 'shift']).score.transform(lambda v: v - _lse(v.values))
    avg = let.groupby(['qa_id', 'key'], as_index=False).lp.mean()
    avg = avg.rename(columns={'lp': 'score'})
    avg['kind'] = 'letter'
    other = orig[~orig.kind.str.startswith('letter')]
    out = pd.concat([other, avg[['qa_id', 'key', 'kind', 'score']]], ignore_index=True)

    # diagnostics: how many shifts each question got, and how often TTA changes the answer
    shifts = let.groupby('qa_id')['shift'].nunique()
    s0 = let[let['shift'] == 0]
    pick0 = s0.loc[s0.groupby('qa_id').lp.idxmax()].set_index('qa_id').key
    pickT = avg.loc[avg.groupby('qa_id').score.idxmax()].set_index('qa_id').key
    common = pick0.index.intersection(pickT.index)
    tta_q = shifts[shifts > 1].index.intersection(common)
    info = {'letter_questions': int(len(shifts)),
            'questions_with_tta': int(len(tta_q)),
            'shift_counts': {int(k): int(v) for k, v in shifts.value_counts().items()},
            'argmax_changed_by_tta': int((pick0.loc[tta_q] != pickT.loc[tta_q]).sum())}
    return out, info


if __name__ == '__main__':
    args = sys.argv[1:]
    if '-o' not in args or len(args) < 3:       # zero TTA files is allowed: identity merge
        sys.exit(__doc__)
    out_path = args[args.index('-o') + 1]
    files = [a for a in args if a not in ('-o', out_path)]
    orig, ttas = pd.read_csv(files[0]), [pd.read_csv(f) for f in files[1:]]
    out, info = merge(orig, ttas)
    out.to_csv(out_path, index=False)
    print('wrote %s: %d rows | %s' % (out_path, len(out), info))
