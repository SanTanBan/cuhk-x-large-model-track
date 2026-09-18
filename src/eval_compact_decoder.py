"""Cross-subject CV of the public "compact clip-graph decoder" (team Fususu's notebook
`phuongncn/lb-0-77777-exact-submission-research-handoff`, Apache-2.0), per stratum, next to
our own per-stratum CV (src/stratum_acc_sub08.json). It shows which structural signals our
model lacks -- chiefly pairwise temporal precedence for `sequence` and exact option-set
memory. Re-implemented from the notebook's description; same 6 subject folds as cv_full.

    python src/eval_compact_decoder.py
"""
import sys, os, re, json, math, itertools, collections
import pandas as pd

LABELS = 'ABCD'
D = 'data'


def norm(v):
    return re.sub(r'\s+', ' ', str(v).strip().lower())


def atoms(v):
    return tuple(norm(p) for p in str(v).split(',') if p.strip())


class ClipGraphDecoder:
    def __init__(self, smoothing=8.0, support_weight=1.0):
        self.smoothing, self.support_weight = smoothing, support_weight
        self.option_ok, self.option_n = collections.Counter(), collections.Counter()
        self.slice_ok, self.slice_n = collections.Counter(), collections.Counter()
        self.set_memory = collections.defaultdict(collections.Counter)
        self.multi_sizes = collections.defaultdict(collections.Counter)
        self.before = collections.Counter()

    def fit(self, df):
        for _, r in df.iterrows():
            ans, src, cat = str(r.answer).strip().upper(), r.source, r.category
            vals = [str(r[l]) for l in LABELS]
            oset = tuple(sorted(map(norm, vals)))
            if cat != 'sequence':
                for l, v in zip(LABELS, vals):
                    k = (src, cat, norm(v))
                    self.option_n[k] += 1; self.slice_n[(src, cat)] += 1
                    if l in ans:
                        self.option_ok[k] += 1; self.slice_ok[(src, cat)] += 1
            if cat == 'multi':
                self.multi_sizes[(src, cat)][len(ans)] += 1
            elif cat == 'sequence' and set(ans) == set(LABELS):
                o = [norm(r[l]) for l in ans]
                for i, a in enumerate(o):
                    for b in o[i + 1:]:
                        self.before[(a, b)] += 1
            else:
                for l in ans:
                    if l in LABELS:
                        self.set_memory[(src, cat, oset)][norm(r[l])] += 1
        return self

    def _logit(self, r, v):
        base = self.slice_ok[(r.source, r.category)] / max(1, self.slice_n[(r.source, r.category)])
        k = (r.source, r.category, norm(v))
        p = (self.option_ok[k] + self.smoothing * base) / (self.option_n[k] + self.smoothing)
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def _sequence(self, r):
        best = None
        for labels in itertools.permutations(LABELS):
            t = [norm(r[l]) for l in labels]
            s = 0.0
            for i, a in enumerate(t):
                for b in t[i + 1:]:
                    x, y = self.before[(a, b)], self.before[(b, a)]
                    s += math.log((x + 1.5) / (x + y + 3.0))
            if best is None or (s, ''.join(labels)) > best:
                best = (s, ''.join(labels))
        return best[1]

    def predict(self, df):
        out, mem_hit = {}, set()
        for _, g in df.groupby('path'):
            support = collections.Counter()
            for _, r in g.iterrows():
                if r.source == 'HAU' and r.category in {'single', 'multi', 'combination', 'sequence'}:
                    vals = [str(r[l]) for l in LABELS]
                    if r.category == 'combination':
                        for v in vals:
                            support.update(atoms(v))
                    else:
                        support.update(map(norm, vals))
            for _, r in g.iterrows():
                if r.category == 'sequence':
                    out[r.qa_id] = self._sequence(r); continue
                vals = [str(r[l]) for l in LABELS]
                oset = tuple(sorted(map(norm, vals)))
                memory = self.set_memory[(r.source, r.category, oset)]
                if sum(memory.values()):
                    mem_hit.add(r.qa_id)
                sc = []
                for v in vals:
                    own = collections.Counter(atoms(v) if r.category == 'combination' else [norm(v)])
                    ext = sum(max(0, support[a] - own[a]) for a in own)
                    sc.append(self._logit(r, v) + math.log1p(memory[norm(v)]) + self.support_weight * ext)
                if r.category == 'multi':
                    sz = self.multi_sizes[(r.source, r.category)]
                    k = max(sz, key=lambda n: (sz[n], -n)) if sz else 1
                    sel = sorted(range(4), key=lambda i: (-sc[i], i))[:k]
                    out[r.qa_id] = ''.join(LABELS[i] for i in sorted(sel))
                else:
                    out[r.qa_id] = LABELS[max(range(4), key=lambda i: (sc[i], -i))]
        return out, mem_hit


if __name__ == '__main__':
    tr = pd.read_csv(os.path.join(D, 'training_qa.csv'), dtype=str, keep_default_na=False,
                     encoding='utf-8-sig')
    te = pd.read_csv(os.path.join(D, 'test_qa.csv'), dtype=str, keep_default_na=False,
                     encoding='utf-8-sig')
    tr['user'] = tr.path.str.extract(r'user(\d+)')[0].astype(int)
    for df in (tr, te):
        df['qt'] = df.category + '|' + df.source
    te['clipdir'] = te.path.str.replace('\\', '/', regex=False).map(lambda p: '/'.join(p.split('/')[:-2]))
    hs_tr = tr.groupby('path').category.apply(lambda c: (c == 'sequence').any())
    hs_te = te.groupby('clipdir').category.apply(lambda c: (c == 'sequence').any())
    TESTC = collections.Counter((r.qt, bool(hs_te[r.clipdir])) for _, r in te.iterrows())

    users = sorted(tr.user.unique())
    folds = [users[i::6] for i in range(6)]
    res = collections.defaultdict(lambda: [0, 0]); mem = collections.defaultdict(lambda: [0, 0, 0])
    for f in folds:
        m = ClipGraphDecoder().fit(tr[~tr.user.isin(f)])
        val = tr[tr.user.isin(f)]
        pred, hit = m.predict(val)
        for _, r in val.iterrows():
            k = (r.qt, bool(hs_tr[r.path]))
            ok = pred[r.qa_id] == str(r.answer).strip().upper()
            res[k][0] += ok; res[k][1] += 1
            if r.qa_id in hit:
                mem[k][0] += 1; mem[k][1] += ok
            mem[k][2] += 1
    ours = json.load(open('src/stratum_acc_sub08.json')) if os.path.exists('src/stratum_acc_sub08.json') else {}
    print('%-26s %-6s %5s %9s %9s %10s' % ('stratum', 'seq?', 'n', 'compact', 'ours', 'mem-cover'))
    tw_c = tw_o = tot = 0
    for k in sorted(res):
        ok, n = res[k]
        o = ours.get('%s|%s' % (k[0], 'seq' if k[1] else 'noseq'), [float('nan')])[0]
        w = TESTC[k]
        tw_c += w * ok / n; tw_o += w * o; tot += w
        print('%-26s %-6s %5d %9.3f %9.3f %10.2f' % (k[0], 'seq' if k[1] else 'noseq', n, ok / n, o,
                                                   mem[k][0] / max(1, mem[k][2])))
    print('test-weighted: compact %.4f | ours %.4f' % (tw_c / tot, tw_o / tot))
