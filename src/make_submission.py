"""Fit priors on all training data, apply structural constraints, write a submission."""
import pandas as pd, sys, os, json, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, Priors, predict_df, opts

tr, te = load()
W = json.load(open(os.path.join(os.path.dirname(__file__),'best_W.json')))
print("weights:", W)
P = Priors(tr, alpha=6.0)
pred = predict_df(te, P, vlm=None, W=W)

sub = pd.read_csv(os.path.join('data','sample_submission.csv'))
sub['prediction'] = sub.qa_id.map(pred)

# ---- validation ----
assert sub.prediction.notna().all(), f"missing: {sub[sub.prediction.isna()].qa_id.tolist()[:5]}"
assert len(sub)==682 and list(sub.columns)==['qa_id','prediction']
cat = dict(zip(te.qa_id, te.category)); nopt = {r.qa_id: set(opts(r)) for _,r in te.iterrows()}
bad=[]
for _,r in sub.iterrows():
    c, a, O = cat[r.qa_id], str(r.prediction), nopt[r.qa_id]
    if not set(a) <= O: bad.append((r.qa_id,c,a,'option not offered'))
    elif c=='sequence' and (len(a)!=4 or set(a)!=O): bad.append((r.qa_id,c,a,'not a full permutation'))
    elif c=='multi' and (list(a)!=sorted(a) or len(a)<1): bad.append((r.qa_id,c,a,'multi not sorted/empty'))
    elif c in ('single','combination','emotion','object_interaction') and len(a)!=1: bad.append((r.qa_id,c,a,'not single letter'))
print("format violations:", bad[:10], f"({len(bad)} total)")
out = os.path.join('submissions','sub01_priors_structure.csv')
sub.to_csv(out, index=False)
print(f"\nwrote {out}")
print("prediction length distribution by category:")
tmp = sub.merge(te[['qa_id','category']], on='qa_id')
print(tmp.groupby('category').prediction.apply(lambda s: collections.Counter(s.astype(str).str.len()).most_common()).to_string())
print("\nfirst rows:"); print(sub.head(6).to_string(index=False))
