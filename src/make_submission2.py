"""Submission from the joint clip-level model (priors + PMI co-occurrence + structural leaks)."""
import pandas as pd, numpy as np, collections, sys, os, json
sys.path.insert(0,'src')
from solver import load, opts
from joint import Joint, answer_clip
tr,te=load()
W=json.load(open('src/best_joint_W.json')); print("W:",W)
M=Joint(tr)
pred={}
for p,g in te.groupby('path'): pred.update(answer_clip(g,M,W))
sub=pd.read_csv('data/sample_submission.csv')
sub['prediction']=sub.qa_id.map(pred)
assert sub.prediction.notna().all() and len(sub)==682
cat=dict(zip(te.qa_id,te.category)); nopt={r.qa_id:set(opts(r)) for _,r in te.iterrows()}
bad=[]
for _,r in sub.iterrows():
    c,a,O=cat[r.qa_id],str(r.prediction),nopt[r.qa_id]
    if not set(a)<=O: bad.append((r.qa_id,'not offered',a))
    elif c=='sequence' and set(a)!=O: bad.append((r.qa_id,'bad perm',a))
    elif c=='multi' and list(a)!=sorted(a): bad.append((r.qa_id,'unsorted',a))
    elif c in ('single','combination','emotion','object_interaction') and len(a)!=1: bad.append((r.qa_id,'len',a))
print("format violations:",len(bad),bad[:5])
sub.to_csv('submissions/sub02_joint.csv',index=False)
print("wrote submissions/sub02_joint.csv")
old=pd.read_csv('submissions/sub01_priors_structure.csv')
m=old.merge(sub,on='qa_id',suffixes=('_1','_2')).merge(te[['qa_id','category']],on='qa_id')
print("\nchanged vs sub01: %d/682" % (m.prediction_1!=m.prediction_2).sum())
print(m.groupby('category').apply(lambda d:(d.prediction_1!=d.prediction_2).mean().round(3),include_groups=False).to_string())
