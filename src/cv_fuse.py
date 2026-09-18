"""CV for the fused model. Works with or without VLM scores; tunes all weights jointly."""
import pandas as pd, numpy as np, collections, sys, os, json
sys.path.insert(0,'src')
from solver import load, opts
from joint import Joint, clip_features
from fuse import answer_clip_fused, load_vlm, DEFAULT_W

tr,te=load()
VLM_TRAIN = load_vlm('vlm/vlm_scores_train.csv')
print("VLM training evidence for %d questions" % len(VLM_TRAIN))
TESTC=collections.Counter()
for p,g in te.groupby('path'):
    hs=(g.category=='sequence').any()
    for _,r in g.iterrows(): TESTC[(r['qt'],hs)]+=1

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
FOLD=[]
for f in folds:
    M=Joint(tr[~tr.user.isin(f)]); val=tr[tr.user.isin(f)]
    clips=[]
    for p,g in val.groupby('path'):
        clips.append((g, clip_features(g,M), (g.category=='sequence').any()))
    FOLD.append((M,clips))
print("prepared %d folds" % len(FOLD))
# restrict scoring to clips the VLM actually covered, when VLM evidence is present
COVER = set(VLM_TRAIN.keys())

def ev(W, only_vlm=False, verbose=False):
    res=collections.defaultdict(lambda:[0,0])
    for M,clips in FOLD:
        for g,feats,hs in clips:
            if only_vlm and not any(q in COVER for q in g.qa_id): continue
            a=answer_clip_fused(g,M,W,VLM_TRAIN,feats)
            for _,r in g.iterrows():
                k=(r['qt'],hs); res[k][1]+=1; res[k][0]+= (a[r.qa_id]==str(r.answer))
    num=den=0
    for k,n in TESTC.items():
        acc=res[k][0]/res[k][1] if (k in res and res[k][1]>=5) else .25
        num+=n*acc; den+=n
    if verbose:
        for k,v in sorted(res.items()):
            print(f"   {str(k):40s} acc={v[0]/v[1]:.4f} n_val={v[1]:5d} n_test={TESTC.get(k,0)}")
    return num/den,res

if __name__=='__main__':
    W=dict(DEFAULT_W)
    base=ev(W)[0]
    print(f"\nno-VLM baseline (should match joint model 0.6362): {base:.4f}")
    if not VLM_TRAIN:
        print("No VLM scores yet -- run the Kaggle notebook with SPLIT=train first.")
        sys.exit(0)
    GRID=dict(w_vlm_single=[0,.5,1,2,3,5,8], w_vlm_single_harn=[0,.5,1,2,3,5,8,12],
              w_vlm_emotion=[0,.5,1,2,3,5,8], w_vlm_oi=[0,.5,1,2,3,5],
              w_vlm_comb=[0,.5,1,2,3,5], w_vlm_pres=[0,.1,.25,.5,1,2],
              w_gen=[0,1,2,4], w_bel=[2,3,5,8], w_bel_m=[.5,1,2], thr_m=[-2,-1.5,-1,-.5,0])
    cur=ev(W,only_vlm=True)[0]; print(f"start (VLM-covered clips only): {cur:.4f}")
    for it in range(3):
        for k,vals in GRID.items():
            bv,bs=W[k],cur
            for v in vals:
                W2=dict(W); W2[k]=v; s=ev(W2,only_vlm=True)[0]
                if s>bs: bs,bv=s,v
            W[k]=bv; cur=bs
        print(f"  pass {it+1}: {cur:.4f}")
    print(f"\nFUSED (VLM-covered clips) = {cur:.4f}\nW = {W}\n")
    ev(W,only_vlm=True,verbose=True)
    json.dump(W,open('src/best_fused_W.json','w'),indent=1)
