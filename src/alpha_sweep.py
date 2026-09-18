"""Per-category smoothing sweep for the option-content priors."""
import pandas as pd, numpy as np, collections, sys
sys.path.insert(0,'src'); from solver import load, opts
tr,te=load()
users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
for qt in ['object_interaction|HARn','emotion|HAU']:
    print(f"--- {qt} ---")
    for a in [0.0,0.25,0.5,1.0,2.0,4.0,6.0,10.0,20.0]:
        ok=n=0
        for f in folds:
            trn=tr[~tr.user.isin(f)]; val=tr[(tr.user.isin(f))&(tr.qt==qt)]
            g=trn[trn.qt==qt]; app=collections.Counter(); cor=collections.Counter()
            for _,r in g.iterrows():
                for L in opts(r):
                    app[r[L]]+=1
                    if L in str(r.answer): cor[r[L]]+=1
            base=sum(cor.values())/max(1,sum(app.values()))
            P=lambda o:(cor[o]+a*base)/(app[o]+a)
            for _,r in val.iterrows():
                O=opts(r); n+=1; ok+= (max(O,key=lambda L:P(r[L]))==str(r.answer))
        print(f"   alpha={a:5.2f}  acc={ok/n:.4f}  (n={n})")
