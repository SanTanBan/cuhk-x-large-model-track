"""Does CLIP add over frame-difference motion features? Same protocol, three feature sets."""
import pandas as pd, numpy as np, collections, sys, math
sys.path.insert(0,'src')
from solver import load, opts, lo
from motion_model import motion_table
from feats import combined_X, load_clip_emb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

tr,te=load()
Xm,_=motion_table('motion2')
Xc={}
C=load_clip_emb()
for p,e in C.items(): Xc[p]=np.concatenate([e.mean(0),e.std(0)])
Xb=combined_X()
SETS={'motion(22)':Xm,'clip(1024)':Xc,'motion+clipPCA(118)':Xb}

def probe(qt, label_fn, restrict_opts, Cs=(0.03,0.1,0.3,1.0)):
    users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
    print('--- %s ---'%qt)
    for name,X in SETS.items():
        best=0; bC=None
        for Creg in Cs:
            ok=n=0
            for f in folds:
                trn=tr[(~tr.user.isin(f))&(tr.qt==qt)]; trn=trn[trn.path.isin(X)]
                val=tr[(tr.user.isin(f))&(tr.qt==qt)]; val=val[val.path.isin(X)]
                lab={r.path:label_fn(r) for _,r in trn.iterrows()}
                cnt=collections.Counter(lab.values()); keep={k for k,v in cnt.items() if v>=5}
                ps=[p for p in lab if lab[p] in keep]
                if len(ps)<40: continue
                A=np.array([X[p] for p in ps]); y=np.array([lab[p] for p in ps])
                sc=StandardScaler().fit(A)
                clf=LogisticRegression(max_iter=2000,C=Creg).fit(sc.transform(A),y)
                idx={c:i for i,c in enumerate(clf.classes_)}
                for _,r in val.iterrows():
                    lp=clf.predict_log_proba(sc.transform(X[r.path].reshape(1,-1)))[0]
                    O=opts(r); n+=1
                    pick=max(O,key=lambda L: lp[idx[str(r[L]).strip()]] if str(r[L]).strip() in idx else -1e9)
                    ok+= (pick==str(r.answer))
            if n and ok/n>best: best, bC = ok/n, Creg
        print('   %-22s acc=%.4f (C=%s, n=%d)'%(name,best,bC,n))

# label by the option TEXT (constant per action folder) so it matches at scoring time
probe('single|HARn', lambda r: str(r[str(r.answer)]).strip(), True)
probe('emotion|HAU', lambda r: str(r[str(r.answer)]).strip(), True)
probe('object_interaction|HARn', lambda r: str(r[str(r.answer)]).strip(), True)
