"""Which features should the HAU action-presence model use?
Measured by its effect on `combination` accuracy. Everything weight-independent is
computed once per fold."""
import pandas as pd, numpy as np, collections, sys
sys.path.insert(0,'src')
from solver import load, opts
from joint import Joint, clip_features
from motion_model import motion_table, HAUPresence
from feats import load_clip_emb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

tr,te=load()
Xm,_=motion_table('motion2')
C=load_clip_emb()
Xc={p:np.concatenate([e.mean(0),e.std(0)]) for p,e in C.items()}
paths=sorted(Xc); Z=StandardScaler().fit_transform(np.array([Xc[p] for p in paths]))
R=PCA(n_components=64,random_state=0).fit_transform(Z)
Xcp={p:R[i] for i,p in enumerate(paths)}
Xboth={p:np.concatenate([Xm[p],Xcp[p]]) for p in Xcp if p in Xm}
SETS={'motion(22)':Xm,'clipPCA(64)':Xcp,'motion+clipPCA(86)':Xboth}

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
# base scores + per-option presence means, computed once
BASE=[]   # (hs, base_scores, {setname: [mean logodds per option]}, correct_index)
for f in folds:
    trn=tr[~tr.user.isin(f)]; M=Joint(trn)
    HPs={n:HAUPresence(trn,X) for n,X in SETS.items()}
    for p,g in tr[tr.user.isin(f)].groupby('path'):
        cb=g[g.category=='combination']
        if len(cb)==0: continue
        K,comb=clip_features(g,M)
        base=[comb['marg'][i]+3.0*comb['pmi'][i]+25.0*comb['ovl'][i] for i in range(len(comb['O']))]
        pres={}
        for n,HP in HPs.items():
            mp=HP.logodds(p)
            pres[n]=[float(np.mean([mp.get(a,0.0) for a in comb['sets'][i]] or [0.0]))
                     for i in range(len(comb['O']))]
        r=cb.iloc[0]
        BASE.append(((g.category=='sequence').any(), base, pres, comb['O'].index(str(r.answer))))
print('cached %d combination questions\n'%len(BASE))

for name in ['none']+list(SETS):
    for w in ([0.0] if name=='none' else [0.25,0.5,1.0,2.0,3.0,5.0]):
        res=collections.defaultdict(lambda:[0,0])
        for hs,base,pres,ci in BASE:
            sc=list(base) if name=='none' else [base[i]+w*pres[name][i] for i in range(len(base))]
            k=int(np.argmax(sc)); res[hs][1]+=1; res[hs][0]+=(k==ci)
        ns=res[False]; hq=res[True]
        print('  %-20s w=%.2f  no-seq=%.4f (n=%d)  seq=%.4f (n=%d)'
              %(name,w,ns[0]/max(1,ns[1]),ns[1],hq[0]/max(1,hq[1]),hq[1]))
