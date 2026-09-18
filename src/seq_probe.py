"""Do CLIP per-slot embeddings help the temporal-order model over motion segments alone?"""
import numpy as np, sys, collections
sys.path.insert(0,'src')
from solver import load, opts
import seq_model as SM
from feats import combined_S, load_clip_emb
from seq_model import load_segments
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

tr,te=load()
Smot=load_segments()
Sboth=combined_S(n_pca=48)
C=load_clip_emb()
slots={}
for p,e in C.items():
    b=np.linspace(0,len(e),SM.NSEG+1).astype(int)
    slots[p]=np.stack([e[b[k]:max(b[k]+1,b[k+1])].mean(0) for k in range(SM.NSEG)])
paths=sorted(slots); flat=np.concatenate([slots[p] for p in paths])
sc=StandardScaler().fit(flat); pc=PCA(n_components=48,random_state=0).fit(sc.transform(flat))
Sclip={p:pc.transform(sc.transform(slots[p])) for p in paths}

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
for name,S in [('motion(11)',Smot),('clipPCA(48)',Sclip),('motion+clipPCA(59)',Sboth)]:
    for C_ in (0.3,1.0,3.0):
        exact=tot=0; pok=ptot=0
        for f in folds:
            sc2,clf=SM.fit(tr[~tr.user.isin(f)],S,C_)
            for _,r in tr[(tr.user.isin(f))&(tr.category=='sequence')].iterrows():
                O=opts(r); p=SM.predict(sc2,clf,S,r.path,r,O); tot+=1
                if p is None: continue
                exact+=(p==str(r.answer)); t=str(r.answer)
                for i in range(4):
                    for j in range(i+1,4):
                        ptot+=1; pok+= (p.index(t[i])<p.index(t[j]))
        print('  %-20s C=%.1f  exact=%.4f  pairwise=%.4f  (n=%d)'%(name,C_,exact/tot,pok/max(1,ptot),tot))
