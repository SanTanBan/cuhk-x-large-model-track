"""Do real action-sets have co-occurrence structure the distractor sampler lacks?"""
import pandas as pd, numpy as np, collections, itertools, sys, math
sys.path.insert(0,'src'); from solver import load, opts, acts_of, lo
tr,te=load()

def fit_pmi(df):
    pos=[]; allsets=[]
    for _,r in df[df.qt=='combination|HAU'].iterrows():
        for L in opts(r):
            S=set(acts_of(r[L])); allsets.append(S)
            if L in str(r.answer): pos.append(S)
    def counts(sets):
        uni=collections.Counter(); pair=collections.Counter()
        for S in sets:
            for a in S: uni[a]+=1
            for a,b in itertools.combinations(sorted(S),2): pair[(a,b)]+=1
        return uni,pair,len(sets)
    return counts(pos)+counts(allsets)

def pmi_score(M,S,k=3.0):
    pu,pp,pn,au,ap,an=M; S=sorted(set(S))
    if len(S)<2: return 0.0
    v=[]
    for a,b in itertools.combinations(S,2):
        p_true=(pp[(a,b)]+k*0.01)/(pn+k); p_pool=(ap[(a,b)]+k*0.01)/(an+k)
        v.append(math.log(p_true/max(p_pool,1e-9)))
    return float(np.mean(v))

def marg_score(M,S,k=6.0):
    pu,pp,pn,au,ap,an=M
    return float(np.mean([lo((pu[a]+k*0.25)/(au[a]+k)) for a in set(S)]))

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
REC=[]
for f in folds:
    M=fit_pmi(tr[~tr.user.isin(f)]); val=tr[tr.user.isin(f)]
    for p,g in val.groupby('path'):
        hs=(g.category=='sequence').any()
        for _,r in g[g.category=='combination'].iterrows():
            O=opts(r)
            feats=[(marg_score(M,acts_of(r[L])), pmi_score(M,acts_of(r[L]))) for L in O]
            REC.append((hs,feats,O.index(str(r.answer))))
print("scored %d combination questions" % len(REC))
print()
for wpmi in [0.0,0.25,0.5,1.0,2.0,3.0,5.0,8.0]:
    hs=[0,0]; ns=[0,0]
    for h,feats,ci in REC:
        k=max(range(len(feats)), key=lambda i: feats[i][0]+wpmi*feats[i][1])
        t = hs if h else ns
        t[1]+=1; t[0]+= (k==ci)
    print("  w_pmi=%4.2f  no-seq clips=%.4f (n=%d)   seq clips=%.4f (n=%d)" % (wpmi,ns[0]/ns[1],ns[1],hs[0]/hs[1],hs[1]))
