"""Honest CV: precompute per-fold features once, then grid-search weights cheaply.
Stratifies by (qtype, clip-has-sequence) and reweights to TEST proportions."""
import pandas as pd, numpy as np, collections, itertools, sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, Priors, opts, acts_of, lo, known_actions

tr, te = load()
TESTC = collections.Counter()
for p,g in te.groupby('path'):
    hs=(g.category=='sequence').any()
    for _,r in g.iterrows(): TESTC[(r['qt'],hs)]+=1
print("TEST strata (qtype, has_sequence) -> count")
for k,v in sorted(TESTC.items(), key=lambda x:-x[1]): print(f"   {str(k):42s} {v}")
print(f"   TOTAL {sum(TESTC.values())}\n")

def build_features(df, P):
    """One record per question with everything needed to score under any W."""
    F=[]
    for p,g in df.groupby('path'):
        K = known_actions(g); hs = K is not None
        for _,r in g.iterrows():
            qt,O = r['qt'], opts(r)
            rec={'qa_id':r.qa_id,'qt':qt,'hs':hs,'O':O,'ans':str(r.answer) if 'answer' in r else None}
            if qt in ('single|HAU','single|HARn','emotion|HAU','object_interaction|HARn'):
                rec['prior']={L: lo(P.p(qt,r[L])) for L in O}
                rec['inK'] ={L: (K is not None and str(r[L]).strip() in K) for L in O}
            elif qt=='combination|HAU':
                rec['prior']={L: lo(P.p_comb(r[L])) for L in O}
                rec['ovl']  ={L: (len(set(acts_of(r[L]))&K)/max(1,len(set(acts_of(r[L])))) if K else 0.0) for L in O}
            elif qt=='multi|HAU':
                rec['prior']={L: lo(P.p(qt,r[L])) for L in O}
                rec['inK'] ={L: (K is not None and str(r[L]).strip() in K) for L in O}
                rec['kin'],rec['kout']=P.multi_known
            elif qt=='sequence|HAU':
                rec['perm']=P.perm.most_common(1)[0][0] if P.perm else 'ABCD'
            F.append(rec)
    return F

def score(rec, W):
    qt,O=rec['qt'],rec['O']
    if qt=='sequence|HAU': return rec['perm']
    if qt=='combination|HAU':
        return max(O,key=lambda L: rec['prior'][L] + W['comb_known']*rec['ovl'][L])
    if qt=='multi|HAU':
        kin,kout=rec['kin'],rec['kout']; sel=[]
        for L in O:
            s=rec['prior'][L] + (W['multi_known']*(lo(kin) if rec['inK'][L] else lo(kout)) if rec['hs'] else 0.0)
            if s>W['multi_thr']: sel.append(L)
        if not sel: sel=[max(O,key=lambda L:rec['prior'][L])]
        return ''.join(sorted(sel))
    return max(O,key=lambda L: rec['prior'][L] + (W['single_known'] if rec['inK'][L] else 0.0))

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
FOLDS=[]
for f in folds:
    P=Priors(tr[~tr.user.isin(f)],6.0)
    FOLDS.append(build_features(tr[tr.user.isin(f)],P))
print("features built\n")

def evaluate(W):
    res=collections.defaultdict(lambda:[0,0])
    for F in FOLDS:
        for rec in F:
            k=(rec['qt'],rec['hs']); res[k][1]+=1; res[k][0]+= (score(rec,W)==rec['ans'])
    num=den=0
    for k,n in TESTC.items():
        a = res[k][0]/res[k][1] if (k in res and res[k][1]>=5) else 0.25
        num+=n*a; den+=n
    return num/den,res

W0={'single_known':0,'comb_known':0,'multi_known':0,'multi_thr':0.0}
print(f"Priors only                : {evaluate(W0)[0]:.4f}")
best=None
for sk in [8.0]:
    for ck in [2.,4.,6.,10.,16.,25.]:
        for mk in [0.5,0.75,1.0,1.5,2.0,3.0]:
            for mt in [-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]:
                W={'single_known':sk,'comb_known':ck,'multi_known':mk,'multi_thr':mt}
                s,_=evaluate(W)
                if best is None or s>best[0]: best=(s,dict(W))
print(f"Priors + structure (tuned) : {best[0]:.4f}")
print(f"   W = {best[1]}\n")
s,res=evaluate(best[1])
for k,v in sorted(res.items()):
    print(f"   {str(k):42s} acc={v[0]/v[1]:.4f}  n_val={v[1]:5d}  n_test={TESTC.get(k,0)}")
json.dump(best[1],open('src/best_W.json','w'),indent=1)
