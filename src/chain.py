"""If we know a clip's action set S (from the combination answer), how well does it
determine the single / multi answers?  ORACLE upper bound first."""
import pandas as pd, numpy as np, collections, sys
sys.path.insert(0,'src'); from solver import load, opts, acts_of
tr,te=load()
st=collections.defaultdict(lambda:[0,0]); det=collections.Counter()
for p,g in tr.groupby('path'):
    cb=g[g.category=='combination']
    if len(cb)==0: continue
    rc=cb.iloc[0]; S=set(acts_of(rc[str(rc.answer)]))
    sq=g[g.category=='sequence']
    K=({str(sq.iloc[0][L]).strip() for L in opts(sq.iloc[0])} if len(sq) else set())
    hs=len(sq)>0
    for _,q in g[g.category=='single'].iterrows():
        inS=[L for L in opts(q) if str(q[L]).strip() in S]
        det[('single_#opts_in_S',len(inS))]+=1
        if len(inS)==1:
            st[('single|S',hs)][1]+=1; st[('single|S',hs)][0]+= (inS[0]==str(q.answer))
    for _,q in g[g.category=='multi'].iterrows():
        pr=''.join(sorted(L for L in opts(q) if str(q[L]).strip() in S))
        st[('multi|S',hs)][1]+=1; st[('multi|S',hs)][0]+= (pr==str(q.answer))
        # union of S and sequence options
        U=S|K
        pr2=''.join(sorted(L for L in opts(q) if str(q[L]).strip() in U))
        st[('multi|S+K',hs)][1]+=1; st[('multi|S+K',hs)][0]+= (pr2==str(q.answer))
    for _,q in g[g.category=='single'].iterrows():
        U=S|K
        inU=[L for L in opts(q) if str(q[L]).strip() in U]
        if len(inU)==1:
            st[('single|S+K',hs)][1]+=1; st[('single|S+K',hs)][0]+= (inU[0]==str(q.answer))
print("ORACLE: S = true combination-answer action set; K = sequence options")
for k,v in sorted(st.items(), key=lambda x:str(x[0])):
    print(f"   {str(k):24s} acc={v[0]/v[1]:.4f}  n={v[1]}")
print("  ",dict(det))
