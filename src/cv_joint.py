import pandas as pd, numpy as np, collections, sys, os, json, math
sys.path.insert(0,'src')
from solver import load, opts, acts_of, lo
from joint import Joint, clip_features, sm
tr,te=load()
TESTC=collections.Counter()
for p,g in te.groupby('path'):
    hs=(g.category=='sequence').any()
    for _,r in g.iterrows(): TESTC[(r['qt'],hs)]+=1

def pack(df, M):
    """Precompute every weight-independent quantity for each clip."""
    out=[]
    for p,g in df.groupby('path'):
        K,comb = clip_features(g,M)
        qs=[]
        for _,r in g.iterrows():
            qt,O=r['qt'],opts(r)
            q={'qt':qt,'O':O,'ans':str(r.answer),'txt':{L:str(r[L]).strip() for L in O}}
            if qt!='sequence|HAU' and qt!='combination|HAU':
                q['prior']={L: lo(M.p(qt,r[L])) for L in O}
            qs.append(q)
        out.append({'K':K,'comb':comb,'qs':qs,'hs':len(K)>0,
                    'perm':M.perm.most_common(1)[0][0] if M.perm else 'ABCD'})
    return out

users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
PACKS=[]
for f in folds:
    M=Joint(tr[~tr.user.isin(f)])
    PACKS.append(pack(tr[tr.user.isin(f)],M))
print("packed %d clips" % sum(len(p) for p in PACKS))

def ev(W, verbose=False):
    res=collections.defaultdict(lambda:[0,0])
    for P in PACKS:
        for c in P:
            comb,K=c['comb'],c['K']
            bel={}
            if comb:
                sc=[comb['marg'][i]+W['w_pmi']*comb['pmi'][i]+W['w_ovl']*comb['ovl'][i] for i in range(len(comb['O']))]
                pr=sm([W['w_sharp']*x for x in sc])
                for i,S in enumerate(comb['sets']):
                    for a in S: bel[a]=bel.get(a,0.)+pr[i]
                cbest=comb['O'][int(np.argmax(sc))]
            for a in K: bel[a]=1.0
            B=lambda t: lo(min(max(bel.get(t,0.0),1e-4),.9999))
            for q in c['qs']:
                qt,O=q['qt'],q['O']
                if qt=='combination|HAU': a=cbest
                elif qt=='sequence|HAU': a=c['perm']
                elif qt in ('single|HAU','single|HARn'):
                    a=max(O,key=lambda L: q['prior'][L]+W['w_bel']*B(q['txt'][L]))
                elif qt=='multi|HAU':
                    sel=[L for L in O if q['prior'][L]+W['w_bel_m']*B(q['txt'][L])>W['thr_m']]
                    if not sel: sel=[max(O,key=lambda L:q['prior'][L])]
                    a=''.join(sorted(sel))
                else: a=max(O,key=lambda L:q['prior'][L])
                k=(qt,c['hs']); res[k][1]+=1; res[k][0]+= (a==q['ans'])
    num=den=0
    for k,n in TESTC.items():
        acc=res[k][0]/res[k][1] if (k in res and res[k][1]>=5) else .25
        num+=n*acc; den+=n
    if verbose:
        for k,v in sorted(res.items()): print(f"   {str(k):40s} acc={v[0]/v[1]:.4f} n_val={v[1]:5d} n_test={TESTC.get(k,0)}")
    return num/den,res

W=dict(w_pmi=2.0,w_ovl=6.0,w_sharp=1.0,w_bel=2.0,w_bel_m=1.0,thr_m=0.0)
GRID=dict(w_pmi=[0,.5,1,2,3,5,8], w_ovl=[0,2,4,6,10,16,25], w_sharp=[.25,.5,1,2,4,8],
          w_bel=[0,.5,1,2,3,5,8], w_bel_m=[0,.25,.5,1,2,3,5], thr_m=[-1.5,-1,-.5,-.25,0,.25,.5,1])
cur=ev(W)[0]; print(f"start {cur:.4f}")
for it in range(4):
    for k,vals in GRID.items():
        bv,bs=W[k],cur
        for v in vals:
            W2=dict(W); W2[k]=v; s=ev(W2)[0]
            if s>bs: bs,bv=s,v
        W[k]=bv; cur=bs
    print(f"  pass {it+1}: {cur:.4f}  {W}")
print(f"\nBEST EXPECTED PRIVATE LB = {cur:.4f}\nW = {W}\n")
ev(W,verbose=True)
json.dump(W,open('src/best_joint_W.json','w'),indent=1)
