"""Conditional logit over option strings.

The marginal prior P(correct | option) ignores which options an item was competing
against. A conditional logit learns a strength per option string by maximising the
likelihood that the true option wins its own 4-way contest -- strictly better specified
when some options mostly co-occur with strong rivals.
"""
import pandas as pd, numpy as np, collections, sys, math
sys.path.insert(0,'src'); from solver import load, opts, lo

def fit_cl(rows, vocab, l2=1.0, iters=300, lr=0.5):
    idx={v:i for i,v in enumerate(vocab)}; th=np.zeros(len(vocab))
    for _ in range(iters):
        g=np.zeros(len(vocab))
        for O,ci in rows:
            s=np.array([th[idx[o]] for o in O]); s-=s.max()
            p=np.exp(s); p/=p.sum()
            for k,o in enumerate(O): g[idx[o]] -= p[k]
            g[idx[O[ci]]] += 1.0
        g -= l2*th
        th += lr*g/max(1,len(rows))
    return {v:th[idx[v]] for v in vocab}

def run(cat, qt):
    tr,te=load()
    users=sorted(tr.user.unique()); folds=[users[i::6] for i in range(6)]
    accs={'marginal':0,'condlogit':0}; n=0
    for f in folds:
        trn=tr[(~tr.user.isin(f))&(tr.qt==qt)]; val=tr[(tr.user.isin(f))&(tr.qt==qt)]
        rows=[]; vocab=set()
        app=collections.Counter(); cor=collections.Counter()
        for _,r in trn.iterrows():
            O=[str(r[L]).strip() for L in opts(r)]
            ci=opts(r).index(str(r.answer))
            rows.append((O,ci)); vocab|=set(O)
            for L in opts(r):
                app[str(r[L]).strip()]+=1
                if L in str(r.answer): cor[str(r[L]).strip()]+=1
        vocab=sorted(vocab)
        th=fit_cl(rows,vocab)
        base=sum(cor.values())/max(1,sum(app.values()))
        pri=lambda o: lo((cor[o]+6*base)/(app[o]+6))
        for _,r in val.iterrows():
            O=opts(r); n+=1
            a1=max(O,key=lambda L: pri(str(r[L]).strip()))
            a2=max(O,key=lambda L: th.get(str(r[L]).strip(), 0.0))
            accs['marginal']+= (a1==str(r.answer)); accs['condlogit']+= (a2==str(r.answer))
    print(f"  {cat:22s} n={n:5d}  marginal={accs['marginal']/n:.4f}  cond-logit={accs['condlogit']/n:.4f}")

print("text-only option strength, marginal prior vs conditional logit:")
for cat,qt in [('emotion','emotion|HAU'),('single|HAU','single|HAU'),
               ('object_interaction','object_interaction|HARn')]:
    run(cat,qt)
