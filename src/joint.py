"""Joint clip-level model.

Per clip we infer a soft distribution over which actions occur in the video, then answer
every question about that clip from it.

  evidence 1  action-content priors        P(option correct | option text)
  evidence 2  action co-occurrence (PMI)   real action-sets co-occur; sampled distractors don't
  evidence 3  the `sequence` question      its 4 options are all true actions of the clip
  evidence 4  the `combination` question   exactly one option IS the clip's action set
  (evidence 5  a VLM over the video        -- pluggable, see predict_clip(vlm=...))
"""
import pandas as pd, numpy as np, collections, itertools, math, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import load, opts, acts_of, lo

def sm(xs):
    m=max(xs); e=[math.exp(x-m) for x in xs]; s=sum(e); return [v/s for v in e]

class Joint:
    def __init__(self, df, alpha=6.0, pmi_k=3.0):
        self.alpha=alpha; self.pmi_k=pmi_k; self.S={}
        for qt in ['single|HAU','single|HARn','multi|HAU','emotion|HAU','object_interaction|HARn']:
            g=df[df.qt==qt]; app=collections.Counter(); cor=collections.Counter()
            for _,r in g.iterrows():
                for L in opts(r):
                    app[r[L]]+=1
                    if L in str(r.answer): cor[r[L]]+=1
            self.S[qt]=(app,cor,(sum(cor.values())/max(1,sum(app.values())) or .25))
        pos=[]; alls=[]
        for _,r in df[df.qt=='combination|HAU'].iterrows():
            for L in opts(r):
                s=set(acts_of(r[L])); alls.append(s)
                if L in str(r.answer): pos.append(s)
        def cnt(ss):
            u=collections.Counter(); p=collections.Counter()
            for S in ss:
                for a in S: u[a]+=1
                for a,b in itertools.combinations(sorted(S),2): p[(a,b)]+=1
            return u,p,len(ss)
        self.pu,self.pp,self.pn = cnt(pos); self.au,self.ap,self.an = cnt(alls)
        self.perm=collections.Counter(df[df.qt=='sequence|HAU'].answer.astype(str))

    NOISE_QT={'single|HARn'}
    def p(self,qt,o):
        app,cor,base=self.S[qt]
        if qt in self.NOISE_QT: return base
        return (cor[o]+self.alpha*base)/(app[o]+self.alpha)
    def marg(self,S):
        k=6.0; return float(np.mean([lo((self.pu[a]+k*.25)/(self.au[a]+k)) for a in set(S)]))
    def pmi(self,S):
        S=sorted(set(S)); k=self.pmi_k
        if len(S)<2: return 0.0
        v=[]
        for a,b in itertools.combinations(S,2):
            pt=(self.pp[(a,b)]+k*.01)/(self.pn+k); pl=(self.ap[(a,b)]+k*.01)/(self.an+k)
            v.append(math.log(pt/max(pl,1e-9)))
        return float(np.mean(v))

def clip_features(g, M):
    """Everything needed to answer this clip's questions under any weight vector."""
    sq=g[g.category=='sequence']
    K=({str(sq.iloc[0][L]).strip() for L in opts(sq.iloc[0])} if len(sq) else set())
    cb=g[g.category=='combination']
    comb=None
    if len(cb):
        r=cb.iloc[0]; O=opts(r)
        comb={'O':O,'sets':[set(acts_of(r[L])) for L in O],
              'marg':[M.marg(acts_of(r[L])) for L in O],
              'pmi':[M.pmi(acts_of(r[L])) for L in O],
              'ovl':[len(set(acts_of(r[L]))&K)/max(1,len(set(acts_of(r[L])))) for L in O]}
    return K, comb

def action_beliefs(K, comb, W):
    """P(action occurs in this clip) from the combination question + sequence options."""
    p={}
    if comb:
        sc=[comb['marg'][i] + W['w_pmi']*comb['pmi'][i] + W['w_ovl']*comb['ovl'][i]
            for i in range(len(comb['O']))]
        pr=sm([W['w_sharp']*x for x in sc])
        for i,S in enumerate(comb['sets']):
            for a in S: p[a]=p.get(a,0.0)+pr[i]
    for a in K: p[a]=1.0
    return p

def answer_clip(g, M, W, vlm=None):
    K, comb = clip_features(g, M)
    bel = action_beliefs(K, comb, W)
    out={}
    for _,r in g.iterrows():
        qt,O = r['qt'], opts(r)
        if qt=='combination|HAU':
            sc=[comb['marg'][i]+W['w_pmi']*comb['pmi'][i]+W['w_ovl']*comb['ovl'][i] for i in range(len(O))]
            out[r.qa_id]=O[int(np.argmax(sc))]
        elif qt in ('single|HAU','single|HARn'):
            out[r.qa_id]=max(O,key=lambda L: lo(M.p(qt,r[L])) + W['w_bel']*lo(min(max(bel.get(str(r[L]).strip(),0.0),1e-4),.9999)))
        elif qt=='multi|HAU':
            sel=[L for L in O if lo(M.p(qt,r[L])) + W['w_bel_m']*lo(min(max(bel.get(str(r[L]).strip(),0.0),1e-4),.9999)) > W['thr_m']]
            if not sel: sel=[max(O,key=lambda L: lo(M.p(qt,r[L])))]
            out[r.qa_id]=''.join(sorted(sel))
        elif qt=='sequence|HAU':
            out[r.qa_id]=M.perm.most_common(1)[0][0] if M.perm else 'ABCD'
        else:
            out[r.qa_id]=max(O,key=lambda L: M.p(qt,r[L]))
    return out
