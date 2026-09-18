"""Fuse VLM evidence into the joint clip-level model.

The VLM never decides anything on its own. Its scores enter as one more log-odds term
alongside the option-content prior, the action co-occurrence (PMI) score and the
structural constraints, with weights tuned on held-out training subjects.

VLM score file (long format, produced by kaggle_notebook/vlm_infer.py):
    qa_id, key, kind, score
    kind=letter   key=option letter   score=log P(letter)
    kind=present  key=option letter   score=logit(Yes)-logit(No) for "does this action occur"
    kind=perm     key=4-letter order  score=1.0
    kind=pair     key="A<B"           score>0 means A judged before B
"""
import pandas as pd, numpy as np, collections, itertools, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import opts, acts_of, lo
from joint import sm


def load_vlm(path):
    """-> {qa_id: {'letter': {L: lp}, 'present': {L: s}, 'perm': str, 'pair': {(a,b): s}}}"""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    V = collections.defaultdict(lambda: {'letter': {}, 'present': {}, 'perm': None, 'pair': {}})
    for r in df.itertuples(index=False):
        e = V[r.qa_id]
        if r.kind == 'letter':
            e['letter'][r.key] = float(r.score)
        elif r.kind == 'present':
            e['present'][r.key] = float(r.score)
        elif r.kind == 'perm':
            e['perm'] = str(r.key)
        elif r.kind == 'pair':
            a, b = str(r.key).split('<')
            e['pair'][(a, b)] = float(r.score)
    return dict(V)


def _centred(d, keys):
    """Centre log-probs over the offered options so the scale is comparable across questions."""
    if not d:
        return {k: 0.0 for k in keys}
    vals = [d.get(k, min(d.values())) for k in keys]
    m = float(np.mean(vals))
    return {k: (d.get(k, min(d.values())) - m) for k in keys}


def best_perm(O, pair, gen, w_gen, perm_score=None, w_seq=0.0, w_pair=1.0):
    """Kemeny-style: pick the permutation of O maximising pairwise agreement,
    plus (optionally) the temporal-order model's log-likelihood for that permutation."""
    best, bs = None, -1e18
    for p in itertools.permutations(O):
        s = w_seq * perm_score(p) if perm_score else 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                a, b = p[i], p[j]          # a before b in this candidate
                if (a, b) in pair:
                    s += w_pair * pair[(a, b)]
                elif (b, a) in pair:
                    s -= w_pair * pair[(b, a)]
        if gen and ''.join(p) == gen:
            s += w_gen
        if s > bs:
            bs, best = s, p
    return ''.join(best)


def clip_evidence(g, MM=None, HP=None, SQ=None, ZS=None):
    """Run every fitted model over this clip ONCE.

    Weight tuning evaluates a clip hundreds of times; without this the sklearn
    predict calls dominate (40 per clip per evaluation) and tuning never finishes.
    """
    path = g.iloc[0].path
    ev = {'mot_h': MM.harn_action_logp(path) if MM else {},
          'mot_e': MM.emotion_logp(path) if MM else {},
          'mot_o': MM.object_logp(path) if MM else {},
          'mot_p': HP.logodds(path) if HP else {},
          'zs_h': ZS.harn_sims(path) if ZS else {},
          'zs_p': ZS.presence(path) if ZS else {},
          'seq': {}, 'prec': {}}
    if SQ is not None:
        for _, r in g[g.category == 'sequence'].iterrows():
            f = SQ.perm_scorer(path, r, opts(r))
            if f is not None:
                ev['seq'][r.qa_id] = {''.join(p): f(p)
                                      for p in itertools.permutations(opts(r))}
            h = SQ.prec_scorer(r, opts(r)) if hasattr(SQ, 'prec_scorer') else None
            if h is not None:
                ev['prec'][r.qa_id] = {''.join(p): h(p)
                                       for p in itertools.permutations(opts(r))}
    return ev


def answer_clip_fused(g, M, W, V, feats=None, MM=None, HP=None, SQ=None, ev=None,
                      ZS=None):
    """Same contract as joint.answer_clip, plus:
        V   VLM evidence (may be empty)
        MM/HP/SQ  fitted motion / presence / temporal-order models (may be None)
        ev  pre-computed clip_evidence(), which is what makes tuning tractable
    `feats` is an optional pre-computed (K, comb) pair -- clip_features is the slow part."""
    from joint import clip_features
    K, comb = feats if feats is not None else clip_features(g, M)
    path = g.iloc[0].path
    if ev is None:
        ev = clip_evidence(g, MM, HP, SQ, ZS)
    mot_h, mot_e, mot_o, mot_p = ev['mot_h'], ev['mot_e'], ev['mot_o'], ev['mot_p']
    zs_h, zs_p = ev.get('zs_h', {}), ev.get('zs_p', {})

    # ---- action beliefs: combination question + sequence leak + VLM presence votes ----
    bel = {}
    cbest = None
    if comb:
        sc = [comb['marg'][i] + W['w_pmi'] * comb['pmi'][i] + W['w_ovl'] * comb['ovl'][i]
              for i in range(len(comb['O']))]
        # routing (09-11): the true combination's actions recur among the clip's multi and
        # single options (47% / 26% of them, vs 21% / 15% for distractors' actions).
        # w_ovl_seq_scale scales it on clips with a sequence question (their overlap leak is
        # already strong); rt keeps the term so w_ovl_bel=0 can leave it out of the beliefs.
        sca = W.get('w_ovl_seq_scale', 1.0) if K else 1.0
        wom, wos = W.get('w_ovl_m', 0.0) * sca, W.get('w_ovl_s', 0.0) * sca
        rt = [0.0] * len(comb['O'])
        if wom or wos:
            nz = lambda x: ' '.join(str(x).strip().lower().split())
            mo = {nz(q[L]) for _, q in g[g.category == 'multi'].iterrows() for L in opts(q)}
            so = {nz(q[L]) for _, q in g[g.category == 'single'].iterrows() for L in opts(q)}
            fm = [float(np.mean([nz(a) in mo for a in comb['sets'][i]] or [0.0]))
                  for i in range(len(comb['O']))]
            fs = [float(np.mean([nz(a) in so for a in comb['sets'][i]] or [0.0]))
                  for i in range(len(comb['O']))]
            sc = [sc[i] + wom * fm[i] + wos * fs[i] for i in range(len(comb['O']))]
            rt = [wom * fm[i] + wos * fs[i] for i in range(len(comb['O']))]
        if mot_p and W.get('w_mot_comb', 0):
            sc = [sc[i] + W['w_mot_comb'] * float(np.mean(
                      [mot_p.get(a, 0.0) for a in comb['sets'][i]] or [0.0]))
                  for i in range(len(comb['O']))]
        if zs_p and W.get('w_zs_comb', 0):
            sc = [sc[i] + W['w_zs_comb'] * float(np.mean(
                      [zs_p.get(a, 0.0) for a in comb['sets'][i]] or [0.0]))
                  for i in range(len(comb['O']))]
        cq = g[g.category == 'combination']
        # Optional per-stratum weight: on clips WITH a sequence question the overlap leak is
        # already near-certain, and in-sample the VLM helped the other stratum but hurt this
        # one. Unset keys fall back to the single w_vlm_comb.
        wvc = W.get('w_vlm_comb_seq' if K else 'w_vlm_comb_noseq', W.get('w_vlm_comb', 0.0))
        if len(cq) and wvc:
            e = V.get(cq.iloc[0].qa_id, {})
            c = _centred(e.get('letter', {}), comb['O'])
            sc = [sc[i] + wvc * c[comb['O'][i]] for i in range(len(comb['O']))]
        # 09-13: time-neighbour option texts (src/neighbours.py); w_nb_comb unset = 0
        nbp = ev.get('nb_p') or {}
        if nbp and W.get('w_nb_comb', 0):
            nzs = lambda x: ' '.join(str(x).strip().lower().split())
            sc = [sc[i] + W['w_nb_comb'] * float(np.mean(
                      [nbp.get(nzs(a), 0.0) for a in comb['sets'][i]] or [0.0]))
                  for i in range(len(comb['O']))]
        # w_ovl_bel=0: the routing picks the combination answer but stays out of the action
        # beliefs, where it made every multi option look present (multi noseq 0.806 -> 0.792)
        scb = sc if W.get('w_ovl_bel', 1.0) else [sc[i] - rt[i] for i in range(len(sc))]
        pr = sm([W['w_sharp'] * x for x in scb])
        for i, S in enumerate(comb['sets']):
            for a in S:
                bel[a] = bel.get(a, 0.0) + pr[i]
        cbest = comb['O'][int(np.argmax(sc))]
    for a in K:
        bel[a] = 1.0
    # Recurrence: in how many combination options an action appears. Distractor combinations
    # are built from the clip's other real actions, so a multi option found in two of them is
    # correct 82-95% of the time vs 9% for one found in none (src/struct_validate.py).
    rec = collections.Counter(a for S in comb['sets'] for a in S) if comb else collections.Counter()
    sgl = {str(q[L]).strip() for _, q in g[g.category == 'single'].iterrows() for L in opts(q)}

    # VLM presence evidence lifts/sinks individual actions
    vpres = {}
    for _, r in g[g.category == 'multi'].iterrows():
        e = V.get(r.qa_id, {})
        for L, s in e.get('present', {}).items():
            if L in opts(r):
                vpres[str(r[L]).strip()] = s

    def B(txt):
        p = min(max(bel.get(txt, 0.0), 1e-4), 0.9999)
        v = W.get('w_vlm_pres', 0.0) * vpres.get(txt, 0.0)
        m = W.get('w_mot_bel', 0.0) * mot_p.get(txt, 0.0)
        z = W.get('w_zs_bel', 0.0) * zs_p.get(txt, 0.0)
        wnb = W.get('w_nb_bel', 0.0)
        if wnb and ev.get('nb_p'):
            z += wnb * ev['nb_p'].get(' '.join(str(txt).strip().lower().split()), 0.0)
        return lo(p) + v + m + z

    out = {}
    s_txt = None     # this clip's predicted single|HAU answer, routed into multi (w_sans_m)
    # multi goes last so the single answer is known when multi is decided; no other branch
    # reads `out`, so the order changes nothing else
    for _, r in sorted(g.iterrows(), key=lambda x: x[1]['category'] == 'multi'):
        qt, O = r['qt'], opts(r)
        e = V.get(r.qa_id, {})
        if qt == 'combination|HAU':
            out[r.qa_id] = cbest
        elif qt == 'sequence|HAU':
            tab = ev['seq'].get(r.qa_id)
            # w_prec: pairwise-precedence prior between action texts (SeqModel.prec_scorer);
            # it and the slot model's log-likelihood share one permutation score.
            ws, wp = W.get('w_seq', 1.0), W.get('w_prec', 0.0)
            ptab = ev.get('prec', {}).get(r.qa_id) if wp else None
            ps = ((lambda p: (ws * tab[''.join(p)] if tab else 0.0)
                             + (wp * ptab[''.join(p)] if ptab else 0.0))
                  if (tab or ptab) else None)
            if e.get('pair') or e.get('perm') or ps:
                out[r.qa_id] = best_perm(O, e.get('pair', {}), e.get('perm'),
                                         W.get('w_gen', 0.0), ps, 1.0,
                                         W.get('w_vlm_pair', 0.0))
            else:
                out[r.qa_id] = M.perm.most_common(1)[0][0] if M.perm else 'ABCD'
        elif qt in ('single|HAU', 'single|HARn'):
            c = _centred(e.get('letter', {}), O)
            harn = qt == 'single|HARn'
            wv = W.get('w_vlm_single_harn' if harn else 'w_vlm_single', 0.0)
            mc = _centred(mot_h, [str(r[L]).strip() for L in O]) if (harn and mot_h) else None
            wm = W.get('w_mot_harn', 0.0) if harn else 0.0
            wz = W.get('w_zs_harn', 0.0) if harn else 0.0
            wr = 0.0 if harn else W.get('w_rec_s', 0.0)
            out[r.qa_id] = max(O, key=lambda L: lo(M.p(qt, r[L]))
                               + W['w_bel'] * B(str(r[L]).strip()) + wv * c[L]
                               + wr * rec[str(r[L]).strip()]
                               + (wm * mc[str(r[L]).strip()] if mc else 0.0)
                               + wz * zs_h.get(str(r[L]).strip(), 0.0))
            if not harn:
                s_txt = str(r[out[r.qa_id]]).strip()
        elif qt == 'multi|HAU':
            wr, wg = W.get('w_rec_m', 0.0), W.get('w_sgl_m', 0.0)
            sel = [L for L in O if lo(M.p(qt, r[L])) + W['w_bel_m'] * B(str(r[L]).strip())
                   + wr * rec[str(r[L]).strip()] + wg * (str(r[L]).strip() in sgl)
                   + W.get('w_sans_m', 0.0) * (str(r[L]).strip() == s_txt)
                   > W['thr_m']]
            if not sel:
                sel = [max(O, key=lambda L: lo(M.p(qt, r[L])))]
            out[r.qa_id] = ''.join(sorted(sel))
        else:  # emotion, object_interaction
            c = _centred(e.get('letter', {}), O)
            emo = qt == 'emotion|HAU'
            wv = W.get('w_vlm_emotion' if emo else 'w_vlm_oi', 0.0)
            src = mot_e if emo else mot_o
            wm = W.get('w_mot_emotion' if emo else 'w_mot_oi', 0.0)
            keys = [str(r[L]).strip() for L in O]
            mc = _centred({k: src[k] for k in keys if k in src}, keys) if src else None
            out[r.qa_id] = max(O, key=lambda L: lo(M.p(qt, r[L])) + wv * c[L]
                               + (wm * mc[str(r[L]).strip()] if mc else 0.0))
    return out


DEFAULT_W = dict(w_pmi=3.0, w_ovl=25.0, w_sharp=0.5, w_bel=5.0, w_bel_m=1.0, thr_m=-1.5,
                 w_vlm_single=0.0, w_vlm_single_harn=0.0, w_vlm_emotion=0.0, w_vlm_oi=0.0,
                 w_vlm_comb=0.0, w_vlm_pres=0.0, w_gen=0.0,
                 w_mot_harn=0.0, w_mot_emotion=0.0, w_mot_oi=0.0,
                 w_mot_comb=0.0, w_mot_bel=0.0, w_seq=1.0,
                 w_zs_harn=0.0, w_zs_comb=0.0, w_zs_bel=0.0, w_vlm_pair=0.0,
                 w_prec=0.0, w_rec_m=0.0, w_sgl_m=0.0, w_rec_s=0.0,
                 w_sans_m=0.0, w_ovl_m=0.0, w_ovl_s=0.0,
                 w_ovl_seq_scale=1.0, w_ovl_bel=1.0)
