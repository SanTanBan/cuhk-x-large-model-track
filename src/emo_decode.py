"""Emotion session decoders (09-13), shared by src/build_session.py; same code as
src/emo_fast_validate2.py, where they were validated.

Rows need: prior, vlm, imu (np arrays over the options), adv (normalised option texts), O (letters).
  emissions  log-softmax of (prior + wv*vlm + w*imu + beta*neighbour-option counts) / tau, where the
             count is how many clips at lags +-1, +-2 in the chain offer that option
  viterbi1   hard "consecutive clips never share the adverb"
  viterbi2   viterbi1 + gamma2 * [same adverb at lag 2]  (second order: state = last two choices)
  viterbiL   viterbi1 + lagw[L] * [same adverb at lag L] for every lag L >= 2 in lagw (09-14, validated
             in src/emo_fast_validate3.py). State = adverbs of the last max(lagw) choices, the best `beam`
             kept per step; exact while the states fit in the beam, and viterbiL(.., {2: g}) == viterbi2(.., g).
"""
import heapq
import numpy as np


def emissions(chain, w, tau, beta, wv):
    em = []
    for i, r in enumerate(chain):
        s = r.prior + wv * r.vlm + w * r.imu
        if beta:
            nb = [set(chain[j].adv) for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(chain)]
            s = s + beta * np.array([sum(a in o for o in nb) for a in r.adv])
        z = s / tau
        z = z - z.max()
        em.append(z - np.log(np.exp(z).sum()))
    return em


def viterbi1(chain, em):
    best, back = [em[0]], []
    for i in range(1, len(chain)):
        b, bk = [], []
        for j, adv in enumerate(chain[i].adv):
            cands = [(best[-1][k], k) for k, pa in enumerate(chain[i - 1].adv) if pa != adv]
            v, k = max(cands) if cands else (-1e18, 0)
            b.append(v + em[i][j]); bk.append(k)
        best.append(np.array(b)); back.append(bk)
    k = int(np.argmax(best[-1])); path = [k]
    for bk in reversed(back):
        k = bk[k]; path.append(k)
    return list(reversed(path))


def viterbi2(chain, em, gamma2):
    n = len(chain)
    if n < 3:
        return viterbi1(chain, em)
    V = {}
    for a, pa in enumerate(chain[0].adv):
        for b, pb in enumerate(chain[1].adv):
            if pa != pb:
                V[(a, b)] = (em[0][a] + em[1][b], None)
    hist = [V]
    for i in range(2, n):
        NV = {}
        for (a, b), (v, _) in hist[-1].items():
            for c, pc in enumerate(chain[i].adv):
                if chain[i - 1].adv[b] == pc:
                    continue
                s = v + em[i][c] + (gamma2 if chain[i - 2].adv[a] == pc else 0.0)
                if (b, c) not in NV or s > NV[(b, c)][0]:
                    NV[(b, c)] = (s, a)
        if not NV:
            return viterbi1(chain, em)
        hist.append(NV)
    (b, c), _ = max(hist[-1].items(), key=lambda kv: kv[1][0])
    path = [c, b]
    for i in range(len(hist) - 1, 0, -1):
        a = hist[i][(b, c)][1]
        path.append(a); b, c = a, b
    return list(reversed(path))


def viterbiL(chain, em, lagw, beam=512):
    n = len(chain)
    L = max(lagw) if lagw else 1
    V = {}
    for a, pa in enumerate(chain[0].adv):
        if (pa,) not in V or em[0][a] > V[(pa,)][0]:
            V[(pa,)] = (em[0][a], None, a)
    hist = [V]
    for i in range(1, n):
        NV = {}
        for key, (v, _, _) in hist[-1].items():
            for c, pc in enumerate(chain[i].adv):
                if key[0] == pc:
                    continue
                s = v + em[i][c]
                for lag, g in lagw.items():
                    if lag <= len(key) and key[lag - 1] == pc:
                        s += g
                nk = ((pc,) + key)[:L]
                if nk not in NV or s > NV[nk][0]:
                    NV[nk] = (s, key, c)
        if not NV:
            return viterbi1(chain, em)
        if len(NV) > beam:
            NV = dict(heapq.nlargest(beam, NV.items(), key=lambda kv: kv[1][0]))
        hist.append(NV)
    key = max(hist[-1], key=lambda k: hist[-1][k][0])
    path = []
    for i in range(n - 1, -1, -1):
        _, prev, c = hist[i][key]
        path.append(c)
        key = prev
    return list(reversed(path))
