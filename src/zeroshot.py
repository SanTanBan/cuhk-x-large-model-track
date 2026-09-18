"""CLIP zero-shot evidence, packaged for the solver.

Zero-shot similarity uses no training labels, so its errors are uncorrelated with the
supervised classifiers' -- which is why it helps in an ensemble despite being individually
weaker (HARn: 0.564 alone, 0.690 supervised alone, 0.709 combined).

Similarities are z-scored per clip across the label vocabulary, so the fusion weights come
out O(1) instead of O(50) (raw cosines live in a narrow band around 0.25).

Text embeddings are cached to features/zs_text.npz -- encoding is fast but not free.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = 'features/zs_text.npz'
TEMPLATES = ["a photo of a person {}", "a person {} at home", "{}"]


def _encode(phrases):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
    p = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    out = {}
    with torch.no_grad():
        for ph in phrases:
            inp = p(text=[t.format(ph) for t in TEMPLATES], return_tensors='pt', padding=True)
            e = m.get_text_features(**inp)
            if not torch.is_tensor(e):
                e = e.pooler_output if hasattr(e, 'pooler_output') else e[0]
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            v = e.mean(0)
            out[ph] = (v / v.norm().clamp_min(1e-8)).numpy().astype(np.float32)
    return out


def text_table(phrases):
    phrases = sorted(set(phrases))
    have = {}
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        have = {str(k): v for k, v in zip(z['keys'], z['vecs'])}
    missing = [p for p in phrases if p not in have]
    if missing:
        have.update(_encode(missing))
        os.makedirs('features', exist_ok=True)
        ks = sorted(have)
        np.savez_compressed(CACHE, keys=np.array(ks),
                            vecs=np.stack([have[k] for k in ks]))
    return {p: have[p] for p in phrases}


class ZeroShot:
    def __init__(self, clip_emb, harn_phrases, hau_actions):
        self.C = clip_emb
        self.T = text_table(list(harn_phrases) + [a.lower() for a in hau_actions])
        self.hau = {a: self.T[a.lower()] for a in hau_actions if a.lower() in self.T}
        self.harn = {p: self.T[p] for p in harn_phrases if p in self.T}

    def _sims(self, path, table):
        E = self.C.get(path)
        if E is None or not table:
            return {}
        keys = list(table)
        S = np.stack([table[k] for k in keys])          # k x 512
        v = (E @ S.T).mean(0)                            # mean over frames
        mu, sd = float(v.mean()), float(v.std()) or 1.0
        return {k: float((v[i] - mu) / sd) for i, k in enumerate(keys)}

    def harn_sims(self, path):
        return self._sims(path, self.harn)

    def presence(self, path):
        return self._sims(path, self.hau)


def build(tr, te, clip_emb):
    from solver import opts
    harn = {str(r[L]).strip() for df in (tr, te)
            for _, r in df[df.qt == 'single|HARn'].iterrows() for L in opts(r)}
    hau = {a.strip() for _, r in tr[tr.category == 'combination'].iterrows()
           for L in opts(r) for a in str(r[L]).split(',')}
    hau |= {a.strip() for _, r in te[te.category == 'combination'].iterrows()
            for L in opts(r) for a in str(r[L]).split(',')}
    return ZeroShot(clip_emb, sorted(harn), sorted(hau))
