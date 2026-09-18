"""CLIP zero-shot: match clip embeddings against text embeddings of the option strings.

The supervised HARn classifier only knows actions it saw enough of; zero-shot similarity
needs no training data at all, so it may help where supervision is thin. Whether it
survives the IR domain shift (CLIP was trained on natural RGB photos) is an empirical
question -- hence this probe.
"""
import numpy as np, torch, sys, collections
sys.path.insert(0, 'src')
from solver import load, opts
from feats import load_clip_emb

TEMPLATES = ["a photo of a person {}", "a person {} at home", "{}"]


def text_embeddings(phrases):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
    p = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    out = {}
    with torch.no_grad():
        for ph in phrases:
            prompts = [t.format(ph) for t in TEMPLATES]
            inp = p(text=prompts, return_tensors='pt', padding=True)
            e = m.get_text_features(**inp)
            if not torch.is_tensor(e):
                e = e.pooler_output if hasattr(e, 'pooler_output') else e[0]
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            v = e.mean(0)
            out[ph] = (v / v.norm().clamp_min(1e-8)).numpy()
    return out


if __name__ == '__main__':
    tr, te = load()
    C = load_clip_emb()
    phrases = sorted({str(r[L]).strip()
                      for df in (tr, te)
                      for _, r in df[df.qt == 'single|HARn'].iterrows()
                      for L in opts(r)})
    print('encoding %d option phrases' % len(phrases), flush=True)
    T = text_embeddings(phrases)

    users = sorted(tr.user.unique())
    folds = [users[i::6] for i in range(6)]
    for pool in ('mean', 'max'):
        ok = n = 0
        for _, r in tr[tr.qt == 'single|HARn'].iterrows():
            if r.path not in C:
                continue
            E = C[r.path]                      # frames x 512
            O = opts(r)
            sims = []
            for L in O:
                s = E @ T[str(r[L]).strip()]
                sims.append(float(s.mean() if pool == 'mean' else s.max()))
            n += 1
            ok += (O[int(np.argmax(sims))] == str(r.answer))
        print('  zero-shot (%s-pool over frames): single|HARn=%.4f  (n=%d, chance 0.333)'
              % (pool, ok / max(1, n), n))
