"""CLIP image embeddings for the sampled IR frames (CPU).

Frame-difference statistics say how a person moves; they say nothing about *what* is in
the scene. CLIP embeddings add semantics -- bathroom mirror, keyboard, broom -- which is
exactly what `single`, `combination`, `multi` and `object_interaction` turn on.

Per-frame embeddings are stored (not pooled) so the same file serves clip-level models
and the per-segment temporal-order model.

Output: features/clip_<split>.npz  with `keys` (clip key), `paths`, and `emb` (n,8,512).
"""
import numpy as np, pandas as pd, torch, os, sys, glob, time, argparse
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = 'openai/clip-vit-base-patch32'
# HAU clips are ~12 s and multi-action, so 8 frames can miss whole actions; HARn clips
# are ~2 s and single-action, where 8 is plenty.
NFRAMES = 8
NFRAMES_HAU = 16
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(MODEL_ID).eval()
    p = CLIPProcessor.from_pretrained(MODEL_ID)
    return m, p


def embed(m, p, images):
    with torch.no_grad():
        inp = p(images=images, return_tensors='pt')
        o = m.get_image_features(**inp)
        if torch.is_tensor(o):
            e = o
        elif hasattr(o, 'pooler_output') and o.pooler_output is not None:
            e = o.pooler_output
        else:
            e = o[0]
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return e.cpu().numpy().astype(np.float32)


def pick(files, n):
    if len(files) <= n:
        return files + [files[-1]] * (n - len(files))
    return [files[int(round(i))] for i in np.linspace(0, len(files) - 1, n)]


def run(split, out):
    man = pd.read_csv('frames/%s/_manifest.csv' % split)
    m, p = load_clip()
    keys, paths, embs = [], [], []
    t0 = time.time()
    for i, r in enumerate(man.itertuples(index=False)):
        fs = sorted(glob.glob(os.path.join('frames', split, r.key, '*.jpg')))
        if not fs:
            continue
        n = NFRAMES_HAU if r.source == 'HAU' else NFRAMES
        ims = [Image.open(f).convert('RGB') for f in pick(fs, n)]
        embs.append(embed(m, p, ims))
        keys.append(r.key); paths.append(r.path)
        for im in ims:
            im.close()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print('  %s %d/%d  %.1f min  eta %.0f min'
                  % (split, i + 1, len(man), el / 60, el / (i + 1) * (len(man) - i - 1) / 60),
                  flush=True)
    # ragged (HAU 16, HARn 8) -> 1-D object array. np.array(list_of_arrays, dtype=object)
    # would build a 3-D array whenever the lengths happen to match, so allocate explicitly.
    arr = np.empty(len(embs), dtype=object)
    for i, e in enumerate(embs):
        arr[i] = e
    np.savez_compressed(out, keys=np.array(keys), paths=np.array(paths), emb=arr)
    print('wrote %s: %d clips (%d HAU-length)' % (out, len(keys),
          sum(1 for e in embs if len(e) == NFRAMES_HAU)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['train', 'test', 'both'])
    a = ap.parse_args()
    os.makedirs('features', exist_ok=True)
    for s in (['test', 'train'] if a.which == 'both' else [a.which]):
        run(s, 'features/clip_%s.npz' % s)
