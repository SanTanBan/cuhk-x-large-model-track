# %% [markdown]
# # CUHK-X Large Model Track - VLM evidence extraction (Qwen2.5-VL)
#
# Reads pre-sampled IR frames and emits **calibrated per-option scores** for every question.
# Nothing is decided here: the scores are fused offline with the structural /
# co-occurrence / motion / CLIP model, whose weights are tuned cross-subject.
#
# Outputs `vlm_scores_test.csv` and `vlm_scores_train.csv`, long format
# `(qa_id, key, kind, score)`:
#
# | kind | key | score |
# |---|---|---|
# | `letter`  | option letter | log P(that letter) |
# | `present` | option letter | logit(Yes) - logit(No) for "does this action occur" |
# | `perm`    | 4-letter order | 1.0 (the generated chronological order) |
# | `pair`    | `"A<B"` | > 0 means A is judged to happen before B |
#
# **Test is scored first** so that a session timeout still leaves the scores that matter.

# %%
SMOKE = False           # True -> a handful of clips, to validate the pipeline cheaply
SPLITS = ["test", "train"]   # which splits this kernel scores; set per kernel by the builder
TRAIN_SHARD = None           # (k, n): shard k of n of the clips OUTSIDE the tuning sample
TRAIN_ALL = False            # True: ALL training clips (TRAIN_SHARD then shards them)
TTA = False                  # re-score letter questions with the options cyclically shifted
TTA_CATS = ["single", "combination", "emotion", "object_interaction"]
TTA_SOURCES = ["HAU", "HARn"]

import os, sys, json, math, gc, time, glob, zipfile, subprocess
from importlib.metadata import version as _v

def _ver(s):
    return tuple(int(x) for x in s.split('+')[0].split('.')[:3] if x.isdigit())

# Qwen2.5-VL needs transformers >= 4.49. Upgrade BEFORE transformers is first imported.
try:
    tv = _v("transformers")
except Exception:
    tv = "0.0.0"
print("transformers installed:", tv)
if _ver(tv) < (4, 49, 0):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers>=4.49,<5", "accelerate"], check=False)
    tv = _v("transformers")
    print("transformers upgraded to:", tv)

import numpy as np, pandas as pd, torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_FRAMES_HAU, MAX_FRAMES_HARN = 12, 8
TRAIN_HAU_CLIPS, TRAIN_HARN_CLIPS = 200, 120
MAX_PIXELS = 256 * 28 * 28          # ~500x390 per frame: enough for IR action recognition
OUT = "/kaggle/working"
T_START = time.time()

# ---- locate inputs robustly: Kaggle's mount layout has changed over time ----
def find_file(name):
    for root, _, files in os.walk("/kaggle/input"):
        if name in files:
            return os.path.join(root, name)
    return None

TEST_QA = find_file("test_qa.csv")
TRAIN_QA = find_file("training_qa.csv")
print("test_qa:", TEST_QA, "| training_qa:", TRAIN_QA)

def resolve_frames():
    # extracted dataset: a directory holding test/_manifest.csv
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.basename(root) == "test" and "_manifest.csv" in files:
            return os.path.dirname(root)
    z = find_file("ir_frames.zip")
    if z is None:
        raise FileNotFoundError("frames dataset not found under /kaggle/input")
    dest = "/kaggle/temp/frames"     # NOT /kaggle/working - keeps it out of the output
    if not os.path.isdir(os.path.join(dest, "test")):
        os.makedirs(dest, exist_ok=True)
        t = time.time()
        zipfile.ZipFile(z).extractall(dest)
        print("unzipped frames in %.0f s" % (time.time() - t))
    return dest

FRAMES = resolve_frames()
print("frames root:", FRAMES, sorted(os.listdir(FRAMES))[:5])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def load_model():
    ngpu = torch.cuda.device_count()
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(ngpu)) / 1e9
    kw = dict(device_map="auto")
    dkey = "dtype" if _ver(tv) >= (4, 56, 0) else "torch_dtype"
    if total >= 30:
        kw[dkey] = torch.float16
        print("loading fp16 across %d GPUs (%.0f GB)" % (ngpu, total))
    else:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                           check=False)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        print("loading 4-bit on %.0f GB" % total)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kw).eval()
    p = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=128 * 28 * 28, max_pixels=MAX_PIXELS)
    return m, p

t = time.time()
model, proc = load_model()
print("model loaded in %.0f s" % (time.time() - t))
tok = proc.tokenizer

def one_id(s):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) >= 1 else None

LETTER_ID = {L: one_id(L) for L in "ABCD"}
YES_ID, NO_ID = one_id("Yes"), one_id("No")
print("letter ids", LETTER_ID, "yes/no", YES_ID, NO_ID)
assert len(set(LETTER_ID.values())) == 4 and YES_ID != NO_ID

# %%
RNG = np.random.RandomState(0)
CATS = ["single", "multi", "combination", "sequence", "emotion", "object_interaction"]

def build_split(split):
    qa = pd.read_csv(TEST_QA if split == "test" else TRAIN_QA, encoding="utf-8-sig")
    man = pd.read_csv(os.path.join(FRAMES, split, "_manifest.csv"))
    KEY = dict(zip(man.path, man.key))
    qa = qa[qa.path.isin(KEY)].copy()
    if SMOKE:
        # the fewest clips that still exercise every question type
        keep, seen = [], set()
        for p, g in qa.groupby("path"):
            new = set(g.category) - seen
            if new:
                keep.append(p); seen |= new
            if seen >= set(CATS) or len(keep) >= 6:
                break
        qa = qa[qa.path.isin(set(keep))]
    elif split == "train":
        # The tuning sample (200 HAU + 120 HARn) is recomputed with a FRESH RandomState(0)
        # so every kernel agrees on exactly which clips it holds; TRAIN_SHARD then picks a
        # round-robin share of the clips OUTSIDE it.
        rng = np.random.RandomState(0)
        sample = []
        for src, n in (("HAU", TRAIN_HAU_CLIPS), ("HARn", TRAIN_HARN_CLIPS)):
            cl = sorted(qa[qa.source == src].path.unique())
            sample += list(rng.permutation(cl)[:n]) if len(cl) > n else list(cl)
        if TRAIN_ALL:
            pool = sorted(qa.path.unique())
            keep = set(pool) if TRAIN_SHARD is None else set(pool[TRAIN_SHARD[0]::TRAIN_SHARD[1]])
        elif TRAIN_SHARD is None:
            keep = set(sample)
        else:
            k, n = TRAIN_SHARD
            rest = sorted(set(qa.path.unique()) - set(sample))
            keep = set(rest[k::n])
        qa = qa[qa.path.isin(keep)]
    print(split, "questions:", len(qa), "clips:", qa.path.nunique(),
          qa.category.value_counts().to_dict(), flush=True)
    return qa, KEY

def frames_for(split, KEY, path, source):
    d = os.path.join(FRAMES, split, KEY[path])
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
    cap = MAX_FRAMES_HARN if source == "HARn" else MAX_FRAMES_HAU
    if len(fs) > cap:
        fs = [fs[int(round(i))] for i in np.linspace(0, len(fs) - 1, cap)]
    return [Image.open(f).convert("RGB") for f in fs]

def opts(r):
    return [L for L in "ABCD" if isinstance(r[L], str) or not pd.isna(r[L])]

PREAMBLE = ("You are watching {n} frames sampled in order from a short infrared video of one "
            "person doing everyday activities at home. The frames are in chronological order.")

N_FWD = [0]

@torch.no_grad()
def _inputs(images, text):
    msgs = [{"role": "user",
             "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]}]
    chat = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[chat], images=images, return_tensors="pt")
    return {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}

@torch.no_grad()
def logprobs(images, text, ids):
    N_FWD[0] += 1
    out = model(**_inputs(images, text))
    lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    return {k: float(lp[i]) for k, i in ids.items() if i is not None}

@torch.no_grad()
def generate(images, text, max_new=12):
    N_FWD[0] += 1
    inp = _inputs(images, text)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

# %%
def score_split(split):
    qa, KEY = build_split(split)
    rows, t0, f0 = [], time.time(), N_FWD[0]
    nclip = qa.path.nunique()
    dest = os.path.join(OUT, "vlm_scores_%s.csv" % split)

    for ci, (path, g) in enumerate(qa.groupby("path")):
        src = g.iloc[0].source
        if TTA and not (g.category.isin(TTA_CATS) & g.source.isin(TTA_SOURCES)).any():
            continue
        imgs = frames_for(split, KEY, path, src)
        if not imgs:
            continue
        pre = PREAMBLE.format(n=len(imgs))

        for _, r in g.iterrows():
            O = opts(r)
            cat = r.category
            if TTA and not (cat in TTA_CATS and r.source in TTA_SOURCES):
                continue
            block = "\n".join("%s. %s" % (L, r[L]) for L in O)
            try:
                if cat in ("single", "combination", "emotion", "object_interaction"):
                    # Shift 0 is the normal run. TTA adds shifts 1..m-1, in which the option
                    # shown as letter O[j] is the ORIGINAL option O[(j+s) % m]. Rows are keyed
                    # by the original letter, so averaging offline cancels letter-position bias.
                    for s in (range(1, len(O)) if TTA else [0]):
                        shown = [O[(j + s) % len(O)] for j in range(len(O))]
                        blk = "\n".join("%s. %s" % (O[j], r[shown[j]]) for j in range(len(O)))
                        q = ("%s\n\nQuestion: %s\n%s\n\nReply with the single letter of the best "
                             "option and nothing else." % (pre, r.question, blk))
                        lp = logprobs(imgs, q, {L: LETTER_ID[L] for L in O})
                        for j, L in enumerate(O):
                            rows.append((r.qa_id, shown[j],
                                         "letter" if s == 0 else "letter_s%d" % s, lp[L]))

                elif cat == "multi":
                    for L in O:
                        q = ("%s\n\nDoes the person do the following at any point in this video: "
                             "\"%s\"?\nReply with exactly Yes or No." % (pre, r[L]))
                        lp = logprobs(imgs, q, {"Yes": YES_ID, "No": NO_ID})
                        rows.append((r.qa_id, L, "present", lp["Yes"] - lp["No"]))

                elif cat == "sequence":
                    q = ("%s\n\nAll four actions below occur in this video. Put them in the "
                         "order they happen, earliest first.\n%s\n\nReply with exactly four "
                         "letters, e.g. CADB. No other text." % (pre, block))
                    txt = "".join(c for c in generate(imgs, q).upper() if c in "ABCD")
                    seen, order = set(), []
                    for c in txt:
                        if c not in seen:
                            seen.add(c); order.append(c)
                    for L in O:
                        if L not in seen:
                            order.append(L)
                    rows.append((r.qa_id, "".join(order[:4]), "perm", 1.0))
                    for i in range(len(O)):
                        for j in range(i + 1, len(O)):
                            a, b = O[i], O[j]
                            q2 = ("%s\n\nWhich happens FIRST in the video?\nA. %s\nB. %s\n"
                                  "Reply with A or B only." % (pre, r[a], r[b]))
                            lp = logprobs(imgs, q2, {"A": LETTER_ID["A"], "B": LETTER_ID["B"]})
                            rows.append((r.qa_id, "%s<%s" % (a, b), "pair", lp["A"] - lp["B"]))
            except Exception as e:
                print("ERR", r.qa_id, type(e).__name__, str(e)[:200], flush=True)
                if SMOKE:
                    raise

        for im in imgs:
            im.close()

        if (ci + 1) % 10 == 0 or SMOKE:
            el = time.time() - t0
            nf = N_FWD[0] - f0
            print("[%s] %d/%d clips  %.1f min  %.2f s/fwd  eta %.0f min"
                  % (split, ci + 1, nclip, el / 60, el / max(1, nf),
                     el / (ci + 1) * (nclip - ci - 1) / 60), flush=True)
            pd.DataFrame(rows, columns=["qa_id", "key", "kind", "score"]).to_csv(dest, index=False)
            gc.collect(); torch.cuda.empty_cache()

    out = pd.DataFrame(rows, columns=["qa_id", "key", "kind", "score"])
    out.to_csv(dest, index=False)
    el = time.time() - t0
    info = dict(split=split, rows=len(out), clips=nclip, minutes=round(el / 60, 1),
                forward_passes=N_FWD[0] - f0,
                sec_per_fwd=round(el / max(1, N_FWD[0] - f0), 3),
                kinds=out.kind.value_counts().to_dict())
    print(json.dumps(info), flush=True)
    return out, info

# %%
INFO = {"transformers": tv, "torch": torch.__version__, "smoke": SMOKE,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
for _s in SPLITS:                          # test first when both: it is what matters
    _, INFO[_s] = score_split(_s)
INFO["total_minutes"] = round((time.time() - T_START) / 60, 1)
json.dump(INFO, open(os.path.join(OUT, "run_info.json"), "w"), indent=1)
print(json.dumps(INFO, indent=1))
if SMOKE:
    print(pd.read_csv(os.path.join(OUT, "vlm_scores_test.csv")).head(30).to_string())
print("\nDONE.")
