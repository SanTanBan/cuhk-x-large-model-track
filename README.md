# CUHK-X Challenge 2026 — Large Model Track (VQA)

Solution code, submissions and working notes for the [CUHK-X Multimodal Human Activity Challenge, Large
Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-large-model-track).
Kaggle team **InSociEUP** (user `SanTanBan`). Published for the organisers' reproduction and
verification stage.

| Submission | Public (342 questions) | Private (340 questions) |
|---|---|---|
| **`sub18_nbw-lag`** — final selection 1 | 0.83918 (287) | **0.79705** |
| **`sub17_nbw-emo13e`** — final selection 2 | 0.83625 (286) | 0.79411 |
| `sub17t_pres-notime` — variant using no recording times | 0.79532 (272) | — |

Everything here is our own code. No answers from other teams are used in the two selected submissions.

## How the system works

The answer to each question is a weighted sum of independent evidence sources, fused per clip and then
decoded under the constraints that hold within a clip and along a recording session.

1. **Option-text priors and within-clip structure** (`src/solver.py`, `src/joint.py`).
   P(option correct | option text) per question type, action co-occurrence, and the constraint that a
   clip's `sequence` options are exactly the actions that occur in that clip, which then constrains its
   `single` / `multi` / `combination` answers.
2. **Per-clip visual features** (`src/frames.py`, `src/motion.py`, `src/motion2.py`, `src/clip_feats.py`,
   `src/feats.py`). Frame-difference motion statistics from the Depth stream, plus CLIP ViT-B/32
   embeddings of sampled IR frames, PCA-reduced before fusion.
3. **VLM evidence** (`notebooks/kaggle_notebook/`). Qwen2.5-VL-7B-Instruct in fp16, sharded by `device_map="auto"`
   across Kaggle's two T4s (31 GB total), scores every option letter for each question. `load_model()` falls
   back to 4-bit nf4 automatically when total GPU memory is under 30 GB, so one 16 GB card also runs it. The resulting log-probabilities are in `vlm/`, so the
   rest of the pipeline runs without a GPU. Per-group fusion weights were validated on held-out subjects
   (`src/vlm_perweight.py`).
4. **Organisers' non-visual supplement** (`src/nonvisual_feats.py`, `src/motion_model.py`). A random
   forest on IMU features for `emotion`; a Skeleton-based classifier for HARn actions and object
   interaction, trained on all 2,925 HARn units.
5. **Recording-session structure** (`src/unit_times.py`, `src/neighbours.py`, `src/emo_decode.py`,
   `src/build_session.py`). See the disclosure below.

### Recording-session structure — what it uses, stated plainly

The Skeleton frames in the organisers' supplement are named `Color_<date>_<time>_<frame>.json`. Sorting
clips by that time and splitting wherever the gap reaches 30 minutes recovers the recording sessions (the
test set forms 16 unbroken ones). Three things use that ordering:

- **Per-session normalisation of IMU features** before the emotion classifier.
- **Session decoding of `emotion`**: consecutive clips never share a manner adverb (0 of 725 in training),
  and repeats by lag run against chance 0.043 — lag 2 0.011, lag 3 0.065, lag 4 0.007, lag 6 0.096. A
  Viterbi decoder applies the hard lag-1 rule plus weights at lags 2–6.
- **Time-neighbour option pooling** for `combination`, `single|HAU` and `multi`: the option *texts* of
  neighbouring clips feed the scores, because consecutive clips share action answers far above chance.

**No labels and no test answers are used anywhere** — only option text, sensor data and file timestamps
from the organisers' own release. Both competition rules ("No using test-set answers in training; no
manual labeling of test questions") are respected. The gain is large: our best build without any of this
(`sub17t_pres-notime`) scores 0.79532 public against 0.83918 with it. Because it leans on how the data was
recorded rather than on the sensors alone, `sub17t_pres-notime` is included so the effect can be measured
directly. Choosing this evidence was a deliberate, disclosed decision — see `docs/NOTES_2026-09-13_14.md`.

## Validation protocol

Nothing shipped on a public-leaderboard hunch. Every change was validated two ways before submission:

- **6-fold cross-subject CV**, weighted by the test category mix.
- **Half-split held-out validation**: tune on one half of the subjects, score on the other, and accept only
  if the change is positive on **both** halves. Rejected ideas are listed in
  `docs/TECHNICAL_RECORD.md` under the dead ends.
- **Leak guards**: every build diffs against its parent and refuses if answers changed outside the
  categories the change was supposed to touch. Rebuilds of earlier submissions are byte-identical.

## Reproducing

### Environment

CPU pipeline: Windows 11, Python 3.12.7, 4 logical cores (Intel i3-10110U), 8 GB RAM. See
`requirements.txt` for pinned versions. The VLM stage ran on Kaggle with two NVIDIA T4s (31 GB total) and
loaded the model in fp16; every run logged `loading fp16 across 2 GPUs (31 GB)`. A single 24 GB card runs the
same fp16 path, and a single 16 GB card runs the automatic 4-bit fallback. The GPU-side pins are in
`requirements-gpu.txt`: torch 2.10.0+cu128 (CUDA 12.8) and transformers >= 4.49. No FlashAttention, xformers,
DeepSpeed, vLLM, Triton kernels or Apex, and nothing is compiled at install time.

### Steps

```bash
# 1. Data (not in this repo — see below): place the competition release and the non-visual
#    supplement under data/, keeping the organisers' names, including
#    data/LMT_IMU_Radar_Skeleton.zip

# 2. Non-visual features and the recording-session timeline
python src/nonvisual_feats.py          # -> features/imu_feats.csv, features/skel_feats.csv
python src/unit_times.py               # -> features/unit_times.csv  (verify: --check=<existing csv>)

# 3. Visual features: sample frames, then motion and CLIP descriptors
python src/extract_frames.py           # -> frames/ (also the Kaggle dataset for the VLM notebook)
#    then the per-clip descriptors in src/motion.py, src/motion2.py, src/clip_feats.py, src/feats.py

# 4. VLM evidence: run notebooks/kaggle_notebook/vlm_infer.ipynb on a GPU (we used Kaggle's 2x T4).
#    Skip this to reuse the scores committed in vlm/.

# 5. Final submission = selection 1 (two steps)
python src/build_stack.py nbwide submissions/sub17_nbw.csv
python src/build_session.py submissions/sub18_nbw-lag.csv --emo=norm+sess --w-emo=5 --tau=0.5 \
       --nbopt=0.25 --lagw=src/emo_fast3_report.json:order2+LR36 \
       --base-sub=submissions/sub17_nbw_base.csv
```

Step 5 rebuilds `sub18_nbw-lag.csv` (md5 `86e216768efc1198bdd11857e916808d`). **Verified on 2026-09-18** from
the VLM scores committed in `vlm/`: step 1 takes 71 s and reproduces `sub17_nbw_base.csv`
(md5 `d2918a3b844baa5c64c3f158eb4cb2ce`), step 2 takes 42 s and reproduces `sub18_nbw-lag.csv` — 113 s in total
on 4 CPU cores, with no GPU. `src/build_stack.py` refuses
to build unless the validation report it reads was accepted on both subject halves, and both steps run
their own leak guards. Selection 2, `sub17_nbw-emo13e.csv`, is the same action model with the first-order
emotion decoder (`--emo=norm+sess --w-emo=12 --tau=0.25`).

Switches (`CUHKX_EMO`, `CUHKX_HARN`, `CUHKX_NB`, `CUHKX_PRES`, …) are all off by default, and with them
unset the pipeline reproduces the earlier submission byte for byte. The build environment for the final
model is in `src/best_nb_wide_W_config.json`.

## Layout

| Path | What it holds |
|---|---|
| `src/` | The whole CPU pipeline: features, models, fusion, decoders, validators, builders |
| `notebooks/` | Kaggle notebooks for the Qwen2.5-VL scoring runs, with their kernel metadata |
| `vlm/` | VLM option log-probabilities for train and test clips, so no GPU is needed |
| `submissions/` | Every submission we made, including both final selections |
| `docs/TECHNICAL_RECORD.md` | Full technical record: what worked, what was rejected, and why |
| `docs/STRATEGY_PLAN.md`, `docs/DAILY_LOG.md`, `docs/NOTES_2026-09-13_14.md` | Day-by-day working notes |

## Not included

Under the CUHK-X License v2.0 the dataset is the organisers' to distribute, so nothing derived from their
media is republished here:

- `data/` — the competition release and the IMU/Radar/Skeleton supplement
  ([Hugging Face](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Large_Model_Track), Google Drive or
  Baidu Wangpan, per the competition's Data tab)
- `frames/`, `kaggle_dataset/` — IR frames sampled from the competition videos (rebuild with
  `src/extract_frames.py`)
- `features/` — the derived feature tables (rebuild with the scripts in step 2 and 3; about 75 MB)
- Another team's publicly released answer file, which earlier blend experiments referenced. It is not part
  of either selected submission.
