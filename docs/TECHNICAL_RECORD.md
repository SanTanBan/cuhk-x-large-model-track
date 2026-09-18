# CUHK-X Competition — Large Model Track

Multiple-choice VQA over short privacy-preserving videos (Depth / IR / Thermal, **no RGB**) of
one person doing everyday activities at home. 682 test questions, scored by plain accuracy.
Deadline **2026-09-15 15:55 UTC**. Private LB top-15 advances to the Selection Stage.

## Results so far

| Model | Cross-subject CV | Public LB |
|---|---|---|
| Public baselines (other teams' notebooks) | — | 0.178 – 0.216 |
| Best public notebook | — | 0.778 |
| Option-content priors only | 0.375 | — |
| \+ within-clip structural constraints | 0.461 | — |
| \+ action co-occurrence (PMI), joint clip inference | 0.636 | 0.608 |
| \+ CPU motion models (no GPU, no VLM) | 0.673 | 0.675 |
| \+ motion action-presence, all weights tuned | 0.708 | 0.678 |
| \+ CLIP embeddings, per-task feature selection | 0.713 | 0.696 |
| \+ CLIP zero-shot as independent evidence | 0.718 | 0.705 |
| \+ Qwen2.5-VL, HARn actions only (one weight) | +0.007 held-out | **0.711** |
| \+ Qwen2.5-VL on 4 validated groups, all 1333 training clips scored | +0.016 / +0.009 held-out | **0.722** |
| \+ pairwise-precedence prior for `sequence`, lower `multi` threshold (09-11) | +0.019 / +0.007 held-out | **0.746** |
| \+ single-answer routing (multi), then combination routing (09-11) | +0.008 / +0.010, +0.005 / +0.008 held-out | 0.766 |
| \+ organisers' IMU (emotion) + Skeleton (HARn) (09-11, sub12) | +0.005 / +0.013, +0.011 / +0.009 held-out | **0.792** |
| \+ recording-session emotion decoding (09-13, sub13e) | +0.020 / +0.022 emotion accuracy held-out | **0.816** |
| \+ second-order session decoder (09-13, sub14e) | +0.037 / +0.025 emotion accuracy over sub13e | 0.813 |
| \+ time-neighbour option pooling (09-13, sub15_nb) | +0.0045 / +0.0025 held-out | **0.822** |
| wide pooling grid (09-13, sub17_nbw) | +0.0097 / +0.0131 held-out | 0.833 |
| \+ higher-order emotion decoder (09-14, sub18_nbw-lag) | +0.0049 / +0.0050 emotion accuracy | **0.839** |
| timestamp-free fallback (09-14, sub17t_pres-notime = sub12 + IMU/Skeleton presence) | +0.0016 / +0.0017 held-out | 0.795 |

CV has tracked the LB to within ~0.017. One scare along the way: the motion-presence row
moved CV +0.035 but the LB only +0.003. That is within the public half's own noise
(±0.025 at n=342), but it is also the shape tuning overfit takes, so `src/cv_nested.py`
re-measures the tuning gain on subjects the tuner never saw — it came back **+0.0373**,
matching the plain CV. The tuning is sound; the flat LB move was noise. Re-run on the
final ~16-weight model it gives **+0.0560**: the gain grew as evidence sources were added,
which is what genuinely informative features look like (fitted noise would not replicate
out-of-subject).

CV runs 6-fold grouped by subject and is re-weighted to the test set's own
(question-type × has-sequence-question) mix.

**The public LB is exactly 342 of the 682 questions** — every score on the board is a
clean `x/342` (ours is 241/342). Private is the other 340 and is what decides standings.
Public leaders sit near 0.97, and the Top-15 cutoff is ~0.871. Note that a fully honest
system is bounded near ~0.80 by `emotion` alone (21% of questions, subjective manner
adverbs), so some of the very top public scores may not survive the private split.
No leaderboard probing is used here — it buys nothing on the private half.

## What actually drives the score

The benchmark is auto-generated, and the generator leaves four exploitable regularities.
Each was measured before being used — the ones that didn't hold up are listed at the bottom.

**1. The distractor pool and the answer pool are disjoint for `object_interaction`.**
"a smartphone" / "a cabinet" / "a glass" are correct 100% of the time they appear;
"a mug" / "a shelf" / "a towel" are never correct. Text alone gets 0.77 on that category.

**2. A clip's `sequence` question leaks four true actions.** All four of its options occur
in the video — only the order is being asked. So for the *other* questions about the same
clip they are known-present actions. When exactly one `single` option is among them it is
the answer **224/224 times**.

**3. Real action-sets co-occur; sampled distractors don't.** The `combination` options are
lists of actions, one of which is the video's true set. Scoring options by pairwise
co-occurrence lift (PMI) learned from training answer-sets lifts that category from
0.50 → 0.79 on clips with no sequence question.

**4. Solving `combination` solves everything else.** Given the true action set S,
`single` is answerable with **100%** accuracy on the 80% of clips where exactly one option
lies in S, and `multi` reaches 0.75. So the pipeline infers S first, then reads the rest off it.

Together these make `single|HAU` 0.91, `combination` 0.79–0.94 and `multi` 0.70–0.78
without looking at a single pixel.

## Motion models — the video, without a GPU

The four weak categories all describe *how* someone moves, which frame differencing
measures directly. No VLM, no GPU, ~20 min of CPU:

| Category | n (test) | text-only | \+ motion | \+ CLIP (final) |
|---|---:|---:|---:|---:|
| single \| HARn | 51 | 0.364 | 0.615 | **0.699** |
| emotion | 144 | 0.320 | **0.403** | 0.403 |
| sequence | 39 | 0.042 | **0.276** | 0.276 |
| object_interaction | 21 | 0.774 | 0.812 | **0.857** |

Final per-category CV, all evidence combined: `single|HAU` 0.930 / 0.955,
`combination` 0.882 / 0.958, `object_interaction` 0.857, `multi` 0.735 / 0.779,
`single|HARn` 0.699, `emotion` 0.403 / 0.386, `sequence` 0.276
(the two figures are clips without / with a `sequence` question).

What carries the signal: **periodicity** (autocorrelation of the motion signal separates
jumping jacks / squats / jogging from pouring or reading), **where** the motion is
(vertical centroid separates hair/teeth from floor/feet), **burstiness**, and **posture**
proxies. `sequence` works by splitting each clip into 4 time slots, learning
P(action | slot) — the training answer *is* the chronological order, so the labels are
free — then solving the 4×4 assignment. Pairwise order accuracy reaches 0.74.

`object_interaction` improves via a chain: predict the action from motion, then apply
P(object | action), which is also free because every training `object_interaction`
question sits on a clip whose path names the action.

### CLIP embeddings — semantics, still CPU-only

Motion says *how* someone moves and nothing about *what* is in the scene. CLIP ViT-B/32
over 8 IR frames per clip (~50 min of CPU, `src/clip_feats.py`) supplies the other half.
Measured head-to-head under the same subject-grouped protocol:

| Task | motion (22-d) | CLIP (1024-d) |
|---|---:|---:|
| `single \| HARn` | 0.624 | **0.690** |
| `object_interaction` | 0.774 | **0.827** |
| `emotion` | **0.394** | 0.318 |
| `sequence` (per-slot) | **0.276** | 0.214 |

So the features are **chosen per task** (`feats.task_features`) rather than concatenated —
naive concatenation was worse than either on two of the three, since 1024 CLIP dims
swamp 22 motion dims on ~800 training clips.

#### Zero-shot as a second, uncorrelated opinion

CLIP zero-shot (cosine similarity between a clip's frames and text like *"a photo of a
person washing their face"*) scores only 0.564 on `single|HARn` — well below the 0.690 of
the supervised classifier. It still earns a place, because it uses **no training labels**,
so its errors are uncorrelated with the supervised model's:

| | supervised | zero-shot | ensembled |
|---|---:|---:|---:|
| `single \| HARn` | 0.690 | 0.564 | **0.709** |
| `combination` (no-seq) | 0.873 | — | **0.882** |

Similarities are z-scored per clip across the label vocabulary so the fusion weights come
out O(1); raw cosines sit in a narrow band around 0.25 and would otherwise need weights
of ~50. Text embeddings are cached in `features/zs_text.npz`.

Two things that cost accuracy before being caught:

- **PCA-reducing CLIP for the HARn classifier.** Cutting 1024 → 64 dims dropped
  `single|HARn` from 0.690 to 0.571 — *below* the 0.615 that plain motion features give.
  Separating 44 fine-grained actions depends on directions that are not the top-variance
  ones. The presence model, fit per action on ~800 clips, still wants the reduced version.
- **CLIP for temporal ordering.** 0.214 exact vs 0.276 for motion segments. Adjacent time
  slots are the same room and the same person, so semantic embeddings barely move.

### Where it plateaus

`emotion` stops at ~0.40 and resists everything tried: richer per-segment features, CLIP
embeddings (0.318, worse), adverb clustering to fight label fragmentation (K=4…16, all
worse), a conditional logit instead of marginal priors (0.3189 vs 0.3201), and feeding in
the inferred action set as features (0.4079 vs 0.3968 — inside the ±0.017 noise band at
n=809). Manner adverbs look close to their intrinsic ceiling on this data.

`multi` is likewise at ~0.75, and stays there even when the presence model is given
genuinely better supervision (see the dead-ends list). Between them these two categories
are 42% of the test set, which is what bounds a fully honest system near ~0.80.

## Layout

```
data/        competition CSVs + extracted media (gitignored; ~8 GB)
frames/      evenly-sampled IR JPEGs, 1333 train + 208 test clips (270 MB)
features/    motion features (frame-difference energy) per clip
src/         solver, models, CV harnesses
kaggle_notebook/   Qwen2.5-VL evidence extraction, runs on Kaggle's free GPU
submissions/ generated submission CSVs
```

### Key modules

| File | Role |
|---|---|
| `src/solver.py` | data loading, option-content priors, baseline structural model |
| `src/joint.py` | **the main model** — joint clip-level inference (priors + PMI + leaks) |
| `src/fuse.py` | folds VLM evidence into `joint` as an extra log-odds term |
| `src/motion.py` / `src/motion2.py` | CPU frame-difference clip descriptors |
| `src/segments.py` | per-time-slot descriptors, for `sequence` |
| `src/clip_feats.py` | CLIP ViT-B/32 embeddings of the sampled IR frames |
| `src/feats.py` | **per-task feature selection** (`task_features`) |
| `src/motion_model.py` | fitted action / emotion / object / presence models |
| `src/seq_model.py` | temporal-order model + assignment decode |
| `src/zeroshot.py` | CLIP zero-shot evidence (cached text embeddings) |
| `src/cv_full.py` | subject-grouped CV + weight tuning |
| `src/cv_nested.py` | nested CV — honest measure of the tuning gain |
| `src/make_submission3.py` | writes a validated submission |

## Reproducing

```bash
python src/motion2.py both      # clip-level motion descriptors  (~15 min)
python src/segments.py both     # per-time-slot descriptors      (~15 min)
python src/clip_feats.py both   # CLIP embeddings                (~50 min, CPU)
python src/cv_full.py --clip    # tune, writes src/best_full_W.json (~25 min)
python src/make_submission3.py submissions/sub.csv --clip
```

Probes that produced the design decisions above: `src/clip_probe.py` (which features per
task), `src/presence_probe.py` (features for the presence model), `src/seq_probe.py`,
`src/multi_probe.py`, `src/emotion2.py`, `src/condlogit.py`.

## The VLM half — Qwen2.5-VL on Kaggle's free GPU

One source file, `kaggle_notebook/vlm_infer.py`, is built into two Kaggle kernels that run
**in parallel** (Kaggle allows two GPU sessions), ~3.5 h wall-clock:

| Kernel | Scores |
|---|---|
| `cuhk-x-vlm-evidence` | all 682 test questions |
| `cuhk-x-vlm-evidence-train` | a fixed random sample of 200 HAU + 120 HARn training clips, used only to tune fusion weights |

The model emits calibrated per-option scores, never answers: letter log-probabilities for
single-answer questions, a Yes/No log-odds per option for `multi`, and for `sequence` both a
generated order and all six pairwise "which happens first" judgements.

Things learned getting it to run first time:

- **Pin `machine_shape: NvidiaTeslaT4`.** Kaggle's docs state the P100 is not usable with the
  default image (current PyTorch dropped its architecture), and a bare `enable_gpu: true`
  can land on one.
- **Smoke-test before the long run.** `SMOKE = True` scores the fewest clips that exercise
  every question type (~6 min). It found nothing wrong — but a crash one hour into a 3.5 h
  run would have cost most of a day this close to the deadline.
- Kaggle's image shipped `transformers 5.0.0`; the notebook picks `dtype=` vs `torch_dtype=`
  by version and upgrades only below 4.49. Input paths are discovered by walking
  `/kaggle/input` rather than hard-coded, since the mount layout has changed over time.
- Measured on 2×T4 in fp16: **~6.2 s per forward pass** for a 12-frame HAU clip, ~3 s for an
  8-frame HARn clip.

How good is the VLM on its own? Its standalone accuracy on the 988 training questions it
scored (320 clips), next to the CPU pipeline's cross-subject CV:

| Category | VLM alone | chance | CPU pipeline |
|---|---:|---:|---:|
| single \| HARn | **0.745** | 0.333 | 0.699 |
| single \| HAU | 0.660 | 0.250 | 0.930 |
| combination | 0.655 | 0.250 | 0.882 |
| object_interaction | 0.400 | 0.250 | 0.857 |
| multi (exact set) | 0.300 | 0.067 | 0.735 |
| emotion | 0.300 | 0.250 | 0.403 |
| sequence (exact order) | 0.119 | 0.042 | 0.276 |

It beats the existing pipeline outright only on HARn action recognition. Everywhere else the
generator's structural regularities and the CPU models are far stronger, so the VLM's value
is as a partly independent opinion inside the fusion, not as a replacement for anything.

**What made it into the submission is one weight.** Tuning all nine VLM weights failed the
nested check (see the dead-ends list). The one effect large enough to trust is HARn action
recognition -- and because the VLM is zero-shot, its accuracy on training clips is an
unbiased estimate of its test accuracy. Sweeping that single weight (not tuning it) and
reporting each half of the subjects separately:

| VLM weight on HARn `single` | half 0 (n=44) | half 1 (n=58) | all covered clips |
|---|---:|---:|---:|
| 0 (current model) | 0.636 | 0.672 | 0.7046 |
| 0.5 | 0.750 | 0.707 | -- |
| 1 | 0.727 | 0.741 | 0.7105 |
| **5 (used)** | **0.727** | **0.776** | **0.7120** |
| 20 | 0.727 | 0.776 | -- |

Every weight >= 0.5 improves both halves, and 5 sits mid-plateau. Expected effect on the
test set: about +0.1 on HARn's 51 questions, i.e. ~5 answers, ~+0.007 accuracy -- small,
but it is the part of the VLM gain that survives out-of-subject. Every other VLM weight
stays at zero, and VLM evidence is off by default unless a weights file switches it on.

**Then 4× the data (09-11, sub08).** Scoring the other 1013 training clips let each VLM
weight be validated on 4087 questions instead of 988. Four groups now pass on both halves of
the subjects — and pass together:

| Group | held-out half 0 | held-out half 1 | verdict |
|---|---:|---:|---|
| single — HARn | +0.0090 | +0.0035 | accept |
| single — HAU | +0.0029 | +0.0020 | accept |
| combination, clips without a sequence question | +0.0018 | +0.0031 | accept |
| emotion | +0.0018 | +0.0006 | accept |
| combination, clips with a sequence question | −0.0011 | +0.0004 | reject |
| object_interaction | 0.0000 | 0.0000 | reject |
| multi presence (+ its threshold) | −0.0028 | −0.0015 | reject |
| sequence (pairwise + generated order) | 0.0000 | −0.0022 | reject |
| **the four accepted groups, together** | **+0.0162** | **+0.0085** | **ship** |

The model that failed the check on 320 clips passes on 1333: the earlier rejection was a
sample-size problem, not a VLM problem. `emotion`, negative on both halves at n=320, is
positive on both at n=1333. With the larger sample the HARn weight settled at 0.5 rather
than 5. sub08 changes 56 answers vs sub07, all inside the accepted categories.

Fusing the scores:

```bash
kaggle kernels output santanubanerjee9/cuhk-x-vlm-evidence       -p vlm_test_run
kaggle kernels output santanubanerjee9/cuhk-x-vlm-evidence-train -p vlm_train_run
cp vlm_test_run/vlm_scores_test.csv vlm_train_run/vlm_scores_train.csv vlm/
python src/check_vlm.py vlm                               # structure + VLM-alone accuracy
python src/cv_full.py --clip --only-vlm --tune-vlm-only   # writes src/best_vlm_W.json
python src/make_submission3.py submissions/sub07_vlm.csv --clip
```

Only the VLM weights are tuned, and only on the ~320 training clips the VLM scored; every
other weight stays at its optimum on all 1333 clips, so the small covered subset cannot
drag the proven weights around. `make_submission3.py` uses `best_vlm_W.json` only when VLM
test scores are actually present.

## 09-11 — the public 0.77777 file, and the signals it pointed to

Team Fususu released the exact file behind their public 0.77777
(`public_ref/fususu_lb0.77777.csv`). Per their notebook it comes from a structural graph
plus **non-visual** specialists: Skeleton for HARn actions and objects, IMU for HAU emotion.
Both come from the organisers' `LMT_(IMU,Radar,Skeleton).zip`, which this pipeline had not
used. With the user's go-ahead, blends that swap whole categories were refereed on the
public LB (`src/blend_public.py`):

| Submission | Public |
|---|---:|
| sub08 (ours) | 0.72222 |
| theirs for emotion + HARn | 0.75730 |
| \+ theirs for multi + sequence | 0.77485 |
| their file | 0.77777 |

On the public half their answers win in every group. Most of the gap (+12 questions) is in
emotion and HARn, where the non-visual data carries the signal.

Their released *compact* decoder, re-implemented under our subject-grouped CV
(`src/eval_compact_decoder.py`), shows where its structure beats ours:

| Category | compact decoder | ours (sub08) |
|---|---:|---:|
| sequence | **0.409** | 0.276 |
| combination (no-seq / seq) | 0.811 / 0.860 | **0.896 / 0.958** |
| multi (no-seq / seq) | 0.315 / 0.234 | **0.747 / 0.779** |
| single \| HARn | 0.420 | **0.779** |

**Scripted order.** The HAU activities follow a largely fixed script: 64% of the action
pairs in training `sequence` answers occur in one order at least 90% of the time. The motion
view of `sequence` had missed that the order is mostly knowable from the option text alone.
A pairwise-precedence prior between action texts (`SeqModel.prec_scorer`, weight `w_prec`)
was added to the slot model. On held-out subjects it lifts `sequence` exact accuracy from
0.248 → 0.562 and 0.303 → 0.407 (`src/prec_validate.py`). On all subjects it goes
0.276 → 0.490, i.e. +0.012 overall.

The public file's `sequence` answers agree with this model on 77% of test questions (41%
before the prior was added), so theirs look precedence-based too. The two differ on 9 test
questions — too few for the public LB to referee — so the swap is decided on labels instead
(`src/seq_disagree.py`). Over 308 CV questions, our model scores 0.490 and precedence alone
0.422. On the 110 questions where they disagree, ours is right 32 times and
precedence-only 11. So our `sequence` answers go into the final blend.

**Multi.** Distractor combinations are built from the clip's other real actions. So a
`multi` option that appears in two of the clip's combination options is correct 82–95% of
the time, against 9% for one that appears in no other question's options. Added as an
explicit term (`w_rec_m`, `src/struct_validate.py`), it validated at weight **0** on both
halves, because the combination posterior already carries it. What did validate, at
+0.0012 / +0.0010, was a lower inclusion threshold (`thr_m` −2). Together with the
precedence prior this is sub09: 26 answers change vs sub08 (23 sequence, 3 multi).

The public file's `multi` answers resemble a pure recurrence rule much more than ours do:
0.84 vs 0.75 agreement on clips without a sequence question. In 29 of 43 disagreements
their answer includes more options than ours. But that rule scores only 0.64 exact on
training, against our 0.75–0.78, so the rule alone doesn't explain their edge in `multi`.

It turns out they have no edge there. The public scores are exact counts out of 342, and
each question counts independently, so the blends can be combined arithmetically:

- B − A = (their seq + multi) − (our old seq + multi) = +6
- sub09 − sub08 = (our new seq + multi) − (our old seq + multi) = +8
- C − their file = our new seq − their seq = 0

So their `multi` answers are 2 questions *worse* than sub09's on the public half. The labels
say why. A `multi` option inside the true combination is always correct (828 / 828). Options
outside it are correct only 9–30% of the time, even when they recur in two combination
options (17–23%). Their decoder includes more of those options than ours does. Blend D (their
file + our `sequence` + our `multi`) therefore scores exactly 268/342 = 0.78363 on the public
half.

**Single-answer routing ("cross-question relation routing" in their notes).** A clip's
`single` question asks which action is performed, so its answer is always a present action.
A `multi` option equal to it is correct **67 / 67** times even when it lies outside the true
combination, against 9% for other such options. Our model decided `multi` without looking
at `single`. Now `multi` is decided after `single`, and the predicted single answer gets a
bonus `w_sans_m`. Both subject halves chose 10, in effect a hard include, and the held-out
gain was **+0.0080 / +0.0095** (+0.0087 on all subjects). That makes it the second-largest
structural gain after the precedence prior. This is sub10, and it feeds blend E.

**Routing into `combination`.** The true combination's actions show up among the same clip's
`multi` options 47% of the time and its `single` options 26% of the time. Distractor
combinations' actions show up 21% and 15%. Until now the combination score used overlap with
the `sequence` options only. Adding overlap with the multi and single options
(`w_ovl_m`, `w_ovl_s`) gave **+0.0052 / +0.0079** held-out, with `w_ovl_m` = 8 on both
halves. Because the combination beliefs feed `single|HAU` and `multi`, this one change moves
all three categories. This is sub11.

Per category, on all subjects (in-sample, `src/stratum_acc.py`), the gain is uneven:

| Category | before | after |
|---|---:|---:|
| combination, clips without a sequence question | 0.896 | **0.967** |
| combination, clips with one | 0.958 | 0.938 |
| multi, clips without a sequence question | 0.806 | 0.792 |

The routing boosts every combination that contains a multi option. That inflates the
beliefs of all multi options, making each one look present. Group `route_c2` tests two fixes:
scaling the routing down on clips that have a sequence question, and keeping it out of
the beliefs.

## 09-11 — the organisers' non-visual data (IMU, Skeleton)

`LMT_(IMU,Radar,Skeleton).zip` (389 MB, organisers' Drive mirror) mirrors the main release's
layout.
- **IMU:** five wearables (chest, both arms, both legs), ~10 Hz, recording acceleration,
  gyro and angles.
- **Skeleton:** one JSON per video frame with 17 root-centred 3D keypoints.
- **Coverage:** 190 of the 208 test clips have IMU and 195 have Skeleton.

`src/nonvisual_feats.py` reads it straight from the zip. It writes 346 IMU features per clip:
distribution stats, band power, spectral entropy, dominant frequency, autocorrelation and
time-slot means per sensor. It also writes 195 Skeleton features: joint position mean/std,
joint speeds, time-slot energy, joint-pair distances and periodicity.

**HARn units without questions.** Only 429 HARn clips carry a `single` question, but the
package has Skeleton data for 2,925 HARn units. Each unit's action label is its folder name.
So the action classifier can train on ~7× the data, holding out the validation fold's users
(`CUHKX_HARN_UNITS=1`).

Standalone cross-subject probes (`src/nv_probe.py`, `src/nv_probe2.py`); in progress:

| Task | current features | non-visual |
|---|---:|---:|
| emotion | motion 0.393 | IMU **0.464** (random forest) / 0.440 (logistic, C = 0.1) |
| single\|HARn | CLIP ~0.69 | Skeleton **0.793** (all units, logistic C = 1) / 0.73 (question clips only) |
| object_interaction (via action) | — | Skeleton 0.835 (all units) |

For actions, IMU is weaker (0.706 at best), and Skeleton + IMU adds nothing over Skeleton
alone (0.790). Logistic regression beats random forests for Skeleton (0.793 vs 0.762). For
emotion it's the other way round: the random forest wins, as the public notebook's notes
say. IMU + Skeleton ties it at 0.464; Skeleton alone manages only 0.40–0.41.

**Objects are the exception.** For `object_interaction`, read through P(object | action),
CLIP stays clearly ahead: 0.865–0.872, against Skeleton's 0.774 (0.835 trained on all
units). CLIP recognises the object in the scene, while a pose only says what the body is
doing. So the best split is actions from Skeleton and objects from CLIP.

**Emotion, head-to-head in the full pipeline** (`src/nv_validate.py`). Both variants were
accepted, and each version's fusion weight was re-tuned on the other half of the subjects:

| IMU evidence for emotion | held-out half 0 | held-out half 1 | all subjects, no-seq / seq |
|---|---:|---:|---:|
| random forest (used) | +0.0052 | +0.0130 | 0.411 → **0.451** / 0.386 → 0.448 |
| logistic, C = 0.1 | +0.0078 | +0.0043 | 0.411 → 0.423 / 0.386 → 0.484 |

The random forest gains more held-out (+0.018 against +0.012), and about 1.6 more test
questions at the test mix. Build switches: `CUHKX_EMO=imu CUHKX_CLF_EMO=rf`, weights in
`src/best_nv_emorf_W.json`.

sub12e (sub11 + this) changes 66 of the 144 test `emotion` answers, and nothing else. The
fusion weight settled at `w_mot_emotion` = 8, so the IMU evidence now dominates the emotion
decision. Agreement with the public file's emotion answers rises from 0.569 to 0.694,
consistent with theirs coming from the same IMU signal.

**HARn, head-to-head.** Skeleton evidence for HARn actions (logistic, C = 1, trained on all
2,925 HARn units) replaced CLIP for both actions and objects, and was **accepted**:
+0.0109 / +0.0087 held-out.

| On all subjects (in-sample) | before | after |
|---|---:|---:|
| single\|HARn | 0.779 | **0.886** |
| object_interaction | 0.857 | **0.925** |

Objects improved in the full pipeline even though CLIP won the standalone object probe. The
option-text prior already carries what CLIP knew about objects, and the pose adds what it
didn't. A variant keeping objects on CLIP (`CUHKX_OBJ_CLIP=1`) also passed, but was weaker:
+0.0082 / +0.0079 held-out, with objects unchanged at 0.857. Skeleton for both actions and
objects wins on both halves, so it is what ships. sub12 =
sub11 + IMU emotion + Skeleton HARn, with weights in `src/best_nv_final_W.json` and build
switches `CUHKX_EMO=imu CUHKX_CLF_EMO=rf CUHKX_HARN=skel CUHKX_C_HARN=1 CUHKX_HARN_UNITS=1`.

They are switched in with environment variables, all unset by default and regression-checked
to reproduce sub11 exactly:
- `CUHKX_EMO` / `CUHKX_HARN` (feature set, `src/feats.py`);
- `CUHKX_C_*` / `CUHKX_CLF_*` (regularisation strength, random forest) and
  `CUHKX_HARN_UNITS` (`src/motion_model.py`).

`src/nv_validate.py` then compares the new evidence with the current one inside the full
pipeline, re-tuning each on one half of the subjects and scoring on the other.

## 09-13 — recording sessions

The organisers' Skeleton frame filenames carry recording times. `features/unit_times.csv` holds one row per unit with
its first and last frame time. The times reveal the protocol: one activity script is recorded back to back, with a
different manner each time.

- **Test was recorded as 16 unbroken sessions of its own.** Every test clip's nearest neighbour in time is another test
  clip (median gap 60 s), and no test clip sits inside a training session.
- **Consecutive clips of one person never share the emotion adverb** (0 / 725 in training). In the action categories
  they share the answer far more often than chance: sequence 0.29, single 0.23, multi 0.13, combination 0.10.
- **Manners cycle in threes.** The same adverb recurs at lag 2 only 0.011 of the time and at lag 3 0.065, against 0.043
  by chance.
- **sub13e.** IMU features are z-scored within each recording session (no labels, so identical at test time), then
  emotion is decoded along sessions with Viterbi under the no-repeat rule. Held-out emotion accuracy +0.020 / +0.022;
  public **+8, to 0.81578**. sub12 had repeated the adverb in 43 of 127 consecutive test pairs.
- **sub14e.** A second-order decoder (lag-2 repeat penalty) plus neighbour-option evidence: +0.037 / +0.025 over
  sub13e's decoder, with identical fits on both halves. Public 278, one below sub13e on ~8 changed public answers,
  which is noise.
- **Build.** `src/build_session.py`, byte-identical to sub12 with its options off; the decoders live in
  `src/emo_decode.py`. Full numbers and the rejected variants are in `NOTES_interactive_0913.md`.

## Things that looked promising and were not

Recorded so they don't get re-tried.

- **Action name in the HARn path** (`HARn/<action>/<user>/<trial>`) — closed in the test set,
  whose clips are anonymised to `LM_test_XXXX`.
- **Test clip IDs grouped by action** — only 51% of adjacent HARn clips share any option,
  versus 100% for genuine same-action pairs. Not action-blocked.
- **`combination` listing actions in chronological order** — 0.483 agree / 0.517 reverse.
  A coin flip; `sequence` gets no help from it.
- **Fitting option-content priors for `single|HARn`** — the per-option deviations have
  z-score sd 0.86, i.e. pure noise, and argmax over noise scored **below** the 1/3 chance
  level (0.282). Replaced with a uniform prior.
- **HAU trial codes (`1-1-1`) encoding a repeated script across users** — same-code clips
  agree on the emotion answer 13.6% of the time vs a 4.5% baseline. Real but far too weak
  to exploit; the scripts are not repeated.
- **Exact option-tuple repeats between train and test** — 0/144 for emotion, 0/144 multi,
  0/139 combination. No lookup table exists.
- **Generator placement artefacts** — the correct option's alphabetical rank, string-length
  rank, and answer-letter agreement across questions of the same clip are all uniform;
  same-clip letter agreement is 0.239 against a ~0.25 chance level.
- **A set-size prior for `multi`.** Answer-set sizes are heavily skewed (1:312, 2:298,
  3:197, 4:2), so scoring all 15 subsets with a size prior looked obviously right. It made
  things worse at every strength tried (0.745 → 0.687 at λ=0, falling to 0.603 at λ=3).
  The scores are under-confident rather than mis-shaped, so what actually helps is a
  *lower* inclusion threshold — the opposite of what the size prior pushes toward.
- **Properly supervising the presence model.** The presence model derives its labels from
  the `combination` answer and calls everything else absent, which is wrong. The `multi`
  questions carry ~3200 (clip, action) pairs with *explicit* negatives, so retraining on
  those should have been a clear win. It gave 0.7466 vs 0.7454 — noise. `multi` is limited
  by intrinsic label ambiguity, not by how well presence is estimated, and sits at its
  ~0.75 ceiling either way.
- **Tuning all nine VLM fusion weights.** On the 320 training clips the VLM scored, the
  tuner found +0.017 (0.7046 -> 0.7219). The nested check -- tune on half the subjects,
  score on the other half -- gave **-0.003**, negative on *both* halves, and the halves chose
  wildly different weights (`w_vlm_comb` 0.5 vs 3; `w_vlm_single_harn` 2 vs 0.5): nine
  weights on ~320 clips fits noise. Of the in-sample gain, 0.0035 was `thr_m` alone drifting
  to the edge of its grid on a subsample, even though the VLM presence weight it was
  un-frozen for came out at zero. Submitting that configuration would most likely have
  *lowered* the private score.
- **Option-order TTA for HAU questions: validated but not shippable (09-11).** All training
  clips were re-scored under every cyclic option order. The four accepted VLM groups then
  gain **+0.0193 / +0.0095** held-out, against +0.0162 / +0.0085 without TTA, so TTA wins
  on both halves (single|HAU and emotion carry it). The weights move to
  `w_vlm_emotion` 0.25 and `w_vlm_comb_noseq` 2. But the matching test-side kernel
  (`cuhk-x-vlm-tta-test`) could not run: the account's weekly GPU quota ran out. Applying
  TTA-tuned weights to un-TTA'd test scores is a mismatch, so
  `submissions/PARTIAL_sub12_tta_no-test-tta_DO-NOT-SUBMIT.csv` (20 answers differ from sub11)
  must never be submitted.
- **Option-order TTA for HARn questions.** Re-scoring HARn letter questions under every
  cyclic option order (so letter-position bias cancels) gave no held-out gain: the `harn`
  group went +0.0090 / +0.0035 → +0.0090 / +0.0022, `object_interaction` 0 / 0 → 0 / −0.0036,
  and at the validated HARn weight it changed only 2 of 682 test answers. HARn was the weak
  case for TTA — 3 options, a mild bias, and a VLM already strong there; the HAU
  categories, with a much larger measured bias, were tested separately.
- **Recurrence ranking for `combination`.** Picking the option whose actions recur most
  across the other options scores 0.197 / 0.263 — below chance, because the distractors are
  built from the true actions. (For `multi` the same recurrence *is* informative; see 09-11.)
- **Recurrence as an explicit term for `single|HAU`** (`w_rec_s`). It was tuned to 0 on both
  halves of the subjects, a held-out gain of exactly zero. The combination beliefs already
  carry the signal.
- **The VLM's order evidence for `sequence`, on top of the precedence prior** (`w_vlm_pair`,
  `w_gen`; group `vseq`, 09-11). Both halves chose weight 0, a held-out gain of exactly zero.
  The text prior already carries what the VLM's pairwise judgements knew.
- **VLM presence votes for `multi`, re-checked on the new base** (`w_vlm_pres`; group
  `vpres`, 09-11). Weight 0 on both halves, and the threshold stayed at −2. The VLM can't tell
  which actions outside the true combination are present.
- **Refining the combination routing** (group `route_c2`, 09-11). It tried scaling the routing
  down on clips with a sequence question, and keeping it out of the action beliefs. The result
  was −0.0029 / +0.0025: both halves wanted scale 0 on sequence clips but disagreed on
  everything else. sub11's routing stays as it is. For the blend, `multi` is taken from sub10,
  a whole-category choice between two validated configs, made on per-category CV.
- **Positional cues for `sequence` from the other questions.** The single answer's position
  in the true order is close to uniform (52 / 44 / 67 / 61 over positions 1–4). The true
  combination's actions are spread evenly too (214 / 201 / 208 / 209). There's no routing
  signal for order.
- **Skeleton time-slot features for `sequence`** (09-12 night, `src/seq_nv_validate.py --seq=skel`). Standalone, the slot model's exact accuracy rose from 0.273 to 0.338, up on both halves. Inside the full pipeline, with the precedence prior on and (C, w_seq, w_prec) re-tuned per half, held-out was −0.0011 / +0.0007: rejected. The prior already carries that order information. `CUHKX_SEQ` stays in `src/seq_model.py`, unset by default.
- **Pooling `sequence` across consecutive same-option-set clips** (09-13, interactive session, `src/seq_pool_validate.py`). Such runs have the identical true order 74/74 times in training, but pooling the slot evidence across the run gave −0.0067 / +0.0065 held-out (all subjects 0.4885 → 0.4885): rejected. The shared precedence prior already makes a run's clips agree.
- **Presence (IMU + Skeleton) on top of wide neighbour pooling** (09-14, `src/nbpres_joint_validate.py`). Each was accepted alone, but against pooling alone the stack scored -0.0006 / -0.0010 held-out: rejected. Pooling already carries the action evidence presence added.
- **Timestamp-free IMU normalisation for emotion** (09-14, `src/emo_notime_validate.py`, decided per clip). Whole-set z-scoring (transductive) +0.0049 / -0.0100; per-subject with true ids (an upper bound: test paths have no ids) +0.0172 / -0.0050; per-session +0.0319 / +0.0000. All rejected. Without timestamps, sub12's raw-IMU random forest stays.
- **All-lag emotion decoding from likelihood ratios** (09-14, `src/emo_fast_validate3.py`, variant `lagLR6`). Weighting lags 2-6 by lam * log LR and dropping the tuned gamma2 gave +0.0123 / -0.0125: inconsistent, rejected. Keeping gamma2 and adding lam * log LR at lags 3-6 passed (`order2+LR36`, +0.0049 / +0.0050).
- **Session decoding of emotion on raw IMU scores** (09-13, `src/emo_fast_validate.py`). −0.0123 / +0.0424, inconsistent, rejected. The same decoding on per-session normalised IMU was accepted (+0.0196 / +0.0224) and ships as sub13e (0.81578); see NOTES_interactive_0913.md.
- **Skeleton features in the HAU action-presence model** (09-13, `src/pres_nv_validate.py --pres=skel`, `CUHKX_PRES`).
  Appending the 195-d Skeleton table to motion + PCA-64 CLIP, with presence C and the fusion weights re-tuned per
  half, scored −0.0002 / −0.0004 held-out: rejected. Neighbour option pooling over the same categories was accepted
  instead (+0.0097 / +0.0131 on the wide grid).

## 09-13 night / 09-14 - validated additions, timestamps, and a timestamp-free fallback

Validated (held-out halves):
- Wide-grid neighbour pooling (`src/nb_validate_wide.py`; w_nb_comb 16, w_nb_bel 4, thr_m -1.5): +0.0097 / +0.0131,
  replacing the narrow grid. Built as `sub17_nbw` (sub14e emotion) and `sub17_nbw-emo13e` (sub13e emotion).
- Presence with IMU + Skeleton features (`src/pres_nv_validate.py`, C = 0.1): +0.0016 / +0.0017. Skeleton alone was
  rejected. Built alone on sub14e as `sub17_pres`; not stacked with pooling (joint check rejected, -0.0006 / -0.0010).
- Higher-order emotion decoder (`order2+LR36`, `viterbiL` in `src/emo_decode.py`): +0.0049 / +0.0050 emotion accuracy.
  Built as `sub18_nbw-lag`: 4 emotion answers differ from sub17_nbw.

**Timestamps.** Every session feature reads recording order from the timestamps in the test skeleton-frame filenames:
per-session IMU normalisation, session decoding, neighbour pooling and the higher-order decoder. No labels or test
answers are used. The Large-track rules only ban test answers in training and manual labelling. In the Small track,
however, the organisers called matching those filenames to external labelled metadata a data leak (forum topic
714827), and they say Stage 2 favours genuine cross-subject progress. A **timestamp-free fallback** is therefore
kept on the leaderboard: `submissions/sub17t_pres-notime.csv` = sub12 + the IMU + Skeleton presence model.
`make_submission3.py` reads times only under `CUHKX_NB`. Choosing between the two families for the final selection
is the user's decision.
