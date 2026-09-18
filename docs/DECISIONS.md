# How this solution was built: decisions, reasoning and dead ends

A compressed record of the work, written at the end of the competition so the reasoning survives
independently of any chat history. Scores in the text are public-leaderboard counts out of 342; the full
ledger with private scores is in [`RESULTS.md`](RESULTS.md).

## The problem, and what made it unusual

682 multiple-choice questions about privacy-preserving (depth / IR) videos of one person doing everyday
activities, across seven question types. The test subjects are disjoint from the training subjects, so
anything learned about a particular person is useless. Two facts shaped every decision:

- **The public leaderboard is an exact instrument.** Scores are clean `x/342` counts, and a change that
  touches one category cannot move another. So the public effect of an idea can be isolated arithmetically,
  and the score of an unsubmitted combination is often known in advance. `sub16_nb-emo13e` was predicted at
  exactly 282 before it was ever submitted; it scored 282.
- **A weak category dominates the ceiling.** `emotion` (manner adverbs: quickly, hurriedly, leisurely …)
  is 21% of the questions and sat near 0.40 accuracy for most of the competition. Roughly half the
  remaining errors lived there, which is why so much of the later work targets it.

## Working method

This mattered more than any single model.

1. **Two independent held-out checks before anything ships.** 6-fold cross-subject CV, plus a half-split:
   tune on one half of the subjects, score on the other, and accept only if the change is positive on
   **both** halves. Several ideas that looked good on one half were negative on the other and were dropped;
   they are listed under dead ends.
2. **Decision rules fixed before seeing results.** When the neighbour-pooling weights landed on the edge of
   their grid, one wider re-run was allowed, with the rule written down first: replace the old configuration
   only if the wider grid wins on both halves. It did (+3 → +7 public). No third widening, because
   re-searching the same subjects repeatedly is how overfitting starts.
3. **Leak guards on every build.** Each new submission is diffed against its parent and refused if any
   answer moved outside the categories the change was supposed to touch. After any edit to shared code, an
   earlier submission is rebuilt and compared byte for byte.
4. **The public leaderboard measures, it does not tune.** Weights were never chosen by leaderboard feedback,
   and no per-question selection was ever made.

## The evidence stack, in the order it was built

| Stage | Idea | Public |
|---|---|---|
| sub01–04 | Option-text priors, action co-occurrence, and within-clip structure: a clip's `sequence` options are exactly the actions in that clip, which constrains its `single` / `multi` / `combination` answers | 0.608 → 0.678 |
| sub05–06 | CPU motion descriptors (frame differencing on the Depth stream) plus CLIP ViT-B/32 semantics, PCA-reduced | 0.696 → 0.705 |
| sub07–08 | Qwen2.5-VL-7B evidence. Full 9-weight tuning overfit, so the VLM was accepted per group after scoring all 1,333 training clips | 0.711 → 0.722 |
| sub09–11 | Pairwise precedence prior for `sequence`; single-answer routing for `multi`; combination routing | 0.746 → 0.766 |
| sub12 | Organisers' non-visual supplement: IMU random forest for `emotion`, Skeleton classifier for HARn actions and objects trained on all 2,925 HARn units | **0.792** |
| sub13e–sub18 | Recording-session structure (below) | **0.839** |

## The four decisions that mattered

### 1. Blending with another team's released answers, then dropping it

A public notebook released a 0.77777 answer file while our own model was at 0.746. Category-level blends
were tested — whole categories only, never per question — and they did beat us at first. Two things ended
that line: our own model passed the released file in **every** category by 09-13, and the organisers
re-run top teams' code, so a blend containing someone else's answers cannot reproduce as our work. Every
submission selected at the end is entirely our own code.

### 2. Recording-session structure, and the disclosure that goes with it

The Skeleton frames in the organisers' supplement are named `Color_<date>_<time>_<frame>.json`. Sorting
clips by that time and splitting at 30-minute gaps recovers the recording sessions; the test set forms 16
unbroken ones. The training data then shows strong regularities:

- consecutive clips **never** share an emotion adverb (0 of 725);
- adverb repeats by lag, against a chance rate of 0.043: lag 2 0.011, lag 3 0.065, lag 4 0.007, lag 6 0.096
  — the protocol cycles manners in threes;
- consecutive clips share action answers far above chance.

Three mechanisms exploit this: per-session normalisation of the IMU features (+6 public), Viterbi decoding
of `emotion` along each session under the no-repeat rule with lag weights (+2, then +2 more for lags 3–6),
and pooling the option *texts* of time-neighbouring clips into the action scores (+7).

Together that is +17 of our 287, and it is the reason this deserved a decision rather than a shrug. During
the final days the Small Model Track's organisers ruled on a report where test skeleton filenames had been
matched to an external labelled file, calling it a data leak, and said Stage 2 favours "genuine progress"
on the cross-subject problem. Our use is different in kind — no labels, no external file, no test answers,
and the Large Model Track's rules prohibit only training on test answers and hand-labelling — but it is the
same timestamps. So:

- it was raised as an explicit decision rather than buried;
- a variant built **without any recording-time information** (`sub17t_pres-notime`, 0.79532 against 0.83918)
  was built and submitted so the effect can be measured by anyone;
- and it is disclosed in the README and in the reproduction survey, in plain words.

An idea that would likely have gained more — decoding HARn actions along the fixed recording script — was
deliberately **not** attempted, because it would lean harder on the same artefact without the user's
agreement.

### 3. Not stacking things that were never validated together

Presence modelling with IMU + Skeleton passed its own held-out check (+0.0016 / +0.0017), and neighbour
pooling passed its own (+0.0097 / +0.0131). Both feed the same action beliefs, so stacking them needed its
own check — which came back **negative** (−0.0006 / −0.0010). Presence was therefore left out of the final
model even though it was individually accepted, and its separate submission (`sub17_pres`, 279) measures it
in isolation.

### 4. The final pair

Kaggle scores the better of two selected submissions. The choice was between pairing our best file with a
second strong variant, or with the timestamp-free build as a hedge. The hedge was rejected on reasoning:
Kaggle takes the maximum either way, so a clean second pick cannot protect a ranking, while a second strong
variant (differing in 19 emotion answers) has real upside at a rank sitting exactly on the top-15%
boundary. Selected: `sub18_nbw-lag` and `sub17_nbw-emo13e`. The private scores afterwards confirmed the
order: 0.79705 and 0.79411, the best two of everything submitted.

## Dead ends — all validated, all rejected

Kept because knowing what does **not** work is half the value of a record like this.

| Idea | Result |
|---|---|
| Option-order TTA for the VLM (its letter-position bias is real and large) | No held-out gain on HARn; HAU never ran, GPU quota exhausted |
| Full 9-weight joint VLM tuning | Overfit; per-group acceptance used instead |
| Pooling `sequence` across consecutive same-option-set clips | −0.0067 / +0.0065 — the shared precedence prior already aligns them |
| Skeleton time-slot features for `sequence` | −0.0011 / +0.0007 in the full pipeline, despite a large standalone gain |
| IMU time-slot features for `sequence` | Weaker standalone than Skeleton, not pursued |
| Session decoding of `emotion` on raw (unnormalised) IMU scores | −0.0123 / +0.0424, inconsistent |
| Timestamp-free IMU normalisation (whole test set, per subject, per session), decided per clip | All rejected; per-subject with **true** ids also failed, so clustering test subjects was not worth building |
| All-lag emotion decoding with every weight from likelihood ratios (`lagLR6`) | +0.0123 / −0.0125 |
| Presence (IMU+Skeleton) stacked on neighbour pooling | −0.0006 / −0.0010 |
| Skeleton-only presence; `route_c2`; recurrence for combination/single; exact option-tuple matching between train and test | Rejected |

## What I would try next

1. **The 288 that got away.** On the final day, `sub15e_emo-order2` revealed that the *simpler* order-2
   emotion decoder scores 3 higher on public than the decoder inside our best file. Combined with wide
   pooling it would have been exactly 288, one above our best — but the day's five submission slots were
   already spent. Cross-validation rated the two decoders equal, so this may be noise on 342 questions;
   it is still the first thing to test.
2. **HARn along the recording script**, if and only if the organisers consider recording order legitimate.
3. **VLM test-time augmentation over option order** for HAU questions, where the position bias is largest.
   Never ran: the shared weekly GPU quota was gone.
4. **A stronger emotion signal from the sensors themselves.** Everything that lifted emotion came from
   normalisation and sequence structure; the per-clip classifier itself never beat ~0.46 accuracy.
