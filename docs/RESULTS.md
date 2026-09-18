# Every submission, with public and private scores

Public = 342 questions, private = 340. Pulled from the Kaggle API after the competition closed.
Because the categories a change touches are disjoint, the effect of each idea can be read off as a
difference of counts. **Bold** marks the two files selected for final scoring.

| # | File | Date | Public | = /342 | Private | Note |
|---|---|---|---|---|---|---|
| 1 | **`sub18_nbw-lag.csv`** | 2026-09-14 | 0.83918 | 287 | 0.79705 | sub18_nbw-lag: sub17_nbw + higher-order (lags 3-6) emotion decoder |
| 2 | **`sub17_nbw-emo13e.csv`** | 2026-09-14 | 0.83625 | 286 | 0.79411 | sub17_nbw-emo13e: wide neighbour option pooling + sub13e session emotion decoder |
| 3 | `sub17_nbw.csv` | 2026-09-14 | 0.83333 | 285 | 0.79117 | sub17_nbw: wide neighbour option pooling + sub14e order-2 emotion decoder |
| 4 | `sub16_nb-emo13e.csv` | 2026-09-14 | 0.82456 | 282 | 0.78235 | sub16_nb-emo13e: narrow neighbour pooling + sub13e emotion (reference) |
| 5 | `sub15e_emo-order2.csv` | 2026-09-15 | 0.82163 | 281 | 0.77352 | sub15e_emo-order2: sub12 + per-session IMU emotion, order-2 decoder (emotion only) |
| 6 | `sub15_nb.csv` | 2026-09-13 | 0.82163 | 281 | 0.77941 | sub15_nb: sub14e + time-neighbour option pooling for combination / single/HAU / multi (option texts of neighbouring clips in the same recording sessio |
| 7 | `sub17_pres.csv` | 2026-09-15 | 0.81578 | 279 | 0.78529 | sub17_pres: sub14e + IMU/Skeleton presence |
| 8 | `sub13e_emo-norm-sess.csv` | 2026-09-13 | 0.81578 | 279 | 0.77941 | sub13e: sub12 + emotion from IMU features normalised within each recording session, decoded along sessions under 'consecutive clips never share the ad |
| 9 | `sub17_nbw_base.csv` | 2026-09-15 | 0.81286 | 278 | 0.79117 | sub17_nbw_base: sub12 + wide neighbour option pooling |
| 10 | `sub16e_emo-nbopt.csv` | 2026-09-15 | 0.81286 | 278 | 0.78235 | sub16e_emo-nbopt: sub12 + per-session IMU emotion, neighbour-option term (emotion only) |
| 11 | `sub14e_emo-order2nb.csv` | 2026-09-13 | 0.81286 | 278 | 0.77647 | sub14e: sub13e's emotion decoder refined - second-order session Viterbi (no repeated adverb at lag 1, penalty at lag 2; the protocol cycles manners in |
| 12 | `sub15e_emo-norm.csv` | 2026-09-13 | 0.80994 | 277 | 0.76764 | sub15e-norm (component check): sub12 + emotion from per-recording-session normalised IMU, independent argmax WITHOUT session decoding (validated +0.02 |
| 13 | `sub15_nb_base.csv` | 2026-09-15 | 0.80116 | 274 | 0.77941 | sub15_nb_base: sub12 + narrow neighbour option pooling |
| 14 | `blendF2_theirs-G1_ours-rest.csv` | 2026-09-12 | 0.80116 | 274 | 0.75882 | blend F2: their emotion/HARn/object answers; ours everywhere else (sequence, multi, combination, single/HAU from sub11). Per-category public counts: o |
| 15 | `sub17t_pres-notime.csv` | 2026-09-14 | 0.79532 | 272 | 0.78529 | sub17t_pres-notime: timestamp-free fallback = sub12 + IMU/Skeleton presence |
| 16 | `blendS1_theirs-HARn-obj_ours-emotion-rest.csv` | 2026-09-13 | 0.79239 | 271 | 0.76470 | blend S1 (G1 split): blend F2 + OUR 44 differing emotion/HAU answers (IMU random-forest, held-out +0.0052/+0.0130); nothing else differs from F2. Whol |
| 17 | `sub12_nv.csv` | 2026-09-12 | 0.79239 | 271 | 0.77647 | sub12: sub11 + organisers' non-visual data - IMU random-forest emotion (held-out +0.0052/+0.0130) and Skeleton HARn trained on all 2925 units (+0.0109 |
| 18 | `blendC_theirs-but-our-seq.csv` | 2026-09-11 | 0.77777 | 266 | 0.71176 | Blend C: public release (team Fususu 0.77777) with OUR sequence/HAU answers (precedence prior + slot model; CV: on disagreements with precedence-only  |
| 19 | `blendB_theirs-G1G2.csv` | 2026-09-11 | 0.77485 | 265 | 0.72352 | Blend B: blend A + public release answers for multi/HAU and sequence/HAU. Whole-category test, 148 answers differ from sub08. |
| 20 | `sub11_routec.csv` | 2026-09-12 | 0.76608 | 262 | 0.72941 | sub11: sub10 + combination routing (its actions recur among the clip's multi/single options; held-out +0.0052/+0.0079); 17 answers differ |
| 21 | `blendA_theirs-G1.csv` | 2026-09-11 | 0.75730 | 259 | 0.70000 | Blend A: sub08 + public release (team Fususu, 0.77777) answers for emotion/HAU, single/HARn, object_interaction/HARn (their IMU/Skeleton specialist ca |
| 22 | `sub11g3b_sub09-plus-G3.csv` | 2026-09-12 | 0.75438 | 258 | 0.70882 | measurement: sub09 with only sub11's combination + single/HAU answers (13 differ) - isolates the combination-routing gain |
| 23 | `sub10_route.csv` | 2026-09-12 | 0.75146 | 257 | 0.70294 | sub10: sub09 + single-answer routing for multi (the clip's single answer is always a present action; held-out +0.0080/+0.0095); 10 multi answers diffe |
| 24 | `sub09_struct.csv` | 2026-09-11 | 0.74561 | 255 | 0.68823 | sub09 (pure model): sub08 + pairwise-precedence prior for sequence (held-out +0.018/+0.006 overall; sequence exact CV 0.276->0.490) + lower multi incl |
| 25 | `sub08_vlm_perweight.csv` | 2026-09-11 | 0.72222 | 247 | 0.67058 | sub08: Qwen2.5-VL evidence in 4 groups validated on held-out subjects (HARn, single/HAU, combination w/o sequence, emotion) after VLM-scoring all 1333 |
| 26 | `sub07_vlm_harn.csv` | 2026-09-10 | 0.71052 | 243 | 0.65882 | sub07: Qwen2.5-VL evidence for single/HARn only (w=5). Full 9-weight VLM tuning failed nested check (-0.003); HARn-only improves both subject halves.  |
| 27 | `sub06_zeroshot.csv` | 2026-09-04 | 0.70467 | 241 | 0.65294 | Adds CLIP zero-shot as independent evidence (HARn action ensemble + HAU presence). 16-frame CLIP for HAU clips. CV 0.7181 |
| 28 | `sub05_clip.csv` | 2026-09-04 | 0.69590 | 238 | 0.65588 | Per-task features: CLIP-1024 for HARn action + object chain, motion for emotion + temporal order, motion+clipPCA for HAU action presence. CV 0.7134 |
| 29 | `sub04_tuned.csv` | 2026-09-04 | 0.67836 | 232 | 0.62647 | Joint structural + motion + action-presence + temporal-order, all weights tuned cross-subject. CV 0.7079 |
| 30 | `sub03_motion.csv` | 2026-09-04 | 0.67543 | 231 | 0.63235 | Joint structural model + CPU motion models (HARn action clf, emotion adverb clf, object via action, temporal-order assignment). CV 0.673 + sequence 0. |
| 31 | `sub02_joint.csv` | 2026-09-04 | 0.60818 | 208 | 0.59411 | Joint clip-level model: option-content priors + action co-occurrence (PMI) + within-clip structural constraints. Cross-subject CV 0.636 |
