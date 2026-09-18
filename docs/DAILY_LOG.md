# Daily log — CUHK-X Large Model Track

One short entry per day: what changed, why, the held-out (validated) gain, the public score.

## 2026-09-04 — sub02 → sub06 (public 0.608 → 0.705)
Generator structure first, then CPU motion features, CLIP, and CLIP zero-shot. Details in README.

## 2026-09-11 — sub07: VLM for HARn actions only
Ran Qwen2.5-VL-7B on Kaggle T4s. Tuning all 9 VLM weights looked +0.017 in-sample but was
−0.003 on held-out subjects → rejected. The one robust effect: VLM on single|HARn
(0.657 → 0.755, improves both subject halves). Expected +0.007 overall.
Public: **0.71052** (243/342), up from 0.70467 — +2 on the public half, in line with the
expected ~+5 across all 682. 17 answers changed, all single|HARn (leak guard passed).
Later the same day: all 1013 remaining training clips were VLM-scored. With 4× the data, four
VLM signals validate on held-out subjects (HARn, single|HAU, combination without a sequence
question, emotion; together +0.016 / +0.009). sub08 is built and leak-guarded (56 answers
changed) — it ships 09-12. Option-order TTA kernels are queued for the 09-13 upgrade.

## 2026-09-11 (cont.) — sub08 early; blend tests with the public 0.77777 file
- sub08: public **0.72222** (247/342), +4 on the public half. Submitted a day early because
  the user noted 5 submissions/day are allowed.
- The user chose to blend with team Fususu's released 0.77777 file by whole category,
  refereed by the public LB.
  - Blend A (theirs for emotion + HARn): 0.75730.
  - Blend B (plus theirs for multi + sequence): 0.77485.
  - Their file alone: 0.77777.

  Theirs is better in every group, so their file is the new base. Their edge comes from
  non-visual data (Skeleton, IMU) we have not used yet; the organizers' zip is pending (gated
  on HF, Drive quota exceeded).
- Two structural signals our model lacks, now being validated on held-out subjects: scripted
  action order for `sequence` (pairwise precedence) and option recurrence for `multi`.
- **sub09 (pure model): public 0.74561** (255/342), +8 over sub08. It adds the precedence
  prior for `sequence` (CV exact 0.276 → 0.490) and a lower `multi` inclusion threshold, with
  a held-out gain of +0.019 / +0.007. Recurrence as an explicit term added nothing, since the
  combination beliefs already carry it.
- Blend C (their file + our `sequence` answers): 0.77777, the same as their file. The 9
  differing answers net zero on the public half. On CV our sequence model wins
  disagreements 32–11, so blend C is the current best candidate.
- Blocked: the organisers' IMU/Skeleton zip. The HF copy is gated (the user has been asked
  to accept access); the Drive mirror is over quota, with retries every 45 min.
- Combining the exact public counts shows our new `multi` beats theirs by 2 on the public
  half. Blend D (their file + our sequence + our multi) should therefore score exactly
  268/342 = 0.78363, above the 31-team tie. It is queued as 09-12's first submission; today's
  quota is used up.
- Single-answer routing for `multi`. A clip's single answer is always a present action
  (67/67 in training), and our model wasn't linking the two. Held-out gain +0.0080 / +0.0095.
  sub10 (10 multi answers changed) and blend E (= blend D with those 10 answers) are built
  and queued for 09-12. Blend E's public score = 268 + (sub10 − sub09), exactly.
- Rejected on the new base: VLM order evidence for `sequence`, and VLM presence votes for
  `multi`.
- Routing into `combination`: overlap with the clip's multi and single options. Held-out gain
  +0.0052 / +0.0079. sub11 is built, and it changes combination, single|HAU and multi. A
  G3-only hybrid (sub10 + sub11's combination/single answers) is also built, so the
  public-count arithmetic can tell whether our G3 now beats theirs.
- The refinement `route_c2` was rejected (−0.0029 / +0.0025). There are four submissions
  queued for 09-12, in order:
  1. G3-only hybrid on sub09.
  2. sub10.
  3. sub11 (the pure hedge).
  4. Blend E or F3, whichever the exact counts favour. Blend F3 wins only if our G3 beats
     theirs on the public half.
- Option-order TTA for HAU validates on the training side: +0.0193 / +0.0095 held-out,
  against +0.0162 / +0.0085 without. The test kernel could not run because the account's
  weekly Kaggle GPU quota (30 h, shared with the user's notebooks) ran out ~20:30 IST; the
  user was notified. TTA is on hold until they OK more GPU time.
- The organisers' IMU/Radar/Skeleton zip arrived at 20:40 IST (Drive try 12). Features were
  extracted with `src/nonvisual_feats.py`: IMU 346-d, Skeleton 195-d.
- IMU emotion (random forest) wins the head-to-head against the current motion evidence:
  +0.0052 / +0.0130 held-out, emotion 0.41 → 0.45. Skeleton for HARn actions, trained on
  all 2,925 HARn units, scores 0.79 standalone; its head-to-head is running, with and
  without keeping objects on CLIP.
- sub12e (sub11 + IMU emotion) is built and leak-guarded: 66 emotion answers change.
- Skeleton HARn (all units) wins its head-to-head: +0.0109 / +0.0087 held-out. single|HARn
  goes 0.78 → 0.89 and objects 0.86 → 0.92 (in-sample). sub12 (emotion + HARn) is built
  and leak-guarded: 77 answers change vs sub11 (66 emotion, 5 single|HARn, 6 objects). It
  replaces sub12e as tomorrow's 4th submission. Agreement with the public file rises to
  0.69 / 0.82 / 1.00 in those three categories, so our non-visual answers now mostly match
  theirs.
- Keeping objects on CLIP also validated, but weaker (+0.0082 / +0.0079, objects unchanged).
  sub12 keeps Skeleton for both. Tomorrow's five submissions are final.
  Tomorrow's plan is now five submissions: the G3 hybrid, sub10, sub11, sub12e (or sub12),
  then the best of four pre-built blends. All four have public counts known exactly from
  the first four scores.

## 2026-09-12 — five submissions; our own model passes their file; blend F2 at 0.801
- The scheduled run fired at 08:06 IST but stopped 19 s in, at its first shell command: an
  automated run cannot answer a tool-permission prompt. It submitted nothing. The user asked
  about it in the evening and all five went in then, 21:57-22:10 IST.
- Public counts (out of 342), each isolating one change:
  - G3 hybrid 258: combination routing **+3**
  - sub10 257: single-answer routing for multi **+2**
  - sub11 262: so sub11's multi is **+4** over sub09's (the public count reverses the CV's
    preference for sub10's multi, by 2 questions)
  - sub12 **271 = 0.79239**: IMU emotion + Skeleton HARn **+9**. Our own model now beats
    their file (266) without using any of its answers.
- Per category, ours - theirs on the public half: sequence 0, multi +6, combination and
  single|HAU +2, emotion/HARn/objects -3.
- So the best blend keeps only their emotion/HARn answers: **blend F2 = 274/342 = 0.80117**
  expected, at the Top-15% line (0.80116). Submitted as the day's fifth.
- **Confirmed: blend F2 = 0.80116 = exactly 274/342**, matching the arithmetic prediction to
  the question. Rank 32 of 201, top 16%. Top-30% (0.77777) is cleared; the top-15% bar has
  moved to 0.81286 as the board filled, so 4 more questions are needed.
- Next: split G1 (one swap for emotion, one for single|HARn; objects already match theirs),
  and try the non-visual features on the categories they have never touched - the sequence
  slot model and the HAU presence model, which still run on motion/CLIP only.

## 2026-09-12 night (scheduled run, 23:53 IST) — no submission; 09-13 files built
- No quota left: the five 09-12 submissions (UTC day) were already in. The quota resets at 05:30 IST; nothing was submitted.
- 09-13 G1 split is pre-built and leak-guarded against F2. S1 (`blendS1_...`, md5 f48f91e261bd) = F2 + our 44 differing emotion answers. S2 (`blendS2_...`, md5 1e7a78b9aa20) = F2 + our 9 differing single|HARn answers. Objects are identical to their file, so **S1 + S2 = 545 exactly**: submit S1, derive S2, and submit S2 only if it is > 274.
- 09-14 idea started early: Skeleton/IMU time-slot features for the `sequence` slot model (`src/seq_nv_probe.py`, switch `CUHKX_SEQ` in `src/seq_model.py`; unset = byte-identical). Standalone cross-subject exact: motion 0.273 → +skel 0.338 (C=0.3; both halves up), +imu 0.299, nv-only 0.334.
- **Rejected in the full pipeline** (`src/seq_nv_validate.py --seq=skel`, head-to-head with (C, w_seq, w_prec) re-tuned per half): held-out **−0.0011 / +0.0007**; sequence exact 0.562 → 0.542 and 0.406 → 0.419. The precedence prior already carries the order signal the slot model gains. Nothing is shipped from it. Next (09-13 05:30 IST onwards): submit S1, derive S2, then HAU presence with IMU/Skeleton for multi/combination.

## 2026-09-13 (scheduled run, 11:07 IST) — G1 split; recording-session structure
- **S1 (F2 + our emotion) = 0.79239 = 271/342**: their emotion is 3 questions better than ours. Derived S2 = 545 − 271 = 274, a tie with F2, so S2 was not submitted. **F2 (274) stays the best blend.** Ledger G1 = emotion +3, single|HARn 0, objects 0.
- The interactive session found the recording-session structure (features/unit_times.csv, from Skeleton frame times; test = 16 unbroken sessions). Consecutive clips never share the emotion adverb (0/725). Per-session normalised IMU + Viterbi decoding validated **+0.0196 / +0.0224** held-out on emotion, giving sub13e (emotion only, `src/build_session.py`). Sequence pooling over same-set runs was rejected (−0.0067 / +0.0065). Details in NOTES_interactive_0913.md.
- This run: IMU/Skeleton appended to the HAU presence model (`CUHKX_PRES`, `src/pres_nv_validate.py`) and time-neighbour option pooling for combination / single|HAU / multi (`CUHKX_NB`, `src/neighbours.py`, `src/nb_validate.py`; neighbours' option texts put the true combination in the top set 90% of the time in training, no answers used). Both switches are off by default, and the sub12 rebuild is byte-identical. Held-out verdicts: pending at 07:10 UTC (CPU shared by 3 sessions).
- Next: build whichever validates on sub12, add sub13e's emotion, leak-guard to combination / single|HAU / multi, submit; its count − sub13e isolates the change. Reproducibility is scored for top teams, and blends with Fususu's answers can't be regenerated, which matters for the 09-15 selection.
- **sub13e = 0.81578 = 279/342** (verified on Kaggle, md5 1d5d3c2018f2), +8 over sub12 from emotion alone. It is our best submission and uses no answers from their file. Ledger (ours − theirs): emotion +5, multi +6, combination/single|HAU +2, sequence 0, single|HARn 0, objects identical. Their file is no longer ahead in any category, so the best blend is our pure model and blends are no longer needed. That also removes the reproducibility problem if two pure models are selected on 09-15. Above the 09-12 top-15% bar of 0.81286; the current bar was not re-checked, since the CLI shows only the top 20.
- **sub14e = 0.81286 = 278/342** (md5 9f597d881241). Its emotion decoder adds a lag-2 repeat penalty and a neighbour-option term: held-out **+0.0368 / +0.0249** over sub13e's decoder, with identical fits on both halves. It changes 17 emotion answers vs sub13e and scores 1 fewer on the public half, which is within noise for 17 answers. The held-out evidence favours sub14e and the public count favours sub13e, so both are candidates for the 09-15 selection.
- Run ended 07:40 UTC. The presence (skel) and neighbour-pooling validators were still tuning on held-out halves (the CPU was shared by 3 sessions; imu+skel was stopped to free CPU), so nothing from them was built or submitted. Both processes keep running, and the interactive session picks up `src/best_nv_pres_W_skel_report.json` / `src/best_nb_W_report.json` and builds at most one group, and only if it is positive on both halves (recipe in STRATEGY_PLAN.md, 09-14 row). Submissions used today: S1, sub13e, sub14e (2 left).

## 2026-09-13 22:00 IST (scheduled run) - no submission, to avoid colliding with the interactive session
- Kaggle shows 3 of 5 slots used for UTC 09-13 (S1, sub13e 279, sub14e 278). The 2 remaining slots expire 05:30 IST.
- The interactive session (FEEDBACK 21:50: "use all remaining submissions") owns them. Its background jobs are live: `nb_validate.py` and `pres_nv_validate.py --pres=skel` (restarted 21:53, no verdict yet), builds of `sub15e_emo-norm.csv` (md5 56248eae6a9d; norm-only, held-out +0.0221 / +0.0050; 37 answers differ from sub13e) and `sub15e_emo-order2.csv` (order2-only, +0.0392 / +0.0200), and a 03:30 IST cutoff timer to submit a fallback if no group validates.
- This run submitted nothing, so the same file can't be submitted twice and a validated group doesn't lose its slot. No kernels launched (GPU quota; TTA still needs the user's OK). Next: the interactive session fills both slots before 05:30 IST; the 09-14 run uses all 5 per FEEDBACK.

## 2026-09-13 evening (interactive session) — all slots in use
- The user: "You should complete all remaining submissions, daily" (recorded in FEEDBACK.md). All 5 slots get
  used every UTC day, each on a distinct validated build or a component-isolating variant.
- The handed-over presence and neighbour-pooling validators died when the scheduled session ended, without writing
  reports. Relaunched ~21:53 IST under the interactive session.
- 4th submission today (21:58 IST): `sub15e_emo-norm` (normalised IMU, no session decoding; validated
  +0.022 / +0.005). It differs from sub13e in 37 emotion answers, so count − 279 isolates the session decoding.
- 5th slot: whichever validator accepts first (one group, `src/build_stack.py`, count − 278 isolates it). If none
  has by 03:30 IST, `sub15e_emo-order2` (order-2 decoder without the neighbour term) goes in before the 05:30 reset.
- `sub15e_emo-norm` = **0.80994 = 277/342**: +6 over sub12 from per-session IMU normalisation alone, 2 below sub13e,
  so session decoding adds +2 on the public half (consistent with held-out). A regex over the CLI table misread it as
  0.66300 at first; always read scores via the Kaggle API (`competition_submissions` → `public_score`).
- **Neighbour option pooling accepted** (relaunched, 22:49 IST): held-out +0.0045 / +0.0025 on
  combination / single|HAU / multi, weights w_nb_comb 4, w_nb_bel 2. Building `sub15_nb` on sub14e as today's 5th
  submission (count − 278 isolates it). Presence (skel) is still running; if it passes it is submitted separately on
  09-14, not stacked.
- `sub15_nb` submitted 22:53 IST as today's 5th: 8 answers differ from sub14e (single|HAU 5, multi 2,
  combination 1), both leak guards OK, md5 324bcc8f6a51. All 5 of today's slots are used.
- **sub15_nb = 0.82163 = 281/342**, our new best (API read): neighbour pooling adds **+3** on the public half from 8
  answers. The categories don't overlap, so `sub16_nb-emo13e` (neighbour pooling + sub13e's emotion, pre-built)
  scores exactly **282**. It is the 09-14 queue's first submission.
- Pre-built for 09-14: `sub16_nb-emo13e` (md5 5dfc75a7ae3f, exactly 282) and `sub16e_emo-nbopt` (md5 f06c2f102aeb).
  Overnight validators: presence skel, presence imu+skel, and neighbour pooling on a wider grid (it replaces the
  accepted weights only if it wins on both halves).
- **Neighbour pooling, wider grid: accepted at +0.0097 / +0.0131 held-out**, beating the narrow run on both halves,
  so it replaces it (w_nb_comb 16, w_nb_bel 4). `sub17_nbw` (sub14e's emotion) and `sub17_nbw-emo13e` (sub13e's) are
  being built for the top of the 09-14 queue.
- Built for the top of the 09-14 queue: `sub17_nbw-emo13e` (md5 fea8881e2ffd) and `sub17_nbw` (md5 f91de8f65162).
  Both leak guards OK; each changes 16 combination / single|HAU / multi answers vs its base.
- Presence with Skeleton features: **rejected**, −0.0002 / −0.0004 held-out. Only the IMU + Skeleton variant is still
  validating.
- Presence with IMU + Skeleton features: **accepted**, +0.0016 / +0.0017 held-out (small). Not stacked with
  neighbour pooling (no joint check); built alone on sub14e as `sub17_pres` for 09-14 queue item 3.
  `sub17_pres` built at 00:20 IST (md5 ff1d3dc392b9; 14 action answers differ from sub14e).
- 09-14 00:45 IST: higher-order emotion decoder `order2+LR36` accepted, but small (+0.0049 / +0.0050 emotion
  accuracy, about +0.7 test answers). The Small-track session flagged that organisers treated test-filename timestamps
  as a leak (topic 714827). Every session-structure gain (+11 public) uses those timestamps. Raised with the user as
  a final-selection decision; a timestamp-free fallback is being put on the board.
- 01:25 IST: `sub18_nbw-lag` built (md5 86e216768efc; sub17_nbw with the higher-order emotion decoder, 4 emotion answers
  differ). The `--lagw` builder edit is regression-clean: a sub14e rebuild is byte-identical.
- 01:37 IST: timestamp-free IMU normalisation for emotion rejected (whole-set, per-subject and per-session, each decided
  per clip). The fallback stays `sub17t_pres-notime`.
- 02:45 IST: joint check presence + wide pooling **rejected** (−0.0006 / −0.0010). 09-14 slot 3 swapped from `sub17_pres`
  to `sub18_nbw-lag` (best session model by CV).

## 2026-09-14 05:31 IST (interactive session): 5 submissions

- **sub18_nbw-lag 0.83918 = 287/342, new best** (wide neighbour pooling + higher-order emotion decoder).
  sub17_nbw-emo13e 286, sub17_nbw 285, sub16_nb-emo13e 282, timestamp-free fallback sub17t_pres-notime 272.
- Wide pooling is +7 on public (narrow was +3), the lag decoder +2, presence +1. CV and public agree on the best file.
- Strategy (brief): all timestamp-based gains are validated and shipped. The remaining decision is the user's:
  timestamp models or the timestamp-free fallback in the final pair (options A/B/C in NOTES).

## 2026-09-14 07:45 IST (scheduled run) - log only, no submission
- Quota: all 5 slots for UTC 09-14 were already used at 05:31 IST by the interactive session. **Best: sub18_nbw-lag 287 = 0.83918**, timestamp-free fallback sub17t_pres-notime 272.
- No new upgrade this run. The owner session `cuhk-x-competition-large-model-3e` is alive and owns the 09-15 slots, and extra CPU validators would compete with it (as happened 09-13). No kernels: the GPU pool is shared with the Small track, and TTA still needs the user's OK.
- Checked: all 5 files in the 09-15 queue exist with the md5s in the plan and pass the format check (682 rows, `qa_id,prediction`).
- Next: 09-15 05:30 IST, submit the 09-15 queue (`src/submit_queue.py`), then remind the user to SELECT the final pair (B recommended: sub18_nbw-lag + sub17t_pres-notime).

## 2026-09-15 21:17 IST (scheduled run) - final day, all 5 slots used
- Owner session `-3e` was gone and nothing had been submitted for UTC 09-15, so per the fallback rule this run submitted the md5-verified 09-15 queue at 15:45 UTC (10 min before the deadline). Session `-ee` stood down to avoid a race.
- Public: sub15e_emo-order2 281, sub17_pres 279, sub16e_emo-nbopt 278, sub17_nbw_base 278, sub15_nb_base 274. The three derived counts (279/278/274) matched exactly. No new upgrade: no time left to validate one.
- **Best stays sub18_nbw-lag 287 = 0.83918.** Final selection (the user's action in the Submissions tab): B = sub18_nbw-lag + sub17t_pres-notime (recommended), or A = sub18_nbw-lag + sub17_nbw-emo13e.
- Correction (16:15 UTC): the Kaggle API gives the deadline as 2026-09-15 23:59 UTC, not 15:55 UTC. The quota resets only at 00:00 UTC, so no more submissions are possible. Final pair: session 76 advises A (sub18_nbw-lag + sub17_nbw-emo13e, the same as auto-select); B (+ sub17t_pres-notime) only helps if the organisers reject individual timestamp-based submissions rather than the team. The user decides.
