# Daily upgrade plan — CUHK-X Large Model Track

Deadline **2026-09-15 23:59 UTC (05:29 IST 09-16)** (Kaggle API, checked 09-15; the earlier 15:55 UTC was wrong). Standing instruction from the user
(2026-09-11): submit **one validated upgrade per day** until the deadline and explain the
strategy very briefly. **Read `FEEDBACK.md` first — the user's feedback overrides this
plan.** Full technical record, including dead ends not to re-try: `README.md`.

**Validation rule.** A change ships only if it improves the score on held-out subjects:
tuned on one half of the subjects, scored on the other, positive on **both** halves (see
`src/vlm_diag.py`, `src/vlm_harn_only.py`). The public LB is 342 questions (±0.025 noise)
and cannot referee small changes. Never submit an unvalidated variant — if it gets lucky
on the public half, Kaggle may auto-select it for the private leaderboard.

**Exception — the blend track (user decision 09-11, `FEEDBACK.md`).** Blends that swap
**whole categories** between our model and the public 0.77777 file are refereed by the public
LB. Public scores are exact counts out of 342, so the effect of a swap can often be derived
exactly from files already scored. Never pick answers question by question.

**Leak guard.** Every build is diffed against the previous best; changes must be confined
to the categories the validated change targets (`finish_vlm.sh` shows the pattern).

## Status

| Day (IST) | Upgrade | Status |
|---|---|---|
| 09-04 | sub02–sub06: structure → motion → CLIP → zero-shot | done, best public 0.70467 |
| 09-11 | sub07: VLM for single\|HARn only (w=5) | **done — public 0.71052** (best) |
| 09-11 (2nd) | sub08: VLM on 4 held-out-validated groups (all 1333 training clips VLM-scored) | **done 09-11 10:05 IST — public 0.72222** (247/342). Submitted a day early at the user's request (5 submissions/day allowed). **Do NOT submit it again.** |
| 09-11 | Blend tests with the public 0.77777 file (user decision, `FEEDBACK.md`) | **done** — see "Blend track" below: their answers beat ours in every category group on the public half |
| 09-11 (3rd) | sub09: precedence prior for `sequence` + lower multi threshold (`src/struct_validate.py` → `src/best_struct_W.json`) | **done 09-11 11:44 IST — public 0.74561** (255/342, +8 over sub08). Held-out gain +0.0191 / +0.0069 overall; 26 answers changed vs sub08 (23 sequence, 3 multi). Recurrence as an explicit term validated at weight 0 (multi) and was rejected (single). **Do NOT submit it again.** |
| 09-11 (4th) | blend C = their file + our `sequence` answers | **done — public 0.77777** (same as their file; the 9 differing answers net zero on the public half). Current best blend candidate |
| 09-12 | Five submissions, in this order. They supersede blends D, E2 and F2; don't submit those. Public counts are out of 342; sub09 = 255.<br>**1st: `submissions/sub11g3b_sub09-plus-G3.csv`** (our answers only) = sub09 with sub11's combination and single-HAU answers (13 differ). Its count − 255 = **ΔG3**.<br>**2nd: `submissions/sub10_route.csv`** (pure; 10 multi answers differ from sub09). Its count − 255 = **ΔM**. The blend takes its multi from sub10 (per-category CV favours it over sub11's).<br>**3rd: `submissions/sub11_routec.csv`** (pure: + combination routing, held-out +0.0052 / +0.0079), count **C11**.<br>**4th: `submissions/sub12_nv.csv`** (pure) = sub11 + the organisers' non-visual evidence. It adds IMU random-forest emotion (held-out +0.0052 / +0.0130) and Skeleton HARn trained on all units (held-out +0.0109 / +0.0087). 77 answers differ from sub11: 66 emotion, 5 single-HARn, 6 objects. Only G1 answers change, so its count − C11 = **ΔG1**. `submissions/sub12e_nv-emotion.csv` (emotion only) is superseded; don't submit it.<br>**5th: the best of four candidates**, all pre-built, with public counts known exactly:<br>• E `submissions/blendE_theirs-but-our-seq-multi2.csv` = 268 + ΔM<br>• F3 `submissions/blendF3_theirs-G1_ours-rest.csv` = E + ΔG3 − 1<br>• EG1 `submissions/blendEG1_theirs-G3_ours-rest.csv` = E + ΔG1 − 12<br>• all-ours `submissions/ours_bestof_sub12.csv` = E + ΔG3 − 1 + ΔG1 − 12<br>Submit the highest. On a tie, prefer the one with fewer of our answers swapped in: E, then F3, then EG1. Record ΔG3, ΔM and ΔG1 in the ledger.<br>**All eight files are pre-built and verified. Submit them exactly as they are and never rebuild them.** sub12 needs the `CUHKX_*` switches, and a plain rebuild would silently drop the IMU/Skeleton evidence. Check each md5 before submitting: `sub11g3b_sub09-plus-G3.csv` 4a32ee0d71b4; `sub10_route.csv` b4b4d272e512; `sub11_routec.csv` 1ce59de48328; `sub12_nv.csv` 201eba66e9ec; `blendE_theirs-but-our-seq-multi2.csv` 954aa85417bd; `blendF3_theirs-G1_ours-rest.csv` ed150c8e5f0f; `blendEG1_theirs-G3_ours-rest.csv` 5b5ad7a52e03; `ours_bestof_sub12.csv` 74f631b9f23f.<br>Option-order TTA (sub13) is validated but blocked on the GPU quota: see the next cell. | **Done 09-12, 21:57-22:10 IST.** The 06:16 scheduled run stopped 19 s in at a tool-permission prompt and submitted nothing; the five files went in that evening. Counts: hybrid 258, sub10 257, sub11 262, sub12 **271 = 0.79239** (our own model passes their 266), blend F2 **274 = 0.80116 confirmed**, exactly as predicted (submitted instead of F3: the public count puts sub11's multi 2 ahead of sub10's). Rank 32 of 201, top 16%; the top-15% bar has moved to 0.81286, so 4 more questions are needed. **TTA:** validated on the training side (+0.0193 / +0.0095 held-out, against +0.0162 / +0.0085 without), but the test kernel was **NOT** pushed because the account's weekly GPU quota (shared with the user's notebooks) ran out 09-11 ~20:30 IST. **Do not push `cuhk-x-vlm-tta-test` without asking the user.** Never submit `submissions/PARTIAL_sub12_tta_no-test-tta_DO-NOT-SUBMIT.csv`: it applies TTA-tuned weights to plain test scores. |
| 09-13 | Split G1, starting from blend F2 (274). **Both files are pre-built and leak-guarded (09-12 23:58 IST); submit them as they are, never rebuild.**<br>• **S1** `submissions/blendS1_theirs-HARn-obj_ours-emotion-rest.csv` (md5 f48f91e261bd) = F2 with our 44 differing `emotion` answers; nothing else differs from F2.<br>• **S2** `submissions/blendS2_theirs-emotion-obj_ours-HARn-rest.csv` (md5 1e7a78b9aa20) = F2 with our 9 differing `single\|HARn` answers; nothing else differs.<br>**Objects are identical to their file, so S1 + S2 = 2·274 − 3 = 545 exactly.** Submit **S1 first** and derive S2 = 545 − S1. Submit S2 only if that derived count is > 274, to put the file on the board for selection. If both are ≤ 274, F2 stays the final blend, and the day's other slots go to the 09-14 work. | **Done 09-13 11:08 IST.** S1 = **0.79239 (271)**, so our emotion is **−3** vs theirs, and S2 = 545 − 271 = **274** (single\|HARn ties). S2 was **not submitted**: it ties F2, and a same-score file adds nothing to the selection. **F2 (274) stays the final blend.** Remaining slots went to the presence work (`src/pres_nv_validate.py`; see the 09-14 row). |
| 09-14 | **DONE 05:31 IST, all 5 used (API):** (scheduled run 07:45 IST: quota confirmed used; only logged; 09-15 queue md5s verified) sub18_nbw-lag **287 = 0.83918 (best)**, sub17_nbw-emo13e 286, sub17_nbw 285, sub16_nb-emo13e 282, sub17t_pres-notime 272 (timestamp-free). The scheduled run only logs.<br>**OWNER: the interactive session `cuhk-x-competition-large-model-3e` submits all 5 at 05:31 IST.** Scheduled run: if the Submissions list already holds these files, only log. If some are missing and that session is alive (ListAgents), message it and don't submit. Only if it is gone, submit the missing ones in order.<br>Exact command: section **09-14 submit command** at the end of this file.<br>**Queue for 09-14: use ALL 5 slots (FEEDBACK). State at 09-13 23:00 IST:** all 5 of 09-13's slots are used (S1 271, sub13e **279**, sub14e 278, sub15e_emo-norm 277, sub15_nb = sub14e + neighbour pooling = **281 = 0.82163, best**; read counts with the Kaggle API, not a regex). Submit in this order, skipping anything already on the Submissions list, and read every score via `KaggleApi().competition_submissions(...)`. **Updated 09-13 23:45 IST:** wide-grid neighbour pooling validated +0.0097 / +0.0131, beating the narrow +0.0045 / +0.0025, so it replaces it:<br>1. **`submissions/sub17_nbw-emo13e.csv`** (md5 fea8881e2ffd; pre-built and verified: vs sub13e only 16 action answers, vs sub17_nbw only 17 emotion answers) = wide neighbour pooling + sub13e's emotion.<br>2. **`submissions/sub17_nbw.csv`** (md5 f91de8f65162; vs sub14e only 16 action answers: single|HAU 8, multi 4, combination 4; vs sub15_nb 8) = wide neighbour pooling + sub14e's emotion. Its count − 278 isolates the wide pooling. #1 is then exactly 279 + that gain, but both must be on the board to be selectable.<br>3. **`submissions/sub18_nbw-lag.csv`** (md5 86e216768efc) = sub17_nbw with the higher-order emotion decoder (`order2+LR36`, +0.0049 / +0.0050), which changes 4 emotion answers. It is the best session model by cross-validation. Swapped in at 02:50 IST for `sub17_pres`: the joint check showed presence doesn't add on top of pooling (−0.0006 / −0.0010), and item 5's count already measures presence. sub17_pres becomes a 09-15 spare.<br>4. `submissions/sub16_nb-emo13e.csv` (md5 5dfc75a7ae3f): the narrow-pooling reference, known to be 282.<br>5. **`submissions/sub17t_pres-notime.csv`** (md5 de2e915a4a1f) = sub12 + presence (IMU + Skeleton). This is the **timestamp-free fallback**: no session normalisation, session decoding or neighbour pooling (swapped in 09-14 00:55 IST after the leakage risk in NOTES). Its count = 271 + (count(sub17_pres) − 278) exactly. It is submitted so the user can select it. The replaced emotion-variant files stay available.<br>Final selection (09-15): the top two public of {sub17_nbw-emo13e, sub17_nbw, sub16_nb-emo13e (282), sub15_nb (281)}. Both include neighbour pooling; they differ only in the emotion decoder (sub13e's vs sub14e's). Update if presence validates and scores higher. Non-visual features for the categories they never touched. **`sequence` done 09-12 night: rejected** (Skeleton slot features, held-out −0.0011 / +0.0007; see README dead ends). **Started 09-13:** (a) IMU/Skeleton in the HAU presence model, `python src/pres_nv_validate.py --pres=skel` (`CUHKX_PRES`, `CUHKX_C_PRES`); (b) time-neighbour option pooling, `python src/nb_validate.py` (`CUHKX_NB=1`, weights `w_nb_comb` / `w_nb_bel`). Both are off by default (sub12 rebuild byte-identical) and write `src/best_nv_pres_W_skel_report.json` / `src/best_nb_W_report.json`. **Build if accepted:** sub12's env + the new switch, `make_submission3.py --clip --weights=<new weights>`, leak-guard vs sub12 to combination / single\|HAU / multi, then `python src/build_session.py OUT+emo.csv --emo=norm+sess --w-emo=5 --tau=0.5 --order2=-4 --nbopt=0.25 --base-sub=OUT.csv` to add sub14e's emotion (`submissions/sub14e_emo-order2nb.csv`, md5 9f597d881241; public **0.81286 = 278**). Count − 278 isolates the change. If both presence and neighbour pooling pass, build only the one with the larger minimum held-out gain: they share beliefs and thr_m and were not validated together. Emotion (sub13e) is owned by the interactive session. | The `sequence` slot model (`src/seq_model.py`, motion segments only) and the HAU presence model (`src/motion_model.py`, motion+CLIP) have never seen IMU or Skeleton. IMU change points and joint velocities per time slot are the obvious evidence for temporal order; presence feeds `multi` and `combination`, where we are already ahead. Build per-time-slot non-visual features, validate on held-out subjects like everything else, then measure each category against the ledger. |
| 09-15 | **Updated 09-14 00:40 IST. OWNER: the interactive session `cuhk-x-competition-large-model-3e` submits all 5 of 09-15's slots (quota resets 05:30 IST; deadline 21:25 IST).** Scheduled run: same fallback rule as 09-14.<br>**Default 09-15 queue** (set 09-14 06:00 IST; used unless the user approves a new idea). All five are validated parts, never on the board, and contain no blends: 1 `sub15e_emo-order2` (b8140d17bb3c), 2 `sub16e_emo-nbopt` (f06c2f102aeb), 3 `sub17_pres` (ff1d3dc392b9; count 279 derivable), 4 `sub17_nbw_base` (d2918a3b844b; sub12 + wide pooling, 278 derivable), 5 `sub15_nb_base` (0c370638b6a6; sub12 + narrow pooling, 274 derivable). They are only there to use the slots; the final pair (A/B/C below) does not depend on them. Exact command: section **09-15 submit command** at the end of this file.<br>**Leakage decision pending (user), see NOTES 09-14.** Session features use test skeleton-frame timestamps. The organisers treated those timestamps as a leak vector in the Small track (topic 714827). Default recommendation: select the best session model plus the best timestamp-free build (`sub17t_pres-notime`, or a validated timestamp-free successor). Kaggle scores the better of the two, so this costs little.<br>Queue, filled on 09-14 from two sources. (a) The best emotion variant by 09-14's public counts, stacked on the best action group. Categories are disjoint, so the stack's count is exact before it is submitted. (b) Newly validated upgrades: **`submissions/sub18_nbw-lag.csv`** (md5 86e216768efc; moved into 09-14 slot 3) = sub17_nbw with the higher-order emotion decoder (`order2+LR36`, +0.0049 / +0.0050), 4 emotion answers vs sub17_nbw; session family. No timestamp-free emotion upgrade: `src/emo_notime_validate.py` rejected whole-set, per-subject and per-session normalisation (decided per clip), so the fallback stays `sub17t_pres-notime`.<br>(c) Joint check presence + wide pooling (`src/nbpres_joint_validate.py`): **rejected** (−0.0006 / −0.0010), so there is no `sub19_nbpres`.<br>**Final selection (updated 09-14 05:40 IST; the user's timestamp decision):** A = sub18_nbw-lag (287) + sub17_nbw-emo13e (286). **B (recommended) = sub18_nbw-lag (287) + sub17t_pres-notime (272).** C = sub17t_pres-notime (272) + sub12_nv (271). Kaggle scores the better of the two picks. No blends: we lead every category, and top teams' code is re-run for reproducibility. | **Done 09-15 15:45 UTC (scheduled run; owner session gone):** all 5 submitted: sub15e_emo-order2 281, sub17_pres 279, sub16e_emo-nbopt 278, sub17_nbw_base 278, sub15_nb_base 274. Best stays sub18_nbw-lag 287. User must SELECT the final pair (B recommended). |

## Blend track (user decision 09-11 — `FEEDBACK.md`)

Reference: team Fususu's released submission `public_ref/fususu_lb0.77777.csv` (MD5
`8f997f26…`, public 0.77777 = 266/342). Per their notebook it comes from a structural graph
plus non-visual specialists: Skeleton for HARn actions/objects, IMU for HAU emotion. Their HAU
action evidence is structural only. `python src/blend_public.py OURS OUT --theirs=G1[,G2,G3]`
takes their answers for whole question-type groups: G1 = emotion|HAU, single|HARn,
object_interaction|HARn; G2 = multi|HAU, sequence|HAU; G3 = combination|HAU, single|HAU.

| Submission | Public | Their net gain, public half |
|---|---|---|
| sub08 (all ours) | 0.72222 (247) | — |
| blend A = theirs on G1 | 0.75730 (259) | G1 +12 |
| blend B = theirs on G1+G2 | 0.77485 (265) | G2 +6 |
| their file (all theirs) | 0.77777 (266) | G3 +1 (noise) |

**Their file beats ours in every group, so it is the base now.** Beating 0.77777 means
our answers must beat theirs in some whole category. Each candidate is validated on held-out
subjects first, then refereed on the public LB as a whole-category swap into their file:
1. `sequence`: pairwise-precedence prior plus our slot model. **Done 09-11, goes in the
   final blend.** Held-out +0.31 / +0.10 in sequence exact accuracy. Their answers agree
   with it 77% of the time, and on CV disagreements with precedence-only ours wins 32 to 11
   (`src/seq_disagree.py`). Candidate file:
   `submissions/blendC_theirs-but-our-seq.csv` (9 answers differ from their file). It was
   submitted 09-11 with the day's last slot and scored 0.77777, the same as their file.
   Nine questions are below the public LB's resolution, so the CV evidence decides. Blend C
   is the current best candidate and the base for further swaps.
2. `multi`: **ours is better; use it (blend D).** The public scores are exact counts out of
   342, so they can be combined arithmetically: (B − A) = (their seq+multi) − (our old
   seq+multi) = +6; (sub09 − sub08) = (our new seq+multi) − (our old) = +8; (C − their
   file) = our new seq − their seq = 0. So their multi is 2 questions *worse* than sub09's on
   the public half. **Blend D** = their file + our sequence + our multi
   (`submissions/blendD_theirs-but-our-seq-multi.csv`, 49 answers differ from their file:
   40 multi, 9 sequence). Its public score is known exactly: 268/342 = **0.78363**, above the
   31-team tie at 0.77777. The CV logic agrees: options outside the true combination are
   right only 9–30% of the time, and their answers include more such options than ours.
   **Superseded before submission** by blends E / F3, which add the single-answer routing
   for multi and possibly our G3. The 09-12 row is authoritative; don't submit blend D.
3. `emotion` / `single|HARn`: our own IMU / Skeleton models. **The zip arrived 09-11 20:40 IST**
   (Drive mirror, try 12). IMU random-forest emotion was accepted and is in sub12e. The Skeleton
   HARn head-to-head was still running at ~22:30. When the zip is in `data/`, first run `python src/lmt_inspect.py` (read-only). It
   prints the structure and maps files onto our clip ids. Then follow their recipe:
   - IMU band power / entropy / dominant frequency / autocorrelation → RandomForest for
     emotion.
   - Skeleton centred joints, temporal bins, velocities → HARn action + object.

   Fuse each into our pipeline as another evidence term, validate on held-out subjects, and
   read the change against the G1 row of the ledger. The row is +12 for them, so we need
   more than +12 on the public half to take G1.

**Public-half ledger** (exact counts, their answers − ours; 09-11). Each question counts
independently, so the ledger adds up exactly.

| Group | Their − ours | Measured against |
|---|---|---|
| G1 (emotion, single\|HARn, object_interaction) | **+3** | sub12 (was +12 vs sub08; ΔG1 = +9 measured 09-12). **Split 09-13 (S1 = 271): emotion +3, single\|HARn 0, objects 0** (objects match theirs exactly). **After sub13e (279, 09-13): emotion −5, so their file is ahead in NO category; the best blend = our pure sub13e** |
| sequence | 0 | sub09 onwards, unchanged |
| multi | **−6** | sub11 (ΔM = +2 for sub10 and +4 for sub11, both over sub09) |
| G3 (combination, single\|HAU) | **−2** | sub11 (ΔG3 = +3 measured 09-12) |

Any later pure submission whose leak guard confines it to one group updates that group's row
for free: new row = old row − (new pure − previous pure). Keep TTA or IMU/Skeleton builds
**separable by group**, e.g. one build per group, so every change can be read off this way.
Take our answers for a group only when its row is ≤ 0 and CV agrees.

Final selection (09-15): the best blend plus our best pure model as the hedge.

## 09-12 — full training coverage (sub08)

Why: tuning all nine VLM weights failed the nested check at n=320 clips (−0.003 held out),
while in-sample it looked like +0.017. The remedy for "too little data to validate" is more
data: score the other 1013 training clips.

- Kernels `santanubanerjee9/cuhk-x-vlm-train-rest-0` and `-rest-1` (folders
  `kaggle_notebook_rest0/1`) each score half of the clips outside the tuning sample.
  If rest-1 isn't running yet: `kaggle kernels push -p kaggle_notebook_rest1 --accelerator NvidiaTeslaT4`
  (max 2 GPU sessions; never stop the user's other notebooks).
- **Everything after the kernels is one command: `bash day2_full_vlm.sh`** (written and
  dry-run on 09-11). It fetches whichever shards are COMPLETE (use what's there; don't wait
  more than ~2 h), merges them with the original 320 clips into `vlm/vlm_scores_train.csv`,
  runs `src/vlm_perweight.py` — each VLM weight (or coupled pair) tuned *alone* on one half
  of the subjects and scored on the other, accepted only if positive on **both** halves, with
  separate `combination` weights for clips with / without a `sequence` question — then
  writes `src/best_vlm_W.json` + `src/vlm_perweight_report.json`, builds
  `submissions/sub08_vlm_perweight.csv`, and leak-guards it against sub07 using the accepted
  groups' categories.
- Submit sub08 only if it differs from sub07 **and** the guard passed. If nothing new was
  accepted, sub08 == sub07: don't submit; log that the larger sample confirmed HARn-only and
  move to the 09-13 candidates.

## Candidates for 09-13 / 09-14 (pick by what validates; feedback may reorder)

1. **Option-order TTA for the VLM** — re-score letter questions with the options cyclically
   shifted and average per-option log-probs, so every option appears in every position once
   and letter-position bias cancels.
   - *Evidence (09-11):* the VLM has a measurable position bias — on `combination` it
     over-picks D (0.29 vs 0.22 true) and its accuracy ranges 0.54–0.76 by which letter holds
     the answer; on `emotion` it over-picks C (0.33 vs 0.23). Test picks show the same skews.
   - *Tooling, ready and tested:* notebook knob `TTA = True` (scores shifts 1..m−1 only;
     rows keyed by the ORIGINAL letter as kinds `letter_s1`...; shift 0 = the existing run);
     `python src/tta_merge.py <orig.csv> <tta.csv> [...] -o <out.csv>` (unit-tested: recovers
     the right answer under a synthetic bias; identity when there are no shifts); smoke kernel
     `santanubanerjee9/cuhk-x-vlm-tta-smoke` — **verified on Kaggle 09-11**: 0 errors, and
     every confident question picks the same ORIGINAL option under all shifts while only the
     near-ties flip with position (the correct mapping — and exactly the bias TTA averages
     out). 3 of 8 smoke questions changed answer after merging, all near-ties.
   - *Scope — decided 09-11 08:00:* the full-coverage run accepted HAU letter groups
     (`single_hau`, `comb_noseq`, `emotion`), so TTA covers every letter category, split so no
     question is scored twice: `cuhk-x-vlm-tta-harn` (HARn letter questions, test + all training
     clips) and `cuhk-x-vlm-tta-test` + `cuhk-x-vlm-tta-train-0/1` (HAU letter questions only).
     A watcher runs **one of these at a time** — the user's own notebooks share the account's
     two GPU slots — in the order harn → train-0 → train-1 → test, ~9.5 h in all. Check with
     `kaggle kernels status santanubanerjee9/<slug>`; **never re-push a kernel that has already
     run** (a re-push starts a new version), and never push onto `cuhk-x-vlm-evidence`.
   - *Validate — one command:* `bash tta_eval.sh [BASELINE_SUB] [OUT_SUB]` (after
     `day2_full_vlm.sh`). It fetches whichever TTA kernels have COMPLETEd — questions whose
     kernel hasn't finished keep their plain scores — merges all shifts into `vlm_tta/` (never
     touching `vlm/`), runs `vlm_perweight.py --vlm-dir=vlm_tta --out-tag=tta`, prints TTA vs
     non-TTA held-out gains per group, builds the candidate with
     `--vlm-dir=vlm_tta --weights=src/best_vlm_W_tta.json`, and leak-guards it against the
     baseline. **Always pass `--out-tag` when validating alternative scores** — without it
     `vlm_perweight.py` overwrites `src/best_vlm_W.json` (which `make_submission3.py` uses by
     default) and the very report TTA must be compared against. Ship only if TTA beats
     non-TTA on held-out subjects.
   - *Early read, HARn only (09-11 09:55):* **no gain** — `harn` +0.0090 / +0.0035 →
     +0.0090 / +0.0022, `object_interaction` 0 / 0 → 0 / −0.0036; only 2 test answers change
     at the validated weight. Not shipped (candidate renamed
     `submissions/REJECTED_sub_tta_harn_partial_no_heldout_gain.csv`; report kept as
     `src/vlm_perweight_report_tta_harnonly.json`). HAU is the real test — its position bias
     was far larger — but expectations are lower now. If HAU TTA also fails, 09-13 falls back
     to candidate 2 or 3, or to "no validated upgrade today".
2. **Denser frames for `sequence`** — 16–20 frames with explicit frame indices in the prompt;
   temporal order is where 12 frames is thinnest (VLM alone: 0.119 exact).
3. **Per-stratum VLM weights** wherever the 09-12 analysis shows opposite effects by stratum.

## Final day 09-15

Submit the best validated configuration if it isn't already the latest. Then tell the user:
the private LB is scored on the submissions they **select** in Kaggle's Submissions tab (up
to 2); unselected, Kaggle auto-picks by public score. Recommend the best validated build
plus `sub06_zeroshot.csv` as a hedge.

## 09-14 submit command

Run once the quota has reset (05:30 IST), from the project folder. It checks each md5 prefix, skips files already on
the Submissions list, stops at 5 per UTC day, then prints the public scores read from the Kaggle API.

```bash
python src/submit_queue.py "submissions/sub17_nbw-emo13e.csv|fea8881e2ffd|sub17_nbw-emo13e: wide neighbour option pooling + sub13e session emotion decoder" "submissions/sub17_nbw.csv|f91de8f65162|sub17_nbw: wide neighbour option pooling + sub14e order-2 emotion decoder" "submissions/sub18_nbw-lag.csv|86e216768efc|sub18_nbw-lag: sub17_nbw + higher-order (lags 3-6) emotion decoder" "submissions/sub16_nb-emo13e.csv|5dfc75a7ae3f|sub16_nb-emo13e: narrow neighbour pooling + sub13e emotion (reference)" "submissions/sub17t_pres-notime.csv|de2e915a4a1f|sub17t_pres-notime: timestamp-free fallback = sub12 + IMU/Skeleton presence"
```

## 09-15 submit command

Final day. Run once the quota has reset (05:30 IST 09-15), before the 21:25 IST deadline, from the project folder.
If the user approved a new idea, its built file replaces the last spares. Then remind the user to SELECT the final
pair in the Submissions tab (options A/B/C in the 09-15 row).

```bash
python src/submit_queue.py "submissions/sub15e_emo-order2.csv|b8140d17bb3c|sub15e_emo-order2: sub12 + per-session IMU emotion, order-2 decoder (emotion only)" "submissions/sub16e_emo-nbopt.csv|f06c2f102aeb|sub16e_emo-nbopt: sub12 + per-session IMU emotion, neighbour-option term (emotion only)" "submissions/sub17_pres.csv|ff1d3dc392b9|sub17_pres: sub14e + IMU/Skeleton presence" "submissions/sub17_nbw_base.csv|d2918a3b844b|sub17_nbw_base: sub12 + wide neighbour option pooling" "submissions/sub15_nb_base.csv|0c370638b6a6|sub15_nb_base: sub12 + narrow neighbour option pooling"
```
