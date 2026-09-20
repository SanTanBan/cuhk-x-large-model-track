#!/usr/bin/env bash
# CUHK-X Challenge 2026, Large Model Track - single entry point (team InSociEUP).
#
#   bash inference.sh
#
# Rebuilds our selected Kaggle submission and checks it against the file we submitted
# (md5 86e216768efc1198bdd11857e916808d). Output goes to work/ so nothing already here is overwritten.
#
# Expects, relative to this directory:
#   data/test_qa.csv, data/training_qa.csv and the competition media (HAU/, HARn/, large_model_track_test/)
#   data/LMT_IMU_Radar_Skeleton.zip          the organisers' non-visual supplement
#   vlm/vlm_scores_test.csv, vlm_scores_train.csv
#
# There is no separate training stage: every model (option-text priors, action co-occurrence, the motion and
# CLIP models, the IMU random forest for emotion, the Skeleton classifier for HARn) is fitted from
# data/training_qa.csv inside step 1, each run. The only persisted numbers are the fusion weights in
# src/best_*.json, produced by the validators in src/ (nb_validate_wide.py, emo_fast_validate3.py,
# pres_nv_validate.py, vlm_perweight.py), which tune on one half of the training subjects and score on the
# other.
#
# The VLM stage is a separate GPU job and is NOT run here: notebooks/kaggle_notebook/vlm_infer.ipynb scores
# every option letter with Qwen2.5-VL-7B-Instruct on Kaggle (2x T4, fp16). Its outputs are the two CSVs in
# vlm/, which this script consumes. To re-run it yourself, execute that notebook on a GPU machine and point
# VLM_DIR at the directory holding the fresh vlm_scores_*.csv:
#
#   VLM_DIR=/path/to/fresh/scores bash inference.sh
#
# Runtime: about 2 minutes on 4 CPU cores when the feature tables already exist, plus roughly an hour the
# first time if they must be built from the raw media. No GPU is needed for this script.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
VLM_DIR="${VLM_DIR:-vlm}"
WORK="${WORK:-work}"
BASE="${BASE:-$WORK/step1_action.csv}"
OUT="${OUT:-$WORK/sub18_nbw-lag.csv}"
EXPECT_MD5="86e216768efc1198bdd11857e916808d"
REFERENCE="submissions/sub18_nbw-lag.csv"

say () { printf '\n== %s\n' "$*"; }
md5 () { "$PY" -c "import hashlib,sys;print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

say "checking inputs"
for f in data/test_qa.csv data/training_qa.csv data/LMT_IMU_Radar_Skeleton.zip; do
    [ -f "$f" ] || { echo "MISSING: $f  (see 'Reproducing' in README.md)"; exit 1; }
done
for f in vlm_scores_test.csv vlm_scores_train.csv; do
    [ -f "$VLM_DIR/$f" ] || { echo "MISSING: $VLM_DIR/$f  (run the VLM notebook, or use the copies in vlm/)"; exit 1; }
done
mkdir -p "$WORK" features
echo "inputs OK"

# Feature tables. Each line is skipped when its output is already present, so a second run costs nothing.
say "features (skipped where already built)"
[ -f features/motion_test.csv  ] || "$PY" src/motion.py both          # frame-difference motion, Depth stream
[ -f features/motion2_test.csv ] || "$PY" src/motion2.py both         # periodicity, motion centroid, posture
[ -f features/seg_test.csv     ] || "$PY" src/segments.py both        # per-time-slot descriptors for temporal order
[ -f features/clip_test.npz    ] || "$PY" src/clip_feats.py both      # CLIP ViT-B/32 embeddings of sampled IR frames
[ -f features/imu_feats.csv    ] || "$PY" src/nonvisual_feats.py      # IMU + Skeleton features from the supplement
[ -f features/unit_times.csv   ] || "$PY" src/unit_times.py           # recording timeline from Skeleton frame names
echo "features ready"

# Step 1: fit everything from the training questions, fuse the VLM evidence, answer the action questions.
# The switches are the build environment recorded in src/best_nb_wide_W_config.json.
say "step 1/2: fit models, fuse VLM evidence, decode action questions"
CUHKX_EMO=imu CUHKX_CLF_EMO=rf CUHKX_HARN=skel CUHKX_C_HARN=1 CUHKX_HARN_UNITS=1 CUHKX_NB=1 \
    "$PY" src/make_submission3.py "$BASE" --clip --weights=src/best_nb_wide_W.json

# Step 2: emotion, decoded along each recording session.
say "step 2/2: emotion decoded along recording sessions"
"$PY" src/build_session.py "$OUT" --emo=norm+sess --w-emo=5 --tau=0.5 --nbopt=0.25 \
    --lagw=src/emo_fast3_report.json:order2+LR36 --base-sub="$BASE"

say "result"
GOT=$(md5 "$OUT")
echo "$OUT"
echo "  md5 $GOT"
if [ "$GOT" = "$EXPECT_MD5" ]; then
    echo "  MATCH - byte-identical to the submission we selected on Kaggle (public 0.83918, private 0.79705)"
else
    echo "  DIFFERS from the expected $EXPECT_MD5"
    [ -f "$REFERENCE" ] && "$PY" - "$OUT" "$REFERENCE" <<'PY'
import sys, pandas as pd
a, b = (pd.read_csv(f, dtype=str).set_index('qa_id').prediction for f in sys.argv[1:3])
d = a.compare(b.reindex(a.index))
print('  %d of %d answers differ from %s' % (len(d), len(a), sys.argv[2]))
PY
fi
