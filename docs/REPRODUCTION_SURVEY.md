# Answers given in the organisers' Reproduction Environment Survey

Submitted 2026-09-18 for the Large Model Track ([discussion
712213](https://www.kaggle.com/competitions/cuhk-x-competition-large-model-track/discussion/712213)).
Recorded here so this repository and the survey say the same thing, and so the numbers can be checked
against the code.

## B. The model

| | Answer |
|---|---|
| B1 Type of submission | Off-the-shelf open-weight model, no fine-tuning |
| B2 Base model | `Qwen/Qwen2.5-VL-7B-Instruct` (7B), fp16, not fine-tuned. Also `openai/clip-vit-base-patch32` for frame embeddings. Our own components are scikit-learn models (random forest, logistic regression) trained from scratch on the provided training data and refit at build time, so there is no neural checkpoint |
| B3 Peak GPU memory, one inference process | 16–24 GB |
| B4 GPUs per inference run | 2 |
| B5 Is parallelism required | A single larger GPU would work |
| B6 Checkpoint size | <1 GB (fitted-weight JSONs and cached VLM log-probabilities only) |
| B7–B8 Closed-source API | Not applicable — none used |

B4 is 2 because that is what the leaderboard runs used, not because the code demands it: `load_model()` in
`notebooks/kaggle_notebook/vlm_infer.py` picks fp16 when total GPU memory is ≥ 30 GB and 4-bit nf4 below
that, so a single 16 GB card would silently change the numerics.

## C. Running the code

| | Answer |
|---|---|
| C1 Container | No — `requirements.txt` (CPU) and `requirements-gpu.txt` (VLM stage) |
| C2 PyTorch | 2.10.0 |
| C3 CUDA | 12.6 or newer (ours: 12.8) |
| C4 Dependencies from their list | bitsandbytes only, and only for the sub-30 GB 4-bit fallback |
| C5 Development hardware | Kaggle notebooks with 2× NVIDIA Tesla T4 (16 GB each, 31 GB total), fp16. Everything after the VLM stage: a CPU-only laptop, Intel i3-10110U, 4 threads, 8 GB RAM |

No FlashAttention, xformers, DeepSpeed, vLLM/SGLang, Triton kernels or Apex, and nothing is compiled at
install time, so the code is not tied to a GPU generation.

## D. Runtime

| | Answer |
|---|---|
| D1 One complete inference run over the full test set | 2–4 hours |
| D2 Peak host RAM | ≤32 GB |

Measured, not estimated: the VLM stage scored all 208 test clips / 2,794 rows in **140.2 minutes** on the
2× T4, at 6.24 s per forward pass over 1,348 passes (`vlm_test_run/run_info.json` from the original run).
Rebuilding the prediction file from the VLM scores committed in `vlm/` takes **113 s on 4 CPU cores** and
reproduces both md5s exactly.

## E and F. Transfers, time zone, and notes

Time zone UTC+5:30. Google Drive, Hugging Face Hub and plain HTTPS links all work for large transfers;
Baidu Netdisk is not usable from here.

The free-text note told the organisers: this repository is the deliverable; there is no checkpoint to
transfer; `Qwen/Qwen2.5-VL-7B-Instruct` and `openai/clip-vit-base-patch32` should be pre-downloaded if the
verification machine is offline; at least 30 GB of total GPU memory (or one 24 GB card) is needed for the
fp16 path that produced our scores; the dataset and everything derived from it is not redistributed here,
per the CUHK-X License v2.0; builds are deterministic, with md5s given; and — stated plainly — our two
selected submissions use the recording times embedded in the Skeleton frame filenames as evidence, with a
variant that uses none of it included so the effect can be measured. See
[`DECISIONS.md`](DECISIONS.md#2-recording-session-structure-and-the-disclosure-that-goes-with-it).
