#!/usr/bin/env bash
# reevaluate_old_runs.sh
#
# Re-trains the best config per method to get weight_l2_dist, forget_NLL,
# and retain_NLL logged to W&B. Weights are deleted after eval (--no-save).
#
# Why re-train?  These runs predate the logging of those metrics, so their
# W&B summaries are missing them.  Re-training with FORCE=1 bypasses the
# idempotency check and creates a new run; NO_SAVE=1 cleans up weights
# after evaluation to save disk space.
#
# After all runs complete, analyze_runs.py is re-run to regenerate
# best_unlearning_models.md with the newly logged values.
#
# Usage (from repo root):
#   ./unlearn/analysis/reevaluate_old_runs.sh

set -euo pipefail
cd "$(dirname "$0")/../.."    # repo root

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export NO_SAVE=1     # delete weights after eval — we only need the W&B metrics
export FORCE=1       # bypass W&B idempotency check; these runs exist but lack L2

echo "=========================================================="
echo "Re-training best configs to log L2 dist + NLL (NO_SAVE=1)"
echo "Model: $BASE"
echo "=========================================================="

# ------------------------------------------------------------------
# GA — ep3_lr3e-05_bs32_rw5.0_ml2048
# ------------------------------------------------------------------
echo -e "\n>>> [ga] ep=3, lr=3e-05, bs=32, rw=5.0, ml=2048 <<<"
EPOCHS=3 LR=3e-05 BATCH_SIZE=32 RETAIN_WEIGHT=5.0 MAX_LENGTH=2048 \
    ./unlearn/run_unlearn.sh ga

# ------------------------------------------------------------------
# grad_diff — ep3_lr4e-05_bs32_fw1.0_mle512_mli2048
# ------------------------------------------------------------------
echo -e "\n>>> [grad_diff] ep=3, lr=4e-05, bs=32, fw=1.0, ml=512, lines=2048 <<<"
EPOCHS=3 LR=4e-05 BATCH_SIZE=32 FORGET_WEIGHT=1.0 MAX_LENGTH=512 MAX_LINES=2048 \
    ./unlearn/run_unlearn.sh grad_diff

# ------------------------------------------------------------------
# CB — ep2_lr1e-05_bs32_a1000.0_sc5.0_ly11-12-13_ml2048
# ------------------------------------------------------------------
echo -e "\n>>> [cb] ep=2, lr=1e-05, bs=32, a=1000, sc=5, ly=11-12-13, ml=2048 <<<"
EPOCHS=2 LR=1e-05 BATCH_SIZE=32 ALPHA=1000.0 STEERING_COEFF=5.0 \
    LAYER_ID="11,12,13" MAX_LENGTH=2048 \
    ./unlearn/run_unlearn.sh cb

# ------------------------------------------------------------------
# CB_LAT — ep3_lr1e-05_bs32_a1000.0_sc5.0_le0.1_ls5_ly11-12-13_ml2048
# (mirror CB with default LAT params)
# ------------------------------------------------------------------
echo -e "\n>>> [cb_lat] ep=3, lr=1e-05, bs=32, a=1000, sc=5, ly=11-12-13, ml=2048 <<<"
EPOCHS=3 LR=1e-05 BATCH_SIZE=32 ALPHA=1000.0 STEERING_COEFF=5.0 \
    LAYER_ID="11,12,13" MAX_LENGTH=2048 LAT_EPS=0.1 LAT_STEPS=5 \
    ./unlearn/run_unlearn.sh cb_lat

# ------------------------------------------------------------------
# NPO — ep3_lr4e-05_bs32_b0.01_rw1.0_ml2048
# ------------------------------------------------------------------
echo -e "\n>>> [npo] ep=3, lr=4e-05, bs=32, beta=0.01, rw=1.0, ml=2048 <<<"
EPOCHS=3 LR=4e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.0 MAX_LENGTH=2048 \
    ./unlearn/run_unlearn.sh npo

# ------------------------------------------------------------------
# SimNPO — ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048
# ------------------------------------------------------------------
echo -e "\n>>> [simnpo] ep=3, lr=3e-05, bs=32, beta=0.01, rw=1.0, ml=2048 <<<"
EPOCHS=3 LR=3e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.0 MAX_LENGTH=2048 \
    ./unlearn/run_unlearn.sh simnpo

# ------------------------------------------------------------------
# DPO — ep3_lr4e-05_bs32_b0.01_mle512_mli2048
# ------------------------------------------------------------------
echo -e "\n>>> [dpo] ep=3, lr=4e-05, bs=32, beta=0.01, ml=512, lines=2048 <<<"
EPOCHS=3 LR=4e-05 BATCH_SIZE=32 BETA=0.01 MAX_LENGTH=512 MAX_LINES=2048 \
    ./unlearn/run_unlearn.sh dpo

# ------------------------------------------------------------------
# wt_dist — ep3_lr3e-05_bs32_wn0.01_ml2048
# ------------------------------------------------------------------
echo -e "\n>>> [wt_dist] ep=3, lr=3e-05, bs=32, noise=0.01, ml=2048 <<<"
EPOCHS=3 LR=3e-05 BATCH_SIZE=32 WT_NOISE_STD=0.01 MAX_LENGTH=2048 \
    ./unlearn/run_unlearn.sh wt_dist

# ------------------------------------------------------------------
# TAR — ta1.0_tlr1e-05_tep1_ml1024
# ------------------------------------------------------------------
echo -e "\n>>> [tar] alpha=1.0, lr=1e-05, ep=1, ml=1024 <<<"
TAR_ALPHA=1.0 TAR_LR=1e-05 TAR_EPOCHS=1 MAX_LENGTH=1024 \
    ./unlearn/run_unlearn.sh tar

# ------------------------------------------------------------------
# Regenerate best_unlearning_models.md
# ------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "Re-generating best_unlearning_models.md ..."
echo "=========================================================="
PYTHON="${PYTHON:-uv run --script}"
$PYTHON unlearn/analysis/analyze_runs.py

echo ""
echo "=========================================================="
echo "Done. best_unlearning_models.md updated with L2 dist."
echo "=========================================================="
