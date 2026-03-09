#!/usr/bin/env bash
# KL_best.sh — Run, save, evaluate, and upload to HuggingFace the best
# config per method from the cross-method comparison table.
#
# Best configs (from cross-method summary):
#   CB        LR=1.3e-05, BS=16, Ly=13-14-15
#   CB_LAT    identical to CB  (same LR/BS/Ly, default LAT params)
#   GA        LR=2e-05, RW=5.0
#   grad_diff LR=1e-05, FW=1.0
#   TAR       tar_lr=1e-05, alpha=5
#   wt_dist   LR=2e-05, noise_std=0.0001
#
# All runs: save weights, run eval benchmarks, push to HuggingFace.

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export PUSH_TO_HUB=1   # upload every model to HuggingFace
# NO_SAVE is intentionally NOT set — we keep weights for HF upload

echo "=========================================================="
echo "KL Best Configs — train + eval + push-to-hub"
echo "Model: $BASE"
echo "=========================================================="

# ------------------------------------------------------------------
# 1. CB  (LR=1.3e-05, BS=16, Ly=13-14-15)
# ------------------------------------------------------------------
echo -e "\n>>> [cb] LR=1.3e-05, BS=16, Ly=13-14-15 <<<"
EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 \
    ALPHA=1000.0 STEERING_COEFF=5.0 LAYER_ID="13,14,15" \
    ./unlearn/run_unlearn.sh cb

# ------------------------------------------------------------------
# 2. CB_LAT  (identical to CB + default LAT params)
# ------------------------------------------------------------------
echo -e "\n>>> [cb_lat] LR=1.3e-05, BS=16, Ly=13-14-15 <<<"
EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 \
    ALPHA=1000.0 STEERING_COEFF=5.0 LAYER_ID="13,14,15" \
    LAT_EPS=0.1 LAT_STEPS=5 \
    ./unlearn/run_unlearn.sh cb_lat

# ------------------------------------------------------------------
# 3. GA  (LR=2e-05, RW=5.0)
# ------------------------------------------------------------------
echo -e "\n>>> [ga] LR=2e-05, RW=5.0 <<<"
EPOCHS=1 LR=2e-05 BATCH_SIZE=32 RETAIN_WEIGHT=5.0 \
    ./unlearn/run_unlearn.sh ga

# ------------------------------------------------------------------
# 4. grad_diff  (LR=1e-05, FW=1.0)
# ------------------------------------------------------------------
echo -e "\n>>> [grad_diff] LR=1e-05, FW=1.0 <<<"
EPOCHS=1 LR=1e-05 BATCH_SIZE=32 FORGET_WEIGHT=1.0 \
    ./unlearn/run_unlearn.sh grad_diff

# ------------------------------------------------------------------
# 5. TAR  (tar_lr=1e-05, alpha=5)
# ------------------------------------------------------------------
echo -e "\n>>> [tar] tar_lr=1e-05, alpha=5 <<<"
TAR_LR=1e-05 TAR_ALPHA=5.0 TAR_EPOCHS=1 \
    ./unlearn/run_unlearn.sh tar

# ------------------------------------------------------------------
# 6. wt_dist  (LR=2e-05, noise_std=0.0001)
# ------------------------------------------------------------------
echo -e "\n>>> [wt_dist] LR=2e-05, noise_std=0.0001 <<<"
EPOCHS=1 LR=2e-05 BATCH_SIZE=32 WT_NOISE_STD=0.0001 \
    ./unlearn/run_unlearn.sh wt_dist

echo ""
echo "=========================================================="
echo "KL_best.sh completed — all models trained, evaluated, and"
echo "uploaded to HuggingFace."
echo "=========================================================="
