#!/usr/bin/env bash
# KL_best.sh — Run the 6 best unlearning configs (one per method).
#
# These configs were selected from the cross-method hyperparameter sweep
# documented in unlearn/experiment_logs/hyper_params_sweep_report.md.
# They maximize L2 weight distance from base while preserving MMLU (~0.45)
# and keeping WMDP Bio Robust Rewritten near random (~0.25).
#
# Metrics can be verified with: uv run --script unlearn/analysis/verify_best_configs.py
#
# Best configs:
#   CB        LR=1.3e-05, BS=16, Alpha=200, SC=10, Ly=13-14-15
#   CB_LAT    LR=1.3e-05, BS=16, Alpha=200, SC=10, Ly=13-14-15, LAT_EPS=0.1, LAT_STEPS=5
#   GA        LR=2e-05, BS=4, RW=5.0
#   grad_diff LR=1e-05, BS=4, FW=1.0
#   TAR       tar_lr=1e-05, alpha=5, tar_epochs=1
#   wt_dist   LR=2e-05, BS=4, noise_std=0.0001

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=1024

echo "=========================================================="
echo "KL Best Configs — train + eval"
echo "Model: $BASE"
echo "=========================================================="

# ------------------------------------------------------------------
# 1. CB  (LR=1.3e-05, BS=16, Alpha=200, SC=10, Ly=13-14-15)
# ------------------------------------------------------------------
echo -e "\n>>> [cb] LR=1.3e-05, BS=16, Alpha=200, SC=10, Ly=13-14-15 <<<"
EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 ALPHA=200.0 STEERING_COEFF=10.0 LAYER_ID="13,14,15" ./unlearn/run_unlearn.sh cb --force

# ------------------------------------------------------------------
# 2. CB_LAT  (same as CB + LAT_EPS=0.1, LAT_STEPS=5)
# ------------------------------------------------------------------
echo -e "\n>>> [cb_lat] LR=1.3e-05, BS=16, Alpha=200, SC=10, Ly=13-14-15, LAT <<<"
EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 ALPHA=200.0 STEERING_COEFF=10.0 LAYER_ID="13,14,15" LAT_EPS=0.1 LAT_STEPS=5 ./unlearn/run_unlearn.sh cb_lat --force

# ------------------------------------------------------------------
# 3. GA  (LR=2e-05, BS=4, RW=5.0)
# ------------------------------------------------------------------
echo -e "\n>>> [ga] LR=2e-05, BS=4, RW=5.0 <<<"
EPOCHS=1 LR=2e-05 BATCH_SIZE=4 RETAIN_WEIGHT=5.0 ./unlearn/run_unlearn.sh ga --force

# ------------------------------------------------------------------
# 4. grad_diff  (LR=1e-05, BS=4, FW=1.0)
# ------------------------------------------------------------------
echo -e "\n>>> [grad_diff] LR=1e-05, BS=4, FW=1.0 <<<"
EPOCHS=1 LR=1e-05 BATCH_SIZE=4 FORGET_WEIGHT=1.0 ./unlearn/run_unlearn.sh grad_diff --force

# ------------------------------------------------------------------
# 5. TAR  (tar_lr=1e-05, alpha=5, tar_epochs=1)
# ------------------------------------------------------------------
echo -e "\n>>> [tar] tar_lr=1e-05, alpha=5 <<<"
TAR_LR=1e-05 TAR_ALPHA=5.0 TAR_EPOCHS=1 ./unlearn/run_unlearn.sh tar --force

# ------------------------------------------------------------------
# 6. wt_dist  (LR=2e-05, BS=4, noise_std=0.0001)
# ------------------------------------------------------------------
echo -e "\n>>> [wt_dist] LR=2e-05, BS=4, noise_std=0.0001 <<<"
EPOCHS=1 LR=2e-05 BATCH_SIZE=4 WT_NOISE_STD=0.0001 ./unlearn/run_unlearn.sh wt_dist --force

echo ""
echo "=========================================================="
echo "KL_best.sh completed — all 6 models trained and evaluated."
echo "=========================================================="
