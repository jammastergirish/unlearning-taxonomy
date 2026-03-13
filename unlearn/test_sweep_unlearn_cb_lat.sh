#!/usr/bin/env bash
# Test sweep for CB, CB-LAT, and LAT after scheduled coefficient changes.
#
# Step 1: Find good batch size (8, 16, 32) with fixed LR and coefficients.
#         Check W&B for smooth loss curves, then fix batch size going forward.
# Step 2: Sweep LR and retain/remove coefficients (not in this script).
#
# CLI arg mapping (internal name → CLI flag):
#   remove_coef  → --steering-coeff / STEERING_COEFF
#   retain_coef  → --alpha / ALPHA  (for CB/CB-LAT)
#   retain_coef  → --retain-weight / RETAIN_WEIGHT  (for LAT)

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

LAYER_ID="5,6,7"
LR="3e-05"
EPOCHS=1
LAT_EPS="0.1"
LAT_STEPS="5"

# Coefficient defaults (using descriptive names)
REMOVE_COEF="20.0"
RETAIN_COEF_CB="100.0"      # --alpha for CB/CB-LAT
RETAIN_COEF_LAT="1.0"       # --retain-weight for LAT

echo "=========================================================="
echo "Test sweep: CB / CB-LAT / LAT — batch size search"
echo "Fixed: LR=${LR}, EPOCHS=${EPOCHS}, LAYER_ID=${LAYER_ID}"
echo "  CB/CB-LAT: remove_coef=${REMOVE_COEF}, retain_coef=${RETAIN_COEF_CB}"
echo "  LAT:       retain_coef=${RETAIN_COEF_LAT}"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# CB: batch size 8, 16, 32 (3 runs)
# ---------------------------------------------------------------
for bs in 8 16 32; do
    echo -e "\n>>> [cb] LR=${LR}, EP=${EPOCHS}, BS=${bs}, remove_coef=${REMOVE_COEF}, retain_coef=${RETAIN_COEF_CB}, LY=${LAYER_ID} <<<"
    EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$bs \
        STEERING_COEFF=$REMOVE_COEF ALPHA=$RETAIN_COEF_CB \
        LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh cb
done

# ---------------------------------------------------------------
# CB-LAT: batch size 8, 16, 32 (3 runs)
# ---------------------------------------------------------------
for bs in 8 16 32; do
    echo -e "\n>>> [cb_lat] LR=${LR}, EP=${EPOCHS}, BS=${bs}, remove_coef=${REMOVE_COEF}, retain_coef=${RETAIN_COEF_CB}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
    EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$bs \
        STEERING_COEFF=$REMOVE_COEF ALPHA=$RETAIN_COEF_CB \
        LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
        LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh cb_lat
done

# ---------------------------------------------------------------
# LAT: batch size 8, 16, 32 (3 runs)
# ---------------------------------------------------------------
for bs in 8 16 32; do
    echo -e "\n>>> [lat] LR=${LR}, EP=${EPOCHS}, BS=${bs}, retain_coef=${RETAIN_COEF_LAT}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
    EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$bs \
        RETAIN_WEIGHT=$RETAIN_COEF_LAT \
        LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
        LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh lat
done

echo "=========================================================="
echo "Test sweep completed! (9 runs total)"
echo "Check W&B for smooth loss curves, then pick batch size."
echo "=========================================================="
