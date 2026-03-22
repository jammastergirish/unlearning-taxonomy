#!/usr/bin/env bash
# Eighth sweep: Fine-tuning cb_lat further based on Sweep 7 results.
#
# Sweep 7 key finding: bs=8 + alpha=50 is the new cb_lat best (Score 0.1177)
# vs the original bs=4 + alpha=100 (Score 0.1023).
# This confirms: larger batch -> need lower alpha to compensate for fewer steps.
#
# This sweep explores:
# 1. Pushing alpha even lower (10, 20) at bs=8, lr=3e-05 to see if we can
#    drive WMDP below 0.25 while keeping MMLU above 0.38.
# 2. Trying a medium LR (5e-5) with the optimal bs=8, alpha=50 config.
# 3. More epochs (4, 5) at the best config (bs=8, alpha=50) to check convergence.
# 4. Increasing lat_eps at bs=8 to see if a stronger adversary
#    can push WMDP even lower without MMLU collapse.

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting EIGHTH hyperparameter sweep"
echo "Fine-tuning cb_lat: alpha, lat_eps, and LR at bs=8"
echo "Model: $BASE"
echo "=========================================================="

METHOD="cb_lat"

# ---------------------------------------------------------------
# 1. Alpha Push: Drive alpha to the floor at bs=8, lr=3e-05
# Best so far:  bs=8, alpha=50 -> Score 0.1177 (MMLU 0.38, WMDP 0.27)
# Question: does alpha=20 give even better WMDP while keeping MMLU?
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Pushing alpha lower at bs=8 <<<"
for alpha in "20.0" "10.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=8 ALPHA=$alpha LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 2. Learning Rate at bs=8, alpha=50
# Our best was lr=3e-05. Let's see if a slightly higher LR (5e-05)
# pushes more unlearning without causing MMLU collapse.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Trying higher LR at bs=8, alpha=50 <<<"
EPOCHS=3 LR=5e-05 BATCH_SIZE=8 ALPHA=50.0 LAT_EPS=0.1 \
    ./unlearn/run_unlearn.sh $METHOD

# ---------------------------------------------------------------
# 3. Stacking bs=16 with optimal alpha
# If bs=8 needed alpha=50 rather than alpha=100, does bs=16 want alpha=25?
# The pattern: lower alpha = less retain preservation per gradient step.
# More steps per epoch (bs=4) can afford alpha=100; fewer (bs=16) may need less.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Testing bs=16 with proportionally lower alpha <<<"
for alpha in "25.0" "50.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=16 ALPHA=$alpha LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 4. Stronger Adversary at bs=8, alpha=50
# lat_eps=0.5 already showed it hurt (Score 0.0786 at bs=4, alpha=100).
# Let's try lat_eps=0.2 — one step larger than default but not as harsh as 0.5.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Stronger adversary at optimal bs=8, alpha=50 <<<"
EPOCHS=3 LR=3e-05 BATCH_SIZE=8 ALPHA=50.0 LAT_EPS=0.2 \
    ./unlearn/run_unlearn.sh $METHOD

# ---------------------------------------------------------------
# 5. Convergence check: extend epochs at the new best config
# (We found ep=3 converges at bs=4, but bs=8 takes fewer steps per epoch)
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Epoch sweep at bs=8, alpha=50 <<<"
for ep in "4" "5"; do
    EPOCHS=$ep LR=3e-05 BATCH_SIZE=8 ALPHA=50.0 LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
done

echo "=========================================================="
echo "Sweep 8 completed successfully!"
echo "=========================================================="
