#!/usr/bin/env bash
# Seventh sweep: Optimizing cb_lat
#
# Our previous cb_lat runs topped out around a Score of 0.10:
# (cb_lat ep3 lr3e-05 bs4 a100.0 sc20.0 le0.1 ls5 ly5-6-7 -> Score 0.1023, MMLU 0.35, WMDP 0.24)
# 
# While WMDP was good (0.24), MMLU dropped heavily to 0.35.
# If we try to maintain MMLU (e.g. lr1.2e-5), WMDP shoots back up to 0.42 (Score 0.02).
#
# Goal: Apply the lessons from SimNPO and NPO to cb_lat to break this trade-off:
# 1. Increase batch size to 8 (cleaner gradients for the complex Min-Max LAT objective)
# 2. Adjust the defense/offense balance: 
#    - Lower alpha (from 100 to 50/20) to allow more representation flexibility
#    - Increase lat_eps (from 0.1 to 0.5) to enforce stronger adversarial robustness

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting SEVENTH hyperparameter sweep"
echo "Optimizing cb_lat (Circuit Breakers + Latent Adversarial Training)"
echo "Model: $BASE"
echo "=========================================================="

METHOD="cb_lat"

# ---------------------------------------------------------------
# 1. The Batch Size 8 Baseline
# We know lr=3e-5 was needed to move WMDP, but it hurt MMLU.
# Let's see if bs=8 fixes the MMLU drop at lr=3e-5.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Testing batch size 8 at high LR <<<"
for ep in "1" "3"; do
    EPOCHS=$ep LR=3e-05 BATCH_SIZE=8 ALPHA=100.0 LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD 
done

# ---------------------------------------------------------------
# 2. Alpha Ablation (Retain Weight Concept)
# In SimNPO, dropping retain_weight improved the trade-off.
# In CB, `alpha` (default 100) controls the retain loss multiplier.
# Let's try relaxing the retain constraint (alpha=20, 50) at a strong LR.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Relaxing the retain constraint (Alpha) <<<"
for alpha in "20.0" "50.0"; do
    EPOCHS=3 LR=3e-05 BATCH_SIZE=4 ALPHA=$alpha LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
    
    # And stack it with the batch size improvement
    EPOCHS=3 LR=3e-05 BATCH_SIZE=8 ALPHA=$alpha LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 3. LAT_EPS Ablation (Adversarial Strength)
# The default lat_eps is 0.1. What if we make the adversary stronger (0.5)?
# This forces the model to be more robust, potentially improving forgetting.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Increasing adversarial perturbation strength (lat-eps) <<<"
for lat_eps in "0.5" "1.0"; do
    # At standard batch size 4
    EPOCHS=3 LR=3e-05 BATCH_SIZE=4 ALPHA=100.0 LAT_EPS=$lat_eps \
        ./unlearn/run_unlearn.sh $METHOD
    
    # At batch size 8
    EPOCHS=3 LR=3e-05 BATCH_SIZE=8 ALPHA=100.0 LAT_EPS=$lat_eps \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 4. Learning Rate Push
# If alpha is lower, or bs is higher, we might be able to handle a larger LR.
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Pushing learning rate with optimal bs=8 and standard alpha=100 <<<"
for lr in "5e-05" "7e-05"; do
    EPOCHS=3 LR=$lr BATCH_SIZE=8 ALPHA=100.0 LAT_EPS=0.1 \
        ./unlearn/run_unlearn.sh $METHOD
done

echo "=========================================================="
echo "Sweep 7 completed successfully!"
echo "=========================================================="
