#!/usr/bin/env bash
# Sixth sweep: Combining the successful directions from Sweep 5.
#
# Finding 1 (SimNPO): Lowering retain_weight at high LR (7e-05) massively improved WMDP 
# without tanking MMLU. (rw=0.3 scored 0.1475, our best yet).
# Finding 2 (NPO): Increasing batch size (to 8) smoothed gradients and improved
# the Pareto frontier significantly (Score 0.1310).
#
# Goal: Test if these two independent improvements (batch size > 4, retain weight < 1.0)
# stack to create an even better model.
#
# Also pushing to extremes: 
# - For SimNPO, test rw=0.1 at 7e-05 (does MMLU hold up?)
# - For NPO, test bs=8,16 crossed with rw=0.3,0.5 at lr=5e-05.
#
# 10 runs total.

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting SIXTH hyperparameter sweep"
echo "Combining larger batch sizes and lower retain weights"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# SimNPO — Exploring the high-LR / low-RW frontier
# Best so far: ep3 lr7e-05 bs4 rw0.3 (Score 0.1475)
# Action 1: Push rw lower to 0.1 at bs=4
# Action 2: Try larger batch sizes (8, 16) at the optimal rw=0.3
# ---------------------------------------------------------------
echo -e "\n>>> [simnpo] Pushing retain_weight lower at optimal high LR <<<"
EPOCHS=3 LR=7e-05 BATCH_SIZE=4 BETA=0.1 RETAIN_WEIGHT=0.1 \
    ./unlearn/run_unlearn.sh simnpo

for bs in "8" "16"; do
    echo -e "\n>>> [simnpo] Testing larger batch size at optimal low RW <<<"
    EPOCHS=3 LR=7e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=0.3 \
        ./unlearn/run_unlearn.sh simnpo
done

# ---------------------------------------------------------------
# NPO — Exploring the large-BS / low-RW frontier
# Best so far:
#   ep3 lr5e-05 bs4 rw0.3 (Score 0.1349)
#   ep3 lr5e-05 bs8 rw1.0 (Score 0.1310)
#   ep3 lr5e-05 bs8 rw0.7 (Score 0.1212)
# Action: Cross the larger batch sizes (8, 16) with the lower RWs (0.3, 0.5)
# ---------------------------------------------------------------
for bs in "8" "16"; do
    for rw in "0.3" "0.5"; do
        echo -e "\n>>> [npo] Crossing large bs=${bs} with low rw=${rw} <<<"
        EPOCHS=3 LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=$rw \
            ./unlearn/run_unlearn.sh npo
    done
done

# ---------------------------------------------------------------
# NPO + SimNPO — Extending Epochs at optimal configurations
# Since larger batches reduce the number of optimization steps, we might
# need more epochs to reach the true minimum. Test ep=4,5 at the best configs.
# ---------------------------------------------------------------
for ep in "4" "5"; do
    # Best NPO large-batch config
    echo -e "\n>>> [npo] Extending epochs for optimal large-batch config <<<"
    EPOCHS=$ep LR=5e-05 BATCH_SIZE=8 BETA=0.1 RETAIN_WEIGHT=1.0 \
        ./unlearn/run_unlearn.sh npo
    
    # Best SimNPO overall config
    echo -e "\n>>> [simnpo] Extending epochs for optimal overall config <<<"
    EPOCHS=$ep LR=7e-05 BATCH_SIZE=4 BETA=0.1 RETAIN_WEIGHT=0.3 \
        ./unlearn/run_unlearn.sh simnpo
done

echo "=========================================================="
echo "Sweep 6 completed successfully!"
echo "=========================================================="
