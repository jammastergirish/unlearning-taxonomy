#!/usr/bin/env bash
# Sweep all hyperparameters for unlearning algorithms.
# Uses guidance from pre-training/safeguards literature.

set -euo pipefail

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Base settings
export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export BATCH_SIZE=4
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

# Common hyperparameter grids
LRS=("1.2e-5" "3e-5" "1e-5")    # From Table 7 / standard
EPOCHS_LIST=("1" "3")
LAYER_ID_REP="5,10,15,20,25,30" # From Section 2.3 for CB/LAT
LAYER_ID_STD="5,6,7"

# Define the methods groups
WEIGHT_METHODS=("ga_simple" "ga" "grad_diff" "dpo" "npo" "simnpo" "wt_dist" "wt_dist_reg")
REP_METHODS=("rmu" "cb" "lat" "cb_lat")

echo "=========================================================="
echo "Starting comprehensive hyperparameter sweep for unlearning"
echo "Model: $BASE"
echo "=========================================================="

# 1. Sweep standard weight-based & contrastive methods
for method in "${WEIGHT_METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for ep in "${EPOCHS_LIST[@]}"; do
            echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep} <<<"
            LR=$lr EPOCHS=$ep ./unlearn/run_unlearn.sh "$method"
        done
    done
done

# 2. Sweep representation-based methods (requires LAYER_ID)
for method in "${REP_METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for ep in "${EPOCHS_LIST[@]}"; do
            echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_REP} <<<"
            LR=$lr EPOCHS=$ep LAYER_ID=$LAYER_ID_REP ./unlearn/run_unlearn.sh "$method"

            # Optionally run with standard layers as a baseline comparison
            echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_STD} <<<"
            LR=$lr EPOCHS=$ep LAYER_ID=$LAYER_ID_STD ./unlearn/run_unlearn.sh "$method"
        done
    done
done

echo "=========================================================="
echo "Sweep completed successfully!"
echo "=========================================================="
