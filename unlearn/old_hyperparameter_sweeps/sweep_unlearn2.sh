#!/usr/bin/env bash
# Second sweep of hyperparameters targeting methods that showed
# either collapse or insufficient unlearning in the first sweep.
# Excludes npo and simnpo which perform well with default parameters.

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

# New, broader hyperparameter grids focusing on gentler learning
# We want to prevent the catastrophic collapse seen in cb/cb_lat/rmu/lat
# and the weak unlearning seen in dpo.
LRS=("5e-6" "1e-5" "2e-5")      # Shifted lower to prevent collapse
EPOCHS_LIST=("3" "5")           # More epochs at lower LR

# For representational methods, we'll try different layer targets
# Fewer layers might prevent catastrophic collapse
LAYER_ID_REP_SPARSE="10,20,30"
LAYER_ID_REP_DENSE="5,10,15,20,25,30"

# Define the methods groups (excluding npo/simnpo)
WEIGHT_METHODS=("ga_simple" "ga" "grad_diff" "dpo" "wt_dist" "wt_dist_reg")
REP_METHODS=("rmu" "cb" "lat" "cb_lat")

echo "=========================================================="
echo "Starting SECOND hyperparameter sweep for unlearning"
echo "Targeting methods that collapsed or underperformed"
echo "Model: $BASE"
echo "=========================================================="

# 1. Sweep weight-based methods
for method in "${WEIGHT_METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for ep in "${EPOCHS_LIST[@]}"; do
            echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep} <<<"
            
            # For GA variants, try varying the retain/forget weight balance
            if [[ "$method" == "ga" ]]; then
                for rw in "1.0" "5.0"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, RETAIN_WEIGHT=${rw} <<<"
                    LR=$lr EPOCHS=$ep RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh "$method"
                done
            elif [[ "$method" == "grad_diff" ]]; then
                for fw in "0.5" "1.0"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, FORGET_WEIGHT=${fw} <<<"
                    LR=$lr EPOCHS=$ep FORGET_WEIGHT=$fw ./unlearn/run_unlearn.sh "$method"
                done
            elif [[ "$method" == "dpo" ]]; then
                # DPO had weak unlearning, try a sharper beta
                for beta in "0.1" "0.5"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, BETA=${beta} <<<"
                    LR=$lr EPOCHS=$ep BETA=$beta ./unlearn/run_unlearn.sh "$method"
                done
            elif [[ "$method" == "wt_dist" ]]; then
                 # Try gentler noise
                 for wn in "0.01" "0.02"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, NOISE=${wn} <<<"
                    LR=$lr EPOCHS=$ep WT_NOISE_STD=$wn ./unlearn/run_unlearn.sh "$method"
                 done
            else
                LR=$lr EPOCHS=$ep ./unlearn/run_unlearn.sh "$method"
            fi
        done
    done
done

# 2. Sweep representation-based methods (requires LAYER_ID)
for method in "${REP_METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
        for ep in "${EPOCHS_LIST[@]}"; do
            # For RMU/CB, tune the alpha (retain weight) to prevent full collapse
            for alpha in "100.0" "500.0"; do
                echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_REP_SPARSE}, ALPHA=${alpha} <<<"
                LR=$lr EPOCHS=$ep LAYER_ID=$LAYER_ID_REP_SPARSE ALPHA=$alpha ./unlearn/run_unlearn.sh "$method"

                echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_REP_DENSE}, ALPHA=${alpha} <<<"
                LR=$lr EPOCHS=$ep LAYER_ID=$LAYER_ID_REP_DENSE ALPHA=$alpha ./unlearn/run_unlearn.sh "$method"
            done
        done
    done
done

echo "=========================================================="
echo "Sweep 2 completed successfully!"
echo "=========================================================="
