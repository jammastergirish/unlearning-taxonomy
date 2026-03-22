#!/usr/bin/env bash
# Third sweep of hyperparameters targeting methods that showed
# either collapse or insufficient unlearning in the previous sweeps.
# This sweep performs surgical, high-resolution sweeps in narrow functional bands.

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

# 1. High-Resolution LR Sweep for Loss Methods
# Sweep 1 (3e-5) collapsed. Sweep 2 (1e-5) was too gentle.
# We sweep the exact margin where grad_diff showed life in Sweep 2.
NARROW_LRS=("1.0e-5" "1.2e-5" "1.4e-5" "1.6e-5")
LOSS_EPOCHS=("1" "2")

# 2. Continuous Depth Span for Activation Methods
# Sweep 1 (5-6-7) collapsed. Sweep 2 (5,10,15,20,25,30) was a bypass (no-op).
# We must target a continuous span of middle layers where facts are usually extracted.
LAYER_ID_CONTINUOUS="12,13,14,15,16,17,18,19,20"
ACT_LRS=("1e-5" "2e-5")
ACT_EPOCHS=("2" "3")

# Define method groups
LOSS_METHODS=("ga_simple" "grad_diff" "dpo" "wt_dist_reg")
ACT_METHODS=("cb" "rmu" "lat" "cb_lat")

echo "=========================================================="
echo "Starting THIRD hyperparameter sweep (Surgical Search)"
echo "Model: $BASE"
echo "=========================================================="

# 1. Sweep Loss methods
for method in "${LOSS_METHODS[@]}"; do
    for lr in "${NARROW_LRS[@]}"; do
        for ep in "${LOSS_EPOCHS[@]}"; do
            
            if [[ "$method" == "grad_diff" ]]; then
                for fw in "1.0" "2.0"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, FORGET_WEIGHT=${fw} <<<"
                    LR=$lr EPOCHS=$ep FORGET_WEIGHT=$fw ./unlearn/run_unlearn.sh "$method"
                done
            elif [[ "$method" == "dpo" ]]; then
                for beta in "0.1" "0.5"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, BETA=${beta} <<<"
                    LR=$lr EPOCHS=$ep BETA=$beta ./unlearn/run_unlearn.sh "$method"
                done
            elif [[ "$method" == "wt_dist_reg" ]]; then
                 for lambda in "0.05" "0.1"; do
                    echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, WT_REG_LAMBDA=${lambda} <<<"
                    LR=$lr EPOCHS=$ep WT_REG_LAMBDA=$lambda ./unlearn/run_unlearn.sh "$method"
                 done
            else
                echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep} <<<"
                LR=$lr EPOCHS=$ep ./unlearn/run_unlearn.sh "$method"
            fi
            
        done
    done
done

# 2. Sweep Activation methods (Continuous Depth)
for method in "${ACT_METHODS[@]}"; do
    for lr in "${ACT_LRS[@]}"; do
        for ep in "${ACT_EPOCHS[@]}"; do
            
            echo -e "\n>>> Running [${method}] LR=${lr}, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_CONTINUOUS} <<<"
            LR=$lr EPOCHS=$ep LAYER_ID=$LAYER_ID_CONTINUOUS ./unlearn/run_unlearn.sh "$method"
            
        done
    done
done

echo "=========================================================="
echo "Surgical Sweep 3 completed successfully!"
echo "=========================================================="
