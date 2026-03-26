#!/usr/bin/env bash
# KL's CB sweep: cross-sweep alpha × steering_coeff × batch_size
#
# Previous best cb (sweep 9):
#   ep3_lr3e-05_bs4_a100.0_sc20.0_ly5-6-7 -> Score 0.0921 (MMLU 0.35, WMDP 0.26)
#
# What's been explored:
#   - LR: 3e-05 is the sweet spot (well-established across sweeps 1-9)
#   - Epochs: 3 is optimal (1 too few, 4-5 no improvement)
#   - Layer ID: 5,6,7 best (tight spans >> broad spreads)
#   - Alpha: 50, 100, 150, 200 tested at bs=4; only 50, 100 at bs=8
#   - Steering coeff: 10, 20, 30, 40 tested only at bs=4 alpha=100
#   - Batch size: 4, 8 tested but not cross-swept with sc/alpha
#
# What's missing:
#   - Alpha × steering_coeff interaction (never cross-swept)
#   - Batch size × steering_coeff interaction (never tested)
#   - Steering_coeff values between 20-40 (only 10, 20, 30, 40 tested)
#
# This sweep fixes LR=3e-05, EPOCHS=3, LAYER_ID=5,6,7 and cross-sweeps:
#   alpha    × steering_coeff × batch_size
#   (3 vals)   (3 vals)         (2 vals)    = 18 runs
#
# Plus 6 exploratory runs at LR=5e-05 with the best batch_size=8 configs.
# Total: 24 runs

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
#export NO_SAVE=1

METHOD="cb"
LAYER_ID="5,6,7"
LR="3e-05"
EPOCHS=1

echo "=========================================================="
echo "KL's CB Sweep: alpha × steering_coeff × batch_size"
echo "Fixed: LR=${LR}, EPOCHS=${EPOCHS}, LAYER_ID=${LAYER_ID}"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. Cross-sweep: alpha × steering_coeff × batch_size (18 runs)
#
# Alpha: 50, 100, 150
#   - 50:  less retain preservation, more room to unlearn
#   - 100: current best
#   - 150: stronger retain, see if MMLU improves enough to justify
#
# Steering coeff: 15, 20, 25
#   - 15:  gentler rerouting
#   - 20:  current best
#   - 25:  between 20 and 30 (never tested)
#
# Batch size: 4, 8
#   - 4: current best baseline
#   - 8: helped cb_lat, may smooth gradients for cb too
# ---------------------------------------------------------------
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "15.0" "20.0" "25.0"; do
        for bs in "4" "8"; do
            echo -e "\n>>> [cb] LR=${LR}, EP=${EPOCHS}, BS=${bs}, ALPHA=${alpha}, SC=${sc}, LY=${LAYER_ID} <<<"
            EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$bs ALPHA=$alpha STEERING_COEFF=$sc \
                LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
        done
    done
done

# ---------------------------------------------------------------
# 2. LR=5e-05 at bs=8 (6 runs)
#
# Sweep 9 tested lr=5e-05 at bs=4 and bs=8 but only with alpha=100 sc=20.
# If bs=8 smooths gradients, a higher LR might push WMDP lower without
# MMLU collapse. Test the 3 alpha values at sc=20 (the known best sc).
# ---------------------------------------------------------------
echo -e "\n>>> [cb] Exploratory: LR=5e-05 at bs=8 <<<"
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "20.0" "25.0"; do
        echo -e "\n>>> [cb] LR=5e-05, EP=${EPOCHS}, BS=8, ALPHA=${alpha}, SC=${sc}, LY=${LAYER_ID} <<<"
        EPOCHS=$EPOCHS LR=5e-05 BATCH_SIZE=8 ALPHA=$alpha STEERING_COEFF=$sc \
            LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
    done
done

echo "=========================================================="
echo "KL's CB Sweep completed!"
echo "=========================================================="
