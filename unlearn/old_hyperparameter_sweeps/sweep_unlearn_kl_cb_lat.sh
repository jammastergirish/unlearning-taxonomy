#!/usr/bin/env bash
# KL's CB_LAT sweep: same grid as CB sweep + fixed LAT defaults
#
# Previous best cb_lat (sweep 8):
#   ep3_lr3e-05_bs8_a50.0_sc20.0_le0.1_ls5_ly5-6-7 -> Score 0.1177 (MMLU ~0.35, WMDP ~0.23)
#
# LAT parameters (fixed, established from sweeps 7-8):
#   - LAT_EPS=0.1:   default; 0.5 and 1.0 hurt, 0.2 marginal
#   - LAT_STEPS=5:   default; never swept
#
# This sweep mirrors sweep_unlearn_kl_cb.sh exactly (same alpha/sc/bs grid)
# but with METHOD=cb_lat and the above LAT defaults.
#
# Cross-sweep: alpha(50/100/150) × sc(15/20/25) × bs(4/8) = 18 runs
# Exploratory: LR=5e-05, bs=8, alpha(50/100/150) × sc(20/25) = 6 runs
# Total: 24 runs

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

METHOD="cb_lat"
LAYER_ID="5,6,7"
LR="3e-05"
EPOCHS=3
LAT_EPS="0.1"
LAT_STEPS="5"

echo "=========================================================="
echo "KL's CB_LAT Sweep: alpha × steering_coeff × batch_size"
echo "Fixed: LR=${LR}, EPOCHS=${EPOCHS}, LAYER_ID=${LAYER_ID}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. Cross-sweep: alpha × steering_coeff × batch_size (18 runs)
# ---------------------------------------------------------------
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "15.0" "20.0" "25.0"; do
        for bs in "4" "8"; do
            echo -e "\n>>> [cb_lat] LR=${LR}, EP=${EPOCHS}, BS=${bs}, ALPHA=${alpha}, SC=${sc}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
            EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$bs ALPHA=$alpha STEERING_COEFF=$sc \
                LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
                LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
        done
    done
done

# ---------------------------------------------------------------
# 2. LR=5e-05 at bs=8 (6 runs)
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Exploratory: LR=5e-05 at bs=8 <<<"
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "20.0" "25.0"; do
        echo -e "\n>>> [cb_lat] LR=5e-05, EP=${EPOCHS}, BS=8, ALPHA=${alpha}, SC=${sc}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
        EPOCHS=$EPOCHS LR=5e-05 BATCH_SIZE=8 ALPHA=$alpha STEERING_COEFF=$sc \
            LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
            LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
    done
done

echo "=========================================================="
echo "KL's CB_LAT Sweep completed!"
echo "=========================================================="
