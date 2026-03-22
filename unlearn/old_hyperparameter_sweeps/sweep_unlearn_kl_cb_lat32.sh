#!/usr/bin/env bash
# KL's CB_LAT sweep (bs=32): same grid as sweep_unlearn_kl_cb_lat.sh but at batch size 32.
#
# Previous best cb_lat (sweep 8):
#   ep3_lr3e-05_bs8_a50.0_sc20.0_le0.1_ls5_ly5-6-7 -> Score 0.1177 (MMLU ~0.35, WMDP ~0.23)
#
# LAT parameters (fixed, established from sweeps 7-8):
#   - LAT_EPS=0.1:   default; 0.5 and 1.0 hurt, 0.2 marginal
#   - LAT_STEPS=5:   default; never swept
#
# This sweep mirrors sweep_unlearn_kl_cb_lat.sh exactly (same alpha/sc grid)
# but fixes batch_size=32 throughout to test whether larger batches help.
#
# Cross-sweep: alpha(50/100/150) × sc(15/20/25) × bs(32) = 9 runs
# Exploratory: LR=5e-05, bs=32, alpha(50/100/150) × sc(20/25) = 6 runs
# Total: 15 runs

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
BS=32

echo "=========================================================="
echo "KL's CB_LAT Sweep (bs=32): alpha × steering_coeff"
echo "Fixed: LR=${LR}, EPOCHS=${EPOCHS}, LAYER_ID=${LAYER_ID}, BS=${BS}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. Cross-sweep: alpha × steering_coeff at bs=32 (9 runs)
# ---------------------------------------------------------------
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "15.0" "20.0" "25.0"; do
        echo -e "\n>>> [cb_lat] LR=${LR}, EP=${EPOCHS}, BS=${BS}, ALPHA=${alpha}, SC=${sc}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
        EPOCHS=$EPOCHS LR=$LR BATCH_SIZE=$BS ALPHA=$alpha STEERING_COEFF=$sc \
            LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
            LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
    done
done

# ---------------------------------------------------------------
# 2. LR=5e-05 at bs=32 (6 runs)
# ---------------------------------------------------------------
echo -e "\n>>> [cb_lat] Exploratory: LR=5e-05 at bs=32 <<<"
for alpha in "50.0" "100.0" "150.0"; do
    for sc in "20.0" "25.0"; do
        echo -e "\n>>> [cb_lat] LR=5e-05, EP=${EPOCHS}, BS=${BS}, ALPHA=${alpha}, SC=${sc}, LAT_EPS=${LAT_EPS}, LAT_STEPS=${LAT_STEPS}, LY=${LAYER_ID} <<<"
        EPOCHS=$EPOCHS LR=5e-05 BATCH_SIZE=$BS ALPHA=$alpha STEERING_COEFF=$sc \
            LAT_EPS=$LAT_EPS LAT_STEPS=$LAT_STEPS \
            LAYER_ID=$LAYER_ID ./unlearn/run_unlearn.sh $METHOD
    done
done

echo "=========================================================="
echo "KL's CB_LAT Sweep (bs=32) completed!"
echo "=========================================================="
