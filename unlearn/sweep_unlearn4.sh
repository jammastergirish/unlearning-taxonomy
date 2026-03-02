#!/usr/bin/env bash
# Fourth sweep: STABILIZED version after gradient explosion at lr5e-05.
#
# Findings from sweeps 1-3:
#   - simnpo ep3 lr5e-05 beats the strong-filter on ALL WMDP metrics (Cat: 0.3189 vs 0.4006)
#   - lr=1e-04 collapses MMLU (<0.30); ceiling confirmed between 5e-05 and 1e-04
#   - CB/CB_LAT: broad layers preserve MMLU but barely unlearn; lr5e-05 never tested
#   - DPO at lr5e-05 never tested despite being same family as NPO/SimNPO
#   - retain_weight has only ever been 1.0; lowering it trades MMLU for more forgetting
#
# STABILITY FIXES APPLIED:
#   - Added gradient clipping (0.5) to prevent explosions
#   - Reduced beta from 0.1 to 0.01 for SimNPO/NPO/DPO (sigmoid stability)
#   - Lowered most learning rates: 5e-05→3e-05, 7e-05→5e-05, CB:5e-05→4e-05
#
# Goals (24 runs total - ADJUSTED FOR STABILITY):
#   1) Extend SimNPO/NPO epoch sweep at lr3e-05 (ep4, ep5)         [4 runs]
#   2) Probe intermediate LR 5e-05 for SimNPO and NPO              [6 runs]
#   3) DPO at lr3e-05 with safer beta values (0.01, 0.1)          [6 runs]
#   4) Retain-weight ablation (0.5, 0.1) for top methods            [4 runs]
#   5) CB/CB_LAT at lr4e-05 with broad layers (WMDP vs collapse)   [4 runs]

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export BATCH_SIZE=4
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1
export GRAD_CLIP=0.5

LAYER_ID_BROAD="5,10,15,20,25,30"

echo "=========================================================="
echo "Starting FOURTH hyperparameter sweep (STABILIZED)"
echo "Fixed gradient explosion issues with safer parameters"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. SimNPO: extend epoch sweep at SAFER lr3e-05 (reduced from 5e-05)
# ---------------------------------------------------------------
for ep in "4" "5"; do
    echo -e "\n>>> [simnpo] LR=3e-05, EPOCHS=${ep}, RETAIN_WEIGHT=1.0, BETA=0.01 <<<"
    EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo
done

# ---------------------------------------------------------------
# 2. NPO: extend epoch sweep at SAFER lr3e-05 (reduced from 5e-05)
# ---------------------------------------------------------------
for ep in "4" "5"; do
    echo -e "\n>>> [npo] LR=3e-05, EPOCHS=${ep}, RETAIN_WEIGHT=1.0, BETA=0.01 <<<"
    EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 3. Probe intermediate LR 5e-05 — safer than original 7e-05
#    May unlock better WMDP/MMLU Pareto point for both methods
# ---------------------------------------------------------------
for ep in "1" "2" "3"; do
    echo -e "\n>>> [simnpo] LR=5e-05, EPOCHS=${ep}, BETA=0.01 <<<"
    EPOCHS=$ep LR=5e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo

    echo -e "\n>>> [npo] LR=5e-05, EPOCHS=${ep}, BETA=0.01 <<<"
    EPOCHS=$ep LR=5e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 4. DPO at lr3e-05 — safer than original 5e-05
#    Lower beta values for stability
# ---------------------------------------------------------------
for ep in "1" "2" "3"; do
    for beta in "0.01" "0.1"; do
        echo -e "\n>>> [dpo] LR=3e-05, EPOCHS=${ep}, BETA=${beta} <<<"
        EPOCHS=$ep LR=3e-05 BETA=$beta ./unlearn/run_unlearn.sh dpo
    done
done

# ---------------------------------------------------------------
# 5. Retain-weight ablation for top methods at lr3e-05 (safer)
#    retain_weight=1.0 has been the only value tried; lower values
#    allow more aggressive forgetting at some MMLU cost
# ---------------------------------------------------------------
for rw in "0.5" "0.1"; do
    echo -e "\n>>> [simnpo] LR=3e-05, EPOCHS=3, RETAIN_WEIGHT=${rw}, BETA=0.01 <<<"
    EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh simnpo

    echo -e "\n>>> [npo] LR=3e-05, EPOCHS=3, RETAIN_WEIGHT=${rw}, BETA=0.01 <<<"
    EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 6. CB / CB_LAT at lr4e-05 with broad layers (safer than 5e-05)
#    Previous: lr3e-05 broad layers barely unlearns (Cat ~0.52).
#    Hypothesis: lr4e-05 broad layers might move WMDP without collapse,
#    as the stable operating regime is wider for circuit-breaking.
# ---------------------------------------------------------------
for ep in "1" "3"; do
    echo -e "\n>>> [cb] LR=4e-05, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_BROAD} <<<"
    EPOCHS=$ep LR=4e-05 LAYER_ID=$LAYER_ID_BROAD ALPHA=100.0 STEERING_COEFF=20.0 \
        ./unlearn/run_unlearn.sh cb

    echo -e "\n>>> [cb_lat] LR=4e-05, EPOCHS=${ep}, LAYER_ID=${LAYER_ID_BROAD} <<<"
    EPOCHS=$ep LR=4e-05 LAYER_ID=$LAYER_ID_BROAD ALPHA=100.0 STEERING_COEFF=20.0 \
        ./unlearn/run_unlearn.sh cb_lat
done

echo "=========================================================="
echo "Sweep 4 completed successfully!"
echo "=========================================================="
