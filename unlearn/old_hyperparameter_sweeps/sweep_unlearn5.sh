#!/usr/bin/env bash
# Fifth sweep: batch size ablation for NPO and SimNPO.
#
# Best configs from sweeps 3-4:
#   simnpo ep3 lr5e-05 bs4 b0.1 rw1.0  → WMDP Cat 0.3189, MMLU 0.4113 (best overall)
#   npo    ep3 lr5e-05 bs4 b0.1 rw1.0  → WMDP Cat 0.3464, MMLU 0.4054
#
# Goal: test whether larger batches improve gradient quality and push WMDP lower.
# Larger batches → less noisy gradients → possibly more precise forgetting.
# Note: LR is NOT scaled with batch size here; the existing lr5e-05 is already
# near the collapse ceiling so scaling up would likely cause collapse.
#
# 8 runs total (4 per method × 2 batch sizes).

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1

echo "=========================================================="
echo "Starting FIFTH hyperparameter sweep"
echo "Batch size ablation for NPO and SimNPO at optimal LR"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# SimNPO — optimal: ep3 lr5e-05 b0.1 rw1.0 (baseline bs=4)
# ---------------------------------------------------------------
for bs in "8" "16"; do
    for ep in "2" "3"; do
        echo -e "\n>>> [simnpo] EPOCHS=${ep}, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=1.0 <<<"
        EPOCHS=$ep LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=1.0 \
            ./unlearn/run_unlearn.sh simnpo
    done
done

# ---------------------------------------------------------------
# NPO — optimal: ep3 lr5e-05 b0.1 rw1.0 (baseline bs=4)
# ---------------------------------------------------------------
for bs in "8" "16"; do
    for ep in "1" "3"; do
        echo -e "\n>>> [npo] EPOCHS=${ep}, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=1.0 <<<"
        EPOCHS=$ep LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=1.0 \
            ./unlearn/run_unlearn.sh npo
    done
done

# ---------------------------------------------------------------
# Retain-weight fine-grained ablation for both methods
#
# Sweep 4 showed rw=0.1 and rw=0.5 at lr5e-05 ep3:
#   rw=0.5 → WMDP Cat 0.2765, MMLU 0.3648  (great WMDP, big MMLU hit)
#   rw=0.1 → WMDP Cat 0.2490, MMLU 0.3472  (best WMDP, heavy MMLU hit)
#
# Fill in the gaps: rw=0.3 and rw=0.7 sit between those data points.
# rw=0.5 included for NPO (ran for simnpo in sweep 4 already, will be skipped).
# Also test the same rw values at lr7e-05 for simnpo (best WMDP LR).
# ---------------------------------------------------------------

# simnpo: rw sweep at optimal lr5e-05
for rw in "0.3" "0.5" "0.7"; do
    echo -e "\n>>> [simnpo] EPOCHS=3, LR=5e-05, BS=4, BETA=0.1, RETAIN_WEIGHT=${rw} <<<"
    EPOCHS=3 LR=5e-05 BATCH_SIZE=4 BETA=0.1 RETAIN_WEIGHT=$rw \
        ./unlearn/run_unlearn.sh simnpo
done

# simnpo: rw sweep at lr7e-05 (best WMDP lr, so far only rw=1.0 tested there)
for rw in "0.3" "0.5" "0.7"; do
    echo -e "\n>>> [simnpo] EPOCHS=3, LR=7e-05, BS=4, BETA=0.1, RETAIN_WEIGHT=${rw} <<<"
    EPOCHS=3 LR=7e-05 BATCH_SIZE=4 BETA=0.1 RETAIN_WEIGHT=$rw \
        ./unlearn/run_unlearn.sh simnpo
done

# npo: rw sweep at optimal lr5e-05 (rw was only ever 1.0 for npo at this lr)
for rw in "0.3" "0.5" "0.7"; do
    echo -e "\n>>> [npo] EPOCHS=3, LR=5e-05, BS=4, BETA=0.1, RETAIN_WEIGHT=${rw} <<<"
    EPOCHS=3 LR=5e-05 BATCH_SIZE=4 BETA=0.1 RETAIN_WEIGHT=$rw \
        ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# Retain-weight × batch-size cross sweep
#
# Combines the two dimensions we're now exploring: does a larger batch
# size interact with retain_weight? rw=0.5 and rw=0.7 are the most
# useful midpoints — low enough to get meaningful unlearning but not
# collapsing MMLU. bs=8 and bs=16 are the two larger sizes from above.
# ---------------------------------------------------------------

# simnpo: rw × bs at lr5e-05
for rw in "0.5" "0.7"; do
    for bs in "8" "16"; do
        echo -e "\n>>> [simnpo] EPOCHS=3, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=${rw} <<<"
        EPOCHS=3 LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=$rw \
            ./unlearn/run_unlearn.sh simnpo
    done
done

# npo: rw × bs at lr5e-05
for rw in "0.5" "0.7"; do
    for bs in "8" "16"; do
        echo -e "\n>>> [npo] EPOCHS=3, LR=5e-05, BS=${bs}, BETA=0.1, RETAIN_WEIGHT=${rw} <<<"
        EPOCHS=3 LR=5e-05 BATCH_SIZE=$bs BETA=0.1 RETAIN_WEIGHT=$rw \
            ./unlearn/run_unlearn.sh npo
    done
done

echo "=========================================================="
echo "Sweep 5 completed successfully!"
echo "=========================================================="
