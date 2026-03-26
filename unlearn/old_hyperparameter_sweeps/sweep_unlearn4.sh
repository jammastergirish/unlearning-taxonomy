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
# Goals - SYSTEMATIC DATA SCALING INTEGRATED:
# Each hyperparameter config is tested at multiple dataset sizes (1024, 2048, 4096)
# to find the optimal data/compute trade-off, following supervisor's methodology:
# "32 steps at batch size 32 = 1024, then double until results plateau"
#
# Core configs to test:
#   1) SimNPO/NPO epoch sweep at lr3e-05 (ep4, ep5)
#   2) SimNPO/NPO intermediate LR 5e-05
#   3) DPO at lr3e-05 with safer beta values (0.01, 0.1)
#   4) Best CB/CB_LAT config

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export BATCH_SIZE=32    # Supervisor's recommendation for cleaner gradient signals
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1
export GRAD_CLIP=0.5

# Data scaling will be done per-config below (1024, 2048, 4096)

LAYER_ID_BROAD="5,10,15,20,25,30"

echo "=========================================================="
echo "Starting FOURTH hyperparameter sweep (STABILIZED)"
echo "Fixed gradient explosion issues with safer parameters"
echo "Model: $BASE"
echo "=========================================================="

# ---------------------------------------------------------------
# 1. SimNPO: best config with systematic data scaling
# ---------------------------------------------------------------
for max_lines in "1024" "2048" "4096"; do
    for ep in "3" "4"; do  # Focus on most promising epochs
        echo -e "\n>>> [simnpo] LR=3e-05, EPOCHS=${ep}, MAX_LINES=${max_lines}, BETA=0.01 <<<"
        MAX_LINES=$max_lines EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo
    done
done

# ---------------------------------------------------------------
# 2. NPO: best config with systematic data scaling
# ---------------------------------------------------------------
for max_lines in "1024" "2048" "4096"; do
    for ep in "3" "4"; do  # Focus on most promising epochs
        echo -e "\n>>> [npo] LR=3e-05, EPOCHS=${ep}, MAX_LINES=${max_lines}, BETA=0.01 <<<"
        MAX_LINES=$max_lines EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    done
done

# ---------------------------------------------------------------
# 3. Higher LR exploration at optimal dataset size only
#    Test lr=5e-05 at the dataset size that performed best above
# ---------------------------------------------------------------
for max_lines in "2048"; do  # Assume 2048 will be optimal, adjust based on results
    echo -e "\n>>> [simnpo] LR=5e-05, EPOCHS=3, MAX_LINES=${max_lines}, BETA=0.01 <<<"
    MAX_LINES=$max_lines EPOCHS=3 LR=5e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo

    echo -e "\n>>> [npo] LR=5e-05, EPOCHS=3, MAX_LINES=${max_lines}, BETA=0.01 <<<"
    MAX_LINES=$max_lines EPOCHS=3 LR=5e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# 4. DPO: test best beta at optimal dataset size
# ---------------------------------------------------------------
for max_lines in "2048"; do  # Use promising dataset size from above
    echo -e "\n>>> [dpo] LR=3e-05, EPOCHS=3, MAX_LINES=${max_lines}, BETA=0.01 <<<"
    MAX_LINES=$max_lines EPOCHS=3 LR=3e-05 BETA=0.01 ./unlearn/run_unlearn.sh dpo
done

echo -e "\n=========================================================="
echo "Data scaling analysis complete!"
echo "Check W&B to identify optimal dataset size, then run:"
echo "  - Retain-weight ablation at optimal size"
echo "  - CB/CB_LAT experiments at optimal size"
echo "=========================================================="

echo "=========================================================="
echo "Sweep 4 completed successfully!"
echo "=========================================================="
