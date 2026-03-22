#!/usr/bin/env bash
# Sweep 4a: NPO-focused optimization sweep
#
# Context from sweep 4:
#   - SimNPO achieved best results with ep3_lr3e-05_bs32_b0.01_rw1.0_ml2048
#   - NPO needs similar systematic exploration to find its optimal configuration
#   - Supervisor recommendation: "32 steps at batch size of 32. Maybe we can also try some 16s?"
#
# This sweep explores:
#   1) NPO with batch sizes 32 and 16
#   2) Multiple epochs (2, 3, 4, 5) to find the sweet spot
#   3) Learning rate variations (2e-05, 3e-05, 5e-05)
#   4) Dataset sizes (1024, 2048, 4096) for optimal data efficiency
#   5) Retain weight variations (0.5, 0.75, 1.0) at best config

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1
export GRAD_CLIP=0.5

echo "=========================================================="
echo "Starting Sweep 4a: NPO Optimization"
echo "Model: $BASE"
echo "Focus: Finding optimal NPO configuration"
echo "=========================================================="

# ---------------------------------------------------------------
# Phase 1: Core NPO exploration with batch size 32
# Test different epochs and data sizes at standard lr=3e-05
# ---------------------------------------------------------------
echo -e "\n=== PHASE 1: NPO with batch size 32 ==="
export BATCH_SIZE=32

for max_lines in "1024" "2048" "4096"; do
    for ep in "2" "3" "4" "5"; do
        echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=${ep}, MAX_LINES=${max_lines}, BETA=0.01 <<<"
        MAX_LINES=$max_lines EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    done
done

# ---------------------------------------------------------------
# Phase 2: NPO with batch size 16 at promising configurations
# Test batch size 16 at the most promising epoch/data combinations
# ---------------------------------------------------------------
echo -e "\n=== PHASE 2: NPO with batch size 16 ==="
export BATCH_SIZE=16

# Focus on epochs 3-4 which typically work best
for max_lines in "1024" "2048" "4096"; do
    for ep in "3" "4"; do
        echo -e "\n>>> [npo] BS=16, LR=3e-05, EPOCHS=${ep}, MAX_LINES=${max_lines}, BETA=0.01 <<<"
        MAX_LINES=$max_lines EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    done
done

# ---------------------------------------------------------------
# Phase 3: Learning rate exploration at optimal data size
# Test different learning rates at ml=2048 (likely optimal based on SimNPO)
# ---------------------------------------------------------------
echo -e "\n=== PHASE 3: Learning rate exploration ==="
export BATCH_SIZE=32

# Lower learning rate
for ep in "3" "4"; do
    echo -e "\n>>> [npo] BS=32, LR=2e-05, EPOCHS=${ep}, MAX_LINES=2048, BETA=0.01 <<<"
    MAX_LINES=2048 EPOCHS=$ep LR=2e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

# Higher learning rate (be careful with stability)
echo -e "\n>>> [npo] BS=32, LR=5e-05, EPOCHS=3, MAX_LINES=2048, BETA=0.01 <<<"
MAX_LINES=2048 EPOCHS=3 LR=5e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo

# ---------------------------------------------------------------
# Phase 4: Retain weight exploration at best configuration
# Test retain_weight variations once we identify the best config
# ---------------------------------------------------------------
echo -e "\n=== PHASE 4: Retain weight exploration ==="
# Using likely optimal: ep3, lr3e-05, ml2048, bs32
export BATCH_SIZE=32

for rw in "0.5" "0.75"; do  # 1.0 already tested above
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=3, MAX_LINES=2048, RETAIN_WEIGHT=${rw} <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh npo
done

# ---------------------------------------------------------------
# Phase 5: Beta parameter exploration
# Test different beta values to optimize the preference optimization
# ---------------------------------------------------------------
echo -e "\n=== PHASE 5: Beta parameter exploration ==="
export BATCH_SIZE=32

for beta in "0.005" "0.02" "0.05"; do  # 0.01 already tested above
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=3, MAX_LINES=2048, BETA=${beta} <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=$beta RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
done

echo -e "\n=========================================================="
echo "NPO optimization sweep complete!"
echo "Check W&B to identify the optimal NPO configuration"
echo "Key metrics to compare:"
echo "  - Score (MMLU - WMDP Robust)"
echo "  - MMLU retention vs WMDP reduction trade-off"
echo "  - Training stability across different batch sizes"
echo "=========================================================="