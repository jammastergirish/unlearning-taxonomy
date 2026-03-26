#!/usr/bin/env bash
# Sweep 4b: Memory-optimized NPO sweep with focused experiments
#
# Addressing OOM issues from 4a by:
#   - Reducing number of concurrent experiments
#   - Adding memory cleanup between runs
#   - Focusing on most promising configurations first
#   - Running in smaller batches
#
# Priority based on supervisor guidance: "32 steps at batch size of 32. Maybe we can also try some 16s?"

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1
export GRAD_CLIP=0.5
# Add memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "Starting Sweep 4b: Memory-Optimized NPO Sweep"
echo "Model: $BASE"
echo "Focus: NPO optimization with memory management"
echo "=========================================================="

# Helper function to cleanup memory between runs
cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ---------------------------------------------------------------
# Priority 1: Most promising NPO configs based on SimNPO results
# Focus on ml=2048 which worked best for SimNPO
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 1: Core NPO with BS=32, ML=2048 ==="
export BATCH_SIZE=32

for ep in "3" "4"; do
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=${ep}, MAX_LINES=2048, BETA=0.01 <<<"
    MAX_LINES=2048 EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 2: Test batch size 16 as suggested
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 2: NPO with BS=16, ML=2048 ==="
export BATCH_SIZE=16

for ep in "3" "4"; do
    echo -e "\n>>> [npo] BS=16, LR=3e-05, EPOCHS=${ep}, MAX_LINES=2048, BETA=0.01 <<<"
    MAX_LINES=2048 EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 3: Test other dataset sizes with best epoch
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 3: Dataset size exploration with EP=3 ==="
export BATCH_SIZE=32

for max_lines in "1024" "4096"; do  # 2048 already tested
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=3, MAX_LINES=${max_lines}, BETA=0.01 <<<"
    MAX_LINES=$max_lines EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 4: Learning rate fine-tuning
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 4: Learning rate exploration ==="
export BATCH_SIZE=32

# Lower LR
echo -e "\n>>> [npo] BS=32, LR=2e-05, EPOCHS=3, MAX_LINES=2048, BETA=0.01 <<<"
MAX_LINES=2048 EPOCHS=3 LR=2e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
cleanup_memory

# Slightly higher LR (but not too high to avoid instability)
echo -e "\n>>> [npo] BS=32, LR=4e-05, EPOCHS=3, MAX_LINES=2048, BETA=0.01 <<<"
MAX_LINES=2048 EPOCHS=3 LR=4e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
cleanup_memory

# ---------------------------------------------------------------
# Priority 5: Retain weight exploration
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 5: Retain weight ablation ==="
export BATCH_SIZE=32

for rw in "0.75" "0.5"; do  # 1.0 already tested
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=3, MAX_LINES=2048, RETAIN_WEIGHT=${rw} <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 6: Beta fine-tuning (if memory permits)
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 6: Beta parameter tuning ==="
export BATCH_SIZE=32

for beta in "0.005" "0.02"; do  # 0.01 already tested
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EPOCHS=3, MAX_LINES=2048, BETA=${beta} <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=$beta RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

echo -e "\n=========================================================="
echo "Memory-optimized NPO sweep complete!"
echo "Total experiments: ~14 (vs 30+ in 4a)"
echo "Check W&B for results and identify optimal configuration"
echo "=========================================================="