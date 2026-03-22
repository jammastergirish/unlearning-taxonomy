#!/usr/bin/env bash
# sweep_unlearn_ga.sh: Hyperparameter sweep for ga and ga_simple methods.
#
# GA (Gradient Ascent) dynamics:
#   loss = -NLL(forget) + retain_weight * NLL(retain)
#
#   No reference model (unlike NPO/SimNPO), so it tends to be more aggressive.
#   The retain_weight is the primary lever for preventing MMLU collapse.
#
# GA Simple has NO retain loss at all — expect faster WMDP drop but also
# faster MMLU collapse, so we run it with fewer epochs and lower LR.
#
# Priority:
#   P1  ga:   retain_weight sweep          — most important axis
#   P2  ga:   epoch sweep                  — control cumulative forgetting
#   P3  ga:   LR sweep                     — coarse step size
#   P4  ga:   retain_weight + LR combos    — best-of cross
#   P5  ga_simple: epoch + LR sweep        — baseline, no retain term

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export MAX_LINES=2048
export NO_SAVE=1
export GRAD_CLIP=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "Starting sweep: GA / GA-Simple"
echo "Model: $BASE"
echo "Goal:  Find best MMLU–WMDP trade-off for gradient ascent"
echo "Score = MMLU - WMDP (Robust).  Base: 0.019 (MMLU=0.45, WMDP=0.43)"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ---------------------------------------------------------------
# Priority 1 — GA retain_weight sweep
# GA has no reference model, so retain_weight directly sets how
# hard the optimiser fights to preserve general capability.
# rw=1.0 is likely too weak (same issue as NPO); try 2-5.
# ---------------------------------------------------------------
echo -e "\n=== P1: GA retain_weight sweep (EP=3, LR=3e-05, BS=32) ==="
BATCH_SIZE=32

for rw in "1.0" "2.0" "3.0" "5.0"; do
    echo -e "\n>>> [ga] EP=3, LR=3e-05, BS=32, ML=2048, RW=${rw} <<<"
    BATCH_SIZE=$BATCH_SIZE EPOCHS=3 LR=3e-05 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh ga
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 2 — GA epoch sweep
# Fewer epochs = less cumulative forgetting.
# Use retain_weight=1.0 to isolate the epoch effect first.
# ---------------------------------------------------------------
echo -e "\n=== P2: GA epoch sweep (LR=3e-05, BS=32, RW=1.0) ==="

for ep in "1" "2" "4"; do
    echo -e "\n>>> [ga] EP=${ep}, LR=3e-05, BS=32, ML=2048, RW=1.0 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh ga
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 3 — GA LR sweep
# Lower LR → gentler ascent per step → less MMLU damage.
# ---------------------------------------------------------------
echo -e "\n=== P3: GA LR sweep (EP=3, BS=32, RW=1.0) ==="

for lr in "1e-05" "2e-05" "5e-05"; do
    echo -e "\n>>> [ga] EP=3, LR=${lr}, BS=32, ML=2048, RW=1.0 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=$lr RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh ga
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 4 — GA best combos
# Cross best axes from P1–P3.
# ---------------------------------------------------------------
echo -e "\n=== P4: GA combination runs ==="

echo -e "\n>>> [ga] EP=3, LR=1e-05, BS=32, ML=2048, RW=2.0 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 RETAIN_WEIGHT=2.0 ./unlearn/run_unlearn.sh ga
cleanup_memory

echo -e "\n>>> [ga] EP=2, LR=3e-05, BS=32, ML=2048, RW=2.0 <<<"
BATCH_SIZE=32 EPOCHS=2 LR=3e-05 RETAIN_WEIGHT=2.0 ./unlearn/run_unlearn.sh ga
cleanup_memory

echo -e "\n>>> [ga] EP=3, LR=2e-05, BS=32, ML=2048, RW=3.0 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=2e-05 RETAIN_WEIGHT=3.0 ./unlearn/run_unlearn.sh ga
cleanup_memory

# ---------------------------------------------------------------
# Priority 5 — GA Simple (no retain loss)
# ga_simple only takes: epochs, lr, batch_size, max_lines.
# Expected: more aggressive unlearning but bigger MMLU hit.
# Use low LR + few epochs to keep it from completely collapsing.
# ---------------------------------------------------------------
echo -e "\n=== P5: GA Simple (no retain loss) ==="

for ep in "1" "2"; do
    for lr in "1e-05" "3e-05"; do
        echo -e "\n>>> [ga_simple] EP=${ep}, LR=${lr}, BS=32, ML=2048 <<<"
        BATCH_SIZE=32 EPOCHS=$ep LR=$lr ./unlearn/run_unlearn.sh ga_simple
        cleanup_memory
    done
done

echo -e "\n=========================================================="
echo "GA sweep complete!  Total experiments: 18"
echo "Check W&B, looking for Score > 0.10 (MMLU > 0.38, WMDP < 0.30)"
echo "=========================================================="
