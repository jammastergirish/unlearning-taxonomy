#!/usr/bin/env bash
# Sweep 4c: NPO capability-preservation sweep
#
# The problem: NPO best score is 0.0728 vs SimNPO's 0.1780.
# Both destroy WMDP similarly (~0.25), but NPO also kills MMLU (0.32 vs 0.43).
# Score = MMLU - WMDP (Robust), so NPO's MMLU collapse is the bottleneck.
#
# Root cause: the retain loss (with rw=1.0) is not strong enough to anchor
# general capability during the NPO forget gradient step.
#
# Strategy:
#   P1 – Stronger retain regularisation (rw=2.0, 3.0, 5.0)
#   P2 – Fewer epochs (ep=1, ep=2) to reduce cumulative forgetting
#   P3 – Higher beta (0.05, 0.1) — reference-model penalty keeps NPO conservative
#   P4 – Lower LR (1e-05) to slow down MMLU degradation
#   P5 – Gradient accumulation trick: effective BS=32 but smaller micro-batch

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1
export GRAD_CLIP=0.5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "Starting Sweep 4c: NPO Capability-Preservation Sweep"
echo "Model: $BASE"
echo "Goal:  Recover MMLU without sacrificing WMDP unlearning"
echo "Baseline NPO best: Score=0.0728, MMLU=0.3159, WMDP=0.2431"
echo "Target:            Score>0.12,  MMLU>0.38,  WMDP<0.28"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ---------------------------------------------------------------
# Priority 1: Stronger retain regularisation (rw)
# Hypothesis: rw=1.0 is too weak; bumping it should prevent MMLU
# collapse while still unlearning WMDP content.
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 1: Stronger retain weight (BS=32, EP=3, LR=3e-05, BETA=0.01) ==="
export BATCH_SIZE=32

for rw in "2.0" "3.0" "5.0"; do
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EP=3, ML=2048, BETA=0.01, RW=${rw} <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=0.01 RETAIN_WEIGHT=$rw ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 2: Fewer epochs
# Hypothesis: ep=3 already over-forgets; ep=1 and ep=2 may give a
# better trade-off between WMDP reduction and MMLU retention.
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 2: Fewer epochs (BS=32, LR=3e-05, BETA=0.01, RW=1.0) ==="
export BATCH_SIZE=32

for ep in "1" "2"; do
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EP=${ep}, ML=2048, BETA=0.01, RW=1.0 <<<"
    MAX_LINES=2048 EPOCHS=$ep LR=3e-05 BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 3: Higher beta (reference-model KL penalty)
# Hypothesis: larger beta tightens the KL constraint to the ref
# model, keeping general weights closer to the original.
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 3: Higher beta (BS=32, EP=3, LR=3e-05, RW=1.0) ==="
export BATCH_SIZE=32

for beta in "0.05" "0.1"; do
    echo -e "\n>>> [npo] BS=32, LR=3e-05, EP=3, ML=2048, BETA=${beta}, RW=1.0 <<<"
    MAX_LINES=2048 EPOCHS=3 LR=3e-05 BETA=$beta RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 4: Lower learning rate
# Hypothesis: 3e-05 is too large; 1e-05 (matching SimNPO's sweet
# spot territory) may reduce MMLU decay significantly.
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 4: Lower LR (BS=32, EP=3, BETA=0.01, RW=1.0) ==="
export BATCH_SIZE=32

for lr in "1e-05" "2e-05"; do
    echo -e "\n>>> [npo] BS=32, LR=${lr}, EP=3, ML=2048, BETA=0.01, RW=1.0 <<<"
    MAX_LINES=2048 EPOCHS=3 LR=$lr BETA=0.01 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh npo
    cleanup_memory
done

# ---------------------------------------------------------------
# Priority 5: Best combos from P1-P4
# Run the most promising pair combos once P1-P4 results land.
# Early hypothesis: higher rw + lower lr together.
# ---------------------------------------------------------------
echo -e "\n=== PRIORITY 5: Combination – lower LR + higher RW ==="
export BATCH_SIZE=32

echo -e "\n>>> [npo] BS=32, LR=1e-05, EP=3, ML=2048, BETA=0.01, RW=2.0 <<<"
MAX_LINES=2048 EPOCHS=3 LR=1e-05 BETA=0.01 RETAIN_WEIGHT=2.0 ./unlearn/run_unlearn.sh npo
cleanup_memory

echo -e "\n>>> [npo] BS=32, LR=1e-05, EP=2, ML=2048, BETA=0.01, RW=2.0 <<<"
MAX_LINES=2048 EPOCHS=2 LR=1e-05 BETA=0.01 RETAIN_WEIGHT=2.0 ./unlearn/run_unlearn.sh npo
cleanup_memory

echo -e "\n>>> [npo] BS=32, LR=2e-05, EP=3, ML=2048, BETA=0.05, RW=2.0 <<<"
MAX_LINES=2048 EPOCHS=3 LR=2e-05 BETA=0.05 RETAIN_WEIGHT=2.0 ./unlearn/run_unlearn.sh npo
cleanup_memory

echo -e "\n=========================================================="
echo "NPO capability-preservation sweep complete!"
echo "Total experiments: 13"
echo "Check W&B, looking for MMLU > 0.38 with WMDP < 0.28"
echo "=========================================================="
