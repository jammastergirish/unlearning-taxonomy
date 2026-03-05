#!/usr/bin/env bash
# sweep_unlearn_wt.sh: Sweep for wt_dist and wt_dist_reg methods.
#
# WT_DIST  (Weight Distortion)
#   1. Add iid Gaussian noise to ALL parameters (std=wt_noise_std)
#   2. Fine-tune on retain set only.
#   The noise disrupts forget-set knowledge; retain fine-tuning recovers
#   general capability.  wt_noise_std is the only real knob.
#
# WT_DIST_REG  (Weight Distance Regularization)
#   loss = NLL_retain − wt_reg_lambda * ||θ − θ_pretrained||²
#   Pushes weights AWAY from pretrained values while staying useful on
#   retain data.  wt_reg_lambda controls the repulsion strength.

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
echo "Starting sweep: wt_dist / wt_dist_reg"
echo "Model: $BASE"
echo "Base model: Score=0.019  (MMLU=0.45, WMDP=0.43)"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ===============================================================
# WT_DIST
# Noise σ determines how much is erased before retain fine-tune.
# Too small → nothing forgotten.  Too large → model collapses.
# Fine-tuning epochs/LR on the retain side then controls recovery.
# ===============================================================
echo -e "\n=== WT_DIST: noise_std sweep (EP=3, LR=3e-05, BS=32) ==="

for std in "0.005" "0.01" "0.02" "0.05" "0.1"; do
    echo -e "\n>>> [wt_dist] WN=${std}, EP=3, LR=3e-05, BS=32, ML=2048 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 WT_NOISE_STD=$std ./unlearn/run_unlearn.sh wt_dist
    cleanup_memory
done

# Retain epoch sweep at the best-looking std (use 0.02 as default)
echo -e "\n=== WT_DIST: epoch sweep at WN=0.02 (LR=3e-05, BS=32) ==="
for ep in "1" "2" "5"; do
    echo -e "\n>>> [wt_dist] WN=0.02, EP=${ep}, LR=3e-05, BS=32, ML=2048 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 WT_NOISE_STD=0.02 ./unlearn/run_unlearn.sh wt_dist
    cleanup_memory
done

# LR sweep at WN=0.02, EP=3
echo -e "\n>>> [wt_dist] WN=0.02, EP=3, LR=1e-05, BS=32 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 WT_NOISE_STD=0.02 ./unlearn/run_unlearn.sh wt_dist
cleanup_memory

# ===============================================================
# WT_DIST_REG
# lambda=0 → pure retain fine-tune, no weight repulsion.
# lambda too high → retain NLL dominates over repulsion → similar risk.
# Actually: large lambda → LARGE repulsion, MMLU may collapse.
# Sweep conservatively: 0.01–1.0.
# ===============================================================
echo -e "\n=== WT_DIST_REG: lambda sweep (EP=3, LR=3e-05, BS=32) ==="

for lam in "0.01" "0.05" "0.1" "0.5" "1.0"; do
    echo -e "\n>>> [wt_dist_reg] WR=${lam}, EP=3, LR=3e-05, BS=32, ML=2048 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 WT_REG_LAMBDA=$lam ./unlearn/run_unlearn.sh wt_dist_reg
    cleanup_memory
done

echo -e "\n=== WT_DIST_REG: epoch sweep at WR=0.1 (LR=3e-05, BS=32) ==="
for ep in "1" "2" "5"; do
    echo -e "\n>>> [wt_dist_reg] WR=0.1, EP=${ep}, LR=3e-05, BS=32, ML=2048 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 WT_REG_LAMBDA=0.1 ./unlearn/run_unlearn.sh wt_dist_reg
    cleanup_memory
done

echo -e "\n>>> [wt_dist_reg] WR=0.1, EP=3, LR=1e-05, BS=32 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 WT_REG_LAMBDA=0.1 ./unlearn/run_unlearn.sh wt_dist_reg
cleanup_memory

echo -e "\n=========================================================="
echo "wt_dist / wt_dist_reg sweep complete!  Total experiments: 17"
echo "Check W&B for Score = MMLU - WMDP (Robust)"
echo "=========================================================="
