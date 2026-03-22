#!/usr/bin/env bash
# sweep_unlearn_rmu2.sh — RMU-only follow-up sweep.
#
# Background
# ----------
# The first RMU run (sweep_unlearn_rmu.sh) used:
#   alpha=50, steering_coeff=20, layers=5,6,7, ep=3, lr=3e-05
# Result: Score=-0.038 (MMLU=0.2295, WMDP=0.2673)
# MMLU collapsed catastrophically — the model lost general capability,
# not just the bio-risk knowledge we wanted to erase.
#
# Root causes to address:
#   1. alpha=50 is too LOW  → retain-side anchor was too weak, general activations drifted
#   2. SC=20 is borderline  → misdirection push may have been too large for this model size
#   3. 3 epochs might be too many at lr=3e-05 for this scale
#
# Target: Score ≥ 0.10 (comparable to best NPO/SimNPO runs)
# Best simnpo for reference: Score=0.178 (MMLU=0.4315, WMDP=0.2535)
#
# Strategy
# ---------
# R1  alpha sweep (SC=20, EP=3, LR=3e-05, LY=5,6,7)
#     → Find the minimum alpha that preserves MMLU ≥ 0.35
#     → Original paper uses alpha=1200; try 100–2000
#
# R2  steering_coeff sweep  (best alpha from R1, EP=3, LY=5,6,7)
#     → Lower SC = subtler misdirection = less MMLU damage
#
# R3  epoch + LR sweep  (best alpha+SC from R2)
#     → 1–2 epochs often sufficient for activation-space methods
#
# R4  layer-id sweep  (best config from R3)
#     → Shallower layers encode less task-specific knowledge;
#       later layers are closer to the output but harder to anchor

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export MAX_LINES=2048
export NO_SAVE=1
export GRAD_CLIP=1.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "Starting sweep: RMU (round 2)"
echo "Model: $BASE"
echo "Goal:  Score = MMLU - WMDP (Robust) ≥ 0.10"
echo "Prior best RMU: Score=-0.038  (MMLU=0.23, WMDP=0.27)"
echo "SimNPO target:  Score=+0.178  (MMLU=0.43, WMDP=0.25)"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ---------------------------------------------------------------
# R1 — alpha sweep
# The original RMU paper uses alpha≈1200 for a ~7B model.
# Our prior alpha=50 was far too low → drastically underanchored.
# Start from 100 and push up to 2000.
# ---------------------------------------------------------------
echo -e "\n=== R1: RMU alpha sweep (SC=20, EP=3, LR=3e-05, LY=5,6,7) ==="

for alpha in "100.0" "300.0" "600.0" "1200.0" "2000.0"; do
    echo -e "\n>>> [rmu] ALPHA=${alpha}, SC=20, EP=3, LR=3e-05, BS=32, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=$alpha STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

# ---------------------------------------------------------------
# R2 — steering_coeff sweep
# SC controls the magnitude of the random-direction target the
# forget activations are pushed toward.  Smaller SC = softer push.
# Use alpha=600 as a mid-range starting point (refine after R1).
# ---------------------------------------------------------------
echo -e "\n=== R2: RMU steering_coeff sweep (ALPHA=600, EP=3, LR=3e-05, LY=5,6,7) ==="

for sc in "5.0" "10.0" "30.0" "50.0"; do
    echo -e "\n>>> [rmu] SC=${sc}, ALPHA=600, EP=3, LR=3e-05, BS=32, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=600.0 STEERING_COEFF=$sc LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

# ---------------------------------------------------------------
# R3 — epoch + LR sweep
# Fewer passes = less cumulative damage to retain-side activations.
# Lower LR = finer moves in weight space, easier for alpha anchor.
# Use alpha=600, SC=20 from R1/R2 defaults here for isolation.
# ---------------------------------------------------------------
echo -e "\n=== R3: RMU epoch + LR sweep (ALPHA=600, SC=20, LY=5,6,7) ==="

for ep in "1" "2"; do
    echo -e "\n>>> [rmu] EP=${ep}, LR=3e-05, BS=32, ALPHA=600, SC=20, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 ALPHA=600.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

for lr in "1e-05" "5e-05"; do
    echo -e "\n>>> [rmu] EP=3, LR=${lr}, BS=32, ALPHA=600, SC=20, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=$lr ALPHA=600.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

# ---------------------------------------------------------------
# R4 — layer-id sweep
# Earlier layers (3-5) encode syntactic/broad knowledge; mid-layers
# (5-7) encode factual recall; final layers (8-12) encode surface
# output form.  Targeting 3-5 is more surgical for factual erasure.
# ---------------------------------------------------------------
echo -e "\n=== R4: RMU layer-id sweep (ALPHA=600, SC=20, EP=3, LR=3e-05) ==="

for layers in "3,4,5" "4,5,6" "7,8,9" "5,6,7,8,9"; do
    echo -e "\n>>> [rmu] LY=${layers}, ALPHA=600, SC=20, EP=3, LR=3e-05, BS=32 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=600.0 STEERING_COEFF=20.0 LAYER_ID="$layers" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

echo -e "\n=========================================================="
echo "RMU sweep 2 complete!  Total experiments: 15"
echo "Check W&B for Score = MMLU - WMDP (Robust)"
echo "=========================================================="
