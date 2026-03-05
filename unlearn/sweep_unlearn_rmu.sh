#!/usr/bin/env bash
# sweep_unlearn_rmu.sh: Sweep for RMU and CB methods.
#
# Both methods push forget-set hidden-state activations toward a random unit
# vector while preserving retain-set activations at the original layer outputs.
#
#   RMU loss = –cos_sim(forget_act, random_target) + alpha * (1 – cos_sim(retain_act, retain_cache))
#   CB  loss = same formula but with an extra steering_coeff multiplier on the forget term
#
# Key parameters:
#   --alpha          : retain-side preservation strength (default 100)
#   --steering-coeff : forget-side push magnitude for CB    (default 20)
#   --layer-id       : comma-separated layer indices to target (default "5,6,7")
#   --epochs / --lr  : standard training knobs
#
# Strategy:
#   P1  rmu: alpha sweep            — primary lever for MMLU preservation
#   P2  rmu: epoch + LR sweep       — convergence and step-size effects
#   P3  rmu: layer-id sweep         — target earlier vs later layers
#   P4  cb:  steering_coeff sweep   — CB's extra magnitude knob
#   P5  cb:  alpha + epoch combos   — borrow RMU insights into CB

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
echo "Starting sweep: RMU / CB"
echo "Model: $BASE"
echo "Goal:  Find best Score = MMLU - WMDP (Robust)"
echo "Base model: Score=0.019  (MMLU=0.45, WMDP=0.43)"
echo "SimNPO best for ref: Score=0.178 (MMLU=0.43, WMDP=0.25)"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ---------------------------------------------------------------
# P1 — RMU alpha sweep
# alpha controls how strongly the retain-side activations are
# anchored.  Too low → MMLU collapses.  Too high → no unlearning.
# Default is 100; cb_lat results suggest 50 can be better.
# ---------------------------------------------------------------
echo -e "\n=== P1: RMU alpha sweep (EP=3, LR=3e-05, BS=32, SC=20, LY=5,6,7) ==="

for alpha in "50.0" "100.0" "200.0" "500.0"; do
    echo -e "\n>>> [rmu] EP=3, LR=3e-05, BS=32, ML=2048, ALPHA=${alpha}, SC=20, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=$alpha STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

# ---------------------------------------------------------------
# P2 — RMU epoch + LR sweep
# Fewer epochs and lower LR → less aggressive unlearning but less
# MMLU damage.  Using alpha=100 (default) to isolate this axis.
# ---------------------------------------------------------------
echo -e "\n=== P2: RMU epoch sweep (ALPHA=100, BS=32, SC=20, LY=5,6,7) ==="

for ep in "1" "2" "5"; do
    echo -e "\n>>> [rmu] EP=${ep}, LR=3e-05, BS=32, ML=2048, ALPHA=100, SC=20 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

echo -e "\n>>> [rmu] EP=3, LR=1e-05, BS=32, ML=2048, ALPHA=100, SC=20 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
    ./unlearn/run_unlearn.sh rmu
cleanup_memory

# ---------------------------------------------------------------
# P3 — RMU layer-id sweep
# Targeting different sets of layers changes which knowledge is
# disrupted.  Middle layers often work best (5-7 of ~32).
# ---------------------------------------------------------------
echo -e "\n=== P3: RMU layer-id sweep (EP=3, LR=3e-05, BS=32, ALPHA=100, SC=20) ==="

for layers in "3,4,5" "7,8,9" "10,11,12" "5,6,7,8"; do
    echo -e "\n>>> [rmu] LY=${layers} <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="$layers" \
        ./unlearn/run_unlearn.sh rmu
    cleanup_memory
done

# ---------------------------------------------------------------
# P4 — CB steering_coeff sweep
# CB adds a steering_coeff multiplier on the forget-side cosine
# loss.  Higher → harder push away from original activations.
# Use cb_lat's best config as a starting point (alpha=50, ep=3).
# ---------------------------------------------------------------
echo -e "\n=== P4: CB steering_coeff sweep (EP=3, LR=3e-05, BS=32, ALPHA=50, LY=5,6,7) ==="

for sc in "10.0" "20.0" "50.0" "100.0"; do
    echo -e "\n>>> [cb] EP=3, LR=3e-05, BS=32, ML=2048, ALPHA=50, SC=${sc}, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=50.0 STEERING_COEFF=$sc LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh cb
    cleanup_memory
done

# ---------------------------------------------------------------
# P5 — CB best combos
# Cross the best alpha from P1 and best sc from P4.
# ---------------------------------------------------------------
echo -e "\n=== P5: CB combination runs ==="

echo -e "\n>>> [cb] EP=3, LR=3e-05, BS=32, ALPHA=200, SC=20, LY=5,6,7 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=3e-05 ALPHA=200.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
    ./unlearn/run_unlearn.sh cb
cleanup_memory

echo -e "\n>>> [cb] EP=3, LR=1e-05, BS=32, ALPHA=100, SC=20, LY=5,6,7 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 ALPHA=100.0 STEERING_COEFF=20.0 LAYER_ID="5,6,7" \
    ./unlearn/run_unlearn.sh cb
cleanup_memory

echo -e "\n=========================================================="
echo "RMU / CB sweep complete!  Total experiments: 17"
echo "Check W&B for Score = MMLU - WMDP (Robust)"
echo "=========================================================="
