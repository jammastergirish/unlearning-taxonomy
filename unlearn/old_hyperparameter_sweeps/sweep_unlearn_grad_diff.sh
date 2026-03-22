#!/usr/bin/env bash
# sweep_unlearn_grad_diff.sh: Sweep for grad_diff, dpo, and lat methods.
#
# GRAD_DIFF
#   loss = forget_weight * (-NLL_forget) + NLL_retain
#   Parameters: epochs, lr, batch_size, forget_weight, max_lines
#   forget_weight scales how hard you push against forget data.
#
# DPO
#   loss = -log σ(β(log π(forget)/π_ref(forget) – log π(retain)/π_ref(retain)))
#   Parameters: epochs, lr, batch_size, beta, max_lines
#   beta controls how tightly the policy is anchored to the reference model.
#
# LAT (Latent Adversarial Training)
#   Inner loop: adversarially perturb hidden states on forget data
#   Outer loop: retain loss to preserve general capability
#   Parameters: epochs, lr, batch_size, lat_eps, lat_steps, retain_weight, layer_id, max_lines

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
echo "Starting sweep: grad_diff / DPO / LAT"
echo "Model: $BASE"
echo "Base model: Score=0.019  (MMLU=0.45, WMDP=0.43)"
echo "Target:     Score>0.12   (MMLU>0.38, WMDP<0.28)"
echo "=========================================================="

cleanup_memory() {
    echo "Cleaning up GPU memory..."
    python -c "import torch; import gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 2
}

# ===============================================================
# GRAD_DIFF
# loss = forget_weight * (−NLL_forget) + NLL_retain
# forget_weight is the only extra knob vs. plain GA.
# Higher → more aggressive forget push.  Start conservative.
# ===============================================================
echo -e "\n=== GRAD_DIFF: forget_weight sweep (EP=3, LR=3e-05, BS=32) ==="

for fw in "0.5" "1.0" "2.0" "5.0"; do
    echo -e "\n>>> [grad_diff] EP=3, LR=3e-05, BS=32, ML=2048, FW=${fw} <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 FORGET_WEIGHT=$fw ./unlearn/run_unlearn.sh grad_diff
    cleanup_memory
done

echo -e "\n=== GRAD_DIFF: epoch + LR combos (FW=1.0) ==="
for ep in "1" "2"; do
    echo -e "\n>>> [grad_diff] EP=${ep}, LR=3e-05, BS=32, FW=1.0 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 FORGET_WEIGHT=1.0 ./unlearn/run_unlearn.sh grad_diff
    cleanup_memory
done
echo -e "\n>>> [grad_diff] EP=3, LR=1e-05, BS=32, FW=1.0 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 FORGET_WEIGHT=1.0 ./unlearn/run_unlearn.sh grad_diff
cleanup_memory

# ===============================================================
# DPO
# A preference-learning approach: treat forget data as "rejected"
# and retain data as "chosen", anchored by beta.
# beta default is 0.1 — but NPO/SimNPO work better at 0.01, so
# sweep a wider range here.
# ===============================================================
echo -e "\n=== DPO: beta sweep (EP=3, LR=3e-05, BS=32) ==="

for beta in "0.01" "0.05" "0.1" "0.5"; do
    echo -e "\n>>> [dpo] EP=3, LR=3e-05, BS=32, ML=2048, BETA=${beta} <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 BETA=$beta ./unlearn/run_unlearn.sh dpo
    cleanup_memory
done

echo -e "\n=== DPO: epoch sweep (BETA=0.05, LR=3e-05, BS=32) ==="
for ep in "1" "2" "5"; do
    echo -e "\n>>> [dpo] EP=${ep}, LR=3e-05, BS=32, ML=2048, BETA=0.05 <<<"
    BATCH_SIZE=32 EPOCHS=$ep LR=3e-05 BETA=0.05 ./unlearn/run_unlearn.sh dpo
    cleanup_memory
done

echo -e "\n>>> [dpo] EP=3, LR=1e-05, BS=32, BETA=0.05 <<<"
BATCH_SIZE=32 EPOCHS=3 LR=1e-05 BETA=0.05 ./unlearn/run_unlearn.sh dpo
cleanup_memory

# ===============================================================
# LAT (Latent Adversarial Training)
# cb_lat has been swept extensively; plain lat hasn't.
# lat only does the adversarial perturbation + retain loss (no CB
# rerouting), so it's lighter and may generalise differently.
# Key axes: lat_eps, retain_weight, layer_id.
# ===============================================================
echo -e "\n=== LAT: lat_eps sweep (EP=3, LR=3e-05, BS=32, RW=1.0, LS=5, LY=5,6,7) ==="

for eps in "0.05" "0.1" "0.2" "0.5"; do
    echo -e "\n>>> [lat] EP=3, LR=3e-05, BS=32, ML=2048, LE=${eps}, LS=5, RW=1.0, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 LAT_EPS=$eps LAT_STEPS=5 RETAIN_WEIGHT=1.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh lat
    cleanup_memory
done

echo -e "\n=== LAT: retain_weight sweep (EP=3, LR=3e-05, LE=0.1, LS=5, LY=5,6,7) ==="
for rw in "2.0" "3.0" "5.0"; do
    echo -e "\n>>> [lat] EP=3, LR=3e-05, BS=32, LE=0.1, LS=5, RW=${rw}, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 LAT_EPS=0.1 LAT_STEPS=5 RETAIN_WEIGHT=$rw LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh lat
    cleanup_memory
done

echo -e "\n=== LAT: lat_steps sweep (EP=3, LR=3e-05, LE=0.1, RW=1.0, LY=5,6,7) ==="
for ls in "2" "10"; do
    echo -e "\n>>> [lat] EP=3, LR=3e-05, BS=32, LE=0.1, LS=${ls}, RW=1.0, LY=5,6,7 <<<"
    BATCH_SIZE=32 EPOCHS=3 LR=3e-05 LAT_EPS=0.1 LAT_STEPS=$ls RETAIN_WEIGHT=1.0 LAYER_ID="5,6,7" \
        ./unlearn/run_unlearn.sh lat
    cleanup_memory
done

echo -e "\n=========================================================="
echo "grad_diff / DPO / LAT sweep complete!  Total experiments: ~22"
echo "Check W&B for Score = MMLU - WMDP (Robust)"
echo "=========================================================="
