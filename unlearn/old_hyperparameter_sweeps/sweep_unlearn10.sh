#!/usr/bin/env bash
# Tenth sweep: TAR (Task Arithmetic Removal) with systematic data scaling.
#
# TAR subtracts a scaled "forget task vector" from the base model weights:
#   θ_unlearned = θ_base - α * (θ_forget_ft − θ_base)
#
# INTEGRATED DATA SCALING: Following supervisor's methodology, each promising
# hyperparameter config is tested at multiple dataset sizes (1024, 2048, 4096)
# to find optimal data/compute trade-off before exploring parameter variations.
#
# TAR has no retain loss during the fine-tuning phase — it fine-tunes on
# forget data first, then negates that direction.  The three knobs are:
#   tar_alpha  — how strongly to negate the forget direction (default 1.0)
#   tar_lr     — learning rate for the forget fine-tuning phase
#   tar_epochs — number of epochs for the forget fine-tuning phase
#
# Mac notes (64 GB unified memory):
#   - device=mps / dtype=auto will run on Apple Silicon GPU.
#   - NO_SAVE=1 avoids writing multi-GB model weights to disk.
#   - W&B logging is handled inside unlearn.py via init_wandb().

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export BATCH_SIZE=32    # Supervisor's recommendation for cleaner gradient signals
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1        # Don't save model weights — W&B has all the metrics we need

# Data scaling will be systematic per config below (1024, 2048, 4096)

echo "=========================================================="
echo "Starting TENTH hyperparameter sweep"
echo "Method: TAR (Task Arithmetic Removal)"
echo "Model:  $BASE"
echo "Device: $DEVICE"
echo "=========================================================="

METHOD="tar"

# ---------------------------------------------------------------
# 1. Systematic Data Scaling for Baseline Config
# Start with default TAR config and find optimal dataset size
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Data scaling sweep (alpha=1.0, lr=1e-5, epochs=1) <<<"
for max_lines in "1024" "2048" "4096"; do
    echo -e "    Testing dataset size: ${max_lines} samples"
    MAX_LINES=$max_lines TAR_ALPHA=1.0 TAR_LR=1e-5 TAR_EPOCHS=1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 2. LR Sweep  (baseline alpha=1.0, epochs=1)
# A higher forget-FT LR builds a stronger task vector per epoch.
# A lower LR may produce a cleaner, more targeted vector.
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# 2. Alpha Sweep at Optimal Dataset Size
# Test different alpha values at the dataset size that worked best above
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Alpha sweep at optimal dataset size <<<"
for max_lines in "2048"; do  # Adjust based on results from step 1
    for alpha in "0.5" "1.0" "2.0" "4.0"; do
        echo -e "    Alpha=${alpha} at ${max_lines} samples"
        MAX_LINES=$max_lines TAR_ALPHA=$alpha TAR_LR=1e-5 TAR_EPOCHS=1 \
            ./unlearn/run_unlearn.sh $METHOD
    done
done

# ---------------------------------------------------------------
# 3. LR Sweep at Optimal Dataset Size
# Test different learning rates at optimal dataset size
# ---------------------------------------------------------------
echo -e "\n>>> [tar] LR sweep at optimal dataset size <<<"
for max_lines in "2048"; do  # Use optimal size from step 1
    for lr in "5e-6" "1e-5" "3e-5"; do
        echo -e "    LR=${lr} at ${max_lines} samples"
        MAX_LINES=$max_lines TAR_ALPHA=1.0 TAR_LR=$lr TAR_EPOCHS=1 \
            ./unlearn/run_unlearn.sh $METHOD
    done
done

# ---------------------------------------------------------------
# 4. Epoch Sweep for Best Config
# Test more epochs for the most promising alpha/lr combination
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Epoch sweep for best config <<<"
for max_lines in "2048"; do  # Use optimal size
    for ep in "2" "3"; do
        echo -e "    Epochs=${ep} at ${max_lines} samples"
        MAX_LINES=$max_lines TAR_ALPHA=2.0 TAR_LR=1e-5 TAR_EPOCHS=$ep \
            ./unlearn/run_unlearn.sh $METHOD
    done
done

echo "=========================================================="
echo "Sweep 10 (TAR) completed successfully!"
echo "=========================================================="
