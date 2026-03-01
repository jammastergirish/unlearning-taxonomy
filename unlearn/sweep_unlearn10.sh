#!/usr/bin/env bash
# Tenth sweep: First systematic sweep of TAR (Task Arithmetic Removal).
#
# TAR subtracts a scaled "forget task vector" from the base model weights:
#   θ_unlearned = θ_base - α * (θ_forget_ft − θ_base)
#
# TAR has no retain loss during the fine-tuning phase — it fine-tunes on
# forget data first, then negates that direction.  The three knobs are:
#   tar_alpha  — how strongly to negate the forget direction (default 1.0)
#   tar_lr     — learning rate for the forget fine-tuning phase
#   tar_epochs — number of epochs for the forget fine-tuning phase
#
# This sweep explores:
#   1. Alpha scaling:   does a larger α push WMDP lower, or does it wreck MMLU?
#   2. LR scaling:      more aggressive forget-FT = stronger task vector
#   3. Epoch scaling:   more forget-FT = larger task vector magnitude
#   4. Alpha × LR:      joint grid to find the interaction sweet spot
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
export EVAL_SPLIT=0.1
export MAX_LENGTH=512
export NO_SAVE=1        # Don't save model weights — W&B has all the metrics we need

echo "=========================================================="
echo "Starting TENTH hyperparameter sweep"
echo "Method: TAR (Task Arithmetic Removal)"
echo "Model:  $BASE"
echo "Device: $DEVICE"
echo "=========================================================="

METHOD="tar"

# ---------------------------------------------------------------
# 1. Alpha Sweep  (baseline lr=1e-5, epochs=1)
# Default alpha=1.0 exactly cancels the fine-tuned direction.
# Larger α over-corrects (stronger forgetting, risk of MMLU collapse).
# Smaller α under-corrects (MMLU preserved but WMDP may stay high).
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Alpha sweep (lr=1e-5, epochs=1) <<<"
for alpha in "0.5" "1.0" "2.0" "4.0" "8.0"; do
    TAR_ALPHA=$alpha TAR_LR=1e-5 TAR_EPOCHS=1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 2. LR Sweep  (baseline alpha=1.0, epochs=1)
# A higher forget-FT LR builds a stronger task vector per epoch.
# A lower LR may produce a cleaner, more targeted vector.
# ---------------------------------------------------------------
echo -e "\n>>> [tar] LR sweep (alpha=1.0, epochs=1) <<<"
for lr in "5e-6" "1e-5" "3e-5" "1e-4"; do
    TAR_ALPHA=1.0 TAR_LR=$lr TAR_EPOCHS=1 \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 3. Epoch Sweep  (baseline alpha=1.0, lr=1e-5)
# More epochs → larger accumulated task vector.
# Too many epochs → the forget-FT model may overfit,
# producing a noisy/uninformative task vector.
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Epoch sweep (alpha=1.0, lr=1e-5) <<<"
for ep in "1" "2" "3" "5"; do
    TAR_ALPHA=1.0 TAR_LR=1e-5 TAR_EPOCHS=$ep \
        ./unlearn/run_unlearn.sh $METHOD
done

# ---------------------------------------------------------------
# 4. Alpha × LR Joint Grid  (epochs=1)
# Explore whether high α requires low LR (and vice versa) to
# avoid MMLU collapse while still pushing WMDP down.
# Only run pairs that are NOT already covered above.
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Alpha × LR joint grid (epochs=1) <<<"
for alpha in "2.0" "4.0"; do
    for lr in "5e-6" "3e-5"; do
        TAR_ALPHA=$alpha TAR_LR=$lr TAR_EPOCHS=1 \
            ./unlearn/run_unlearn.sh $METHOD
    done
done

# ---------------------------------------------------------------
# 5. Best alpha at more epochs
# Once section 1 & 2 are run, manually pick the best alpha and lr.
# These two extra points explore whether longer training at the
# best point helps (usually 2–3 epochs is sufficient for TAR).
# Values below are placeholder estimates — adjust after initial results.
# ---------------------------------------------------------------
echo -e "\n>>> [tar] Best alpha=2.0, lr=1e-5 at higher epochs <<<"
for ep in "2" "3"; do
    TAR_ALPHA=2.0 TAR_LR=1e-5 TAR_EPOCHS=$ep \
        ./unlearn/run_unlearn.sh $METHOD
done

echo "=========================================================="
echo "Sweep 10 (TAR) completed successfully!"
echo "=========================================================="
