#!/usr/bin/env bash
# Single TAR run for testing — just one configuration to isolate any issues.
#
# This runs a single TAR (Task Arithmetic Removal) configuration to test
# whether the evaluation bug fix works correctly without resource contention
# from running multiple jobs simultaneously.
#
# TAR parameters:
#   tar_alpha=2.0  — moderately strong negation of forget direction
#   tar_lr=1e-5    — standard learning rate for forget fine-tuning
#   tar_epochs=1   — single epoch to keep it fast
#

set -euo pipefail

cd "$(dirname "$0")/.."

export BASE="${BASE:-EleutherAI/deep-ignorance-unfiltered}"
export DEVICE="${DEVICE:-auto}"
export DTYPE="${DTYPE:-auto}"
export EVAL_SPLIT=0.1
export MAX_LENGTH=512

echo "=========================================================="
echo "Running single TAR test"
echo "Method: TAR (Task Arithmetic Removal)"
echo "Model:  $BASE"
echo "Device: $DEVICE"
echo "Config: alpha=2.0, lr=1e-5, epochs=1"
echo "=========================================================="

METHOD="tar"

# Single test run
TAR_ALPHA=2.0 TAR_LR=1e-5 TAR_EPOCHS=1 \
    ./unlearn/run_unlearn.sh $METHOD

echo "=========================================================="
echo "Single TAR test completed!"
echo "=========================================================="