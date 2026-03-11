#!/usr/bin/env bash
# Re-run collect_weight_comparison.py for every model in run_all_models.sh.
# Uses --force so it overwrites existing CSVs/plots even if they already exist.
# Run from the project root:
#   ./experiment/run_all_models2.sh

set -uo pipefail

# Always run from the project root (same as pipeline.sh)
cd "$(dirname "$0")/.."

# Load credentials
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

OUTROOT="${OUTROOT:-outputs}"
PARAM_DEVICE="${PARAM_DEVICE:-auto}"
PARAM_DTYPE="${PARAM_DTYPE:-fp16}"

MODEL_A="EleutherAI/deep-ignorance-unfiltered"

MODELS=(
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga"
  "girishgupta/deep-ignorance-unfiltered_unlearned_grad_diff"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga_simple"
  "girishgupta/deep-ignorance-unfiltered_unlearned_npo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_rmu"
  "girishgupta/deep-ignorance-unfiltered_unlearned_lat"
  "girishgupta/deep-ignorance-unfiltered_unlearned_cb_lat"
  "girishgupta/deep-ignorance-unfiltered_unlearned_simnpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_wt_dist"
)

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

for i in "${!MODELS[@]}"; do
  MODEL_B="${MODELS[$i]}"
  N=$((i + 1))

  MODEL_A_DIR="${MODEL_A//\//_}"
  MODEL_B_DIR="${MODEL_B//\//_}"
  COMP="${MODEL_A_DIR}__to__${MODEL_B_DIR}"

  echo ""
  echo "========================================"
  echo "[$N/$TOTAL] $MODEL_A → $MODEL_B"
  echo "========================================"

  if uv run experiment/collect_weight_comparison.py \
      --model-a "$MODEL_A" \
      --model-b "$MODEL_B" \
      --device "$PARAM_DEVICE" \
      --dtype "$PARAM_DTYPE" \
      --outdir "${OUTROOT}/${COMP}/weight_comparison" \
      --plot-outdir "${OUTROOT}/${COMP}/param_plots" \
      --title "$MODEL_A → $MODEL_B"; then
    echo "[OK] $MODEL_B"
    ((PASSED++))
  else
    echo "[FAIL] $MODEL_B (exit $?). Continuing..."
    ((FAILED++))
  fi
done

echo ""
echo "========================================"
echo "Done. Passed: $PASSED / $TOTAL, Failed: $FAILED / $TOTAL"
echo "========================================"
