#!/usr/bin/env bash
# Run pipeline.sh on every unlearning model from girishgupta's HuggingFace.
# Ordered: ga, grad_diff, ga_simple first, then the rest.
# Continues past failures (set +e).

set -uo pipefail

BASE="EleutherAI/deep-ignorance-unfiltered"

# Priority models first, then the rest
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
  MODEL="${MODELS[$i]}"
  N=$((i + 1))
  echo ""
  echo "========================================"
  echo "[$N/$TOTAL] Running pipeline for: $MODEL"
  echo "========================================"
  echo ""

  if MODEL_A="$BASE" MODEL_B="$MODEL" ./experiment/pipeline.sh; then
    echo "[OK] $MODEL completed successfully."
    ((PASSED++))
  else
    echo "[FAIL] $MODEL crashed (exit code $?). Continuing..."
    ((FAILED++))
  fi
done

echo ""
echo "========================================"
echo "All done. Passed: $PASSED / $TOTAL, Failed: $FAILED / $TOTAL"
echo "========================================"
