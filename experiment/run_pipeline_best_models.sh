#!/usr/bin/env bash
# Run the full experimental pipeline on every model in the
# "best-unlearned-models" HuggingFace collection.
#
# Each model is compared against the shared base model (MODEL_A).
# Already-completed steps within each pipeline run are automatically skipped.
#
# Usage:
#   ./experiment/run_pipeline_best_models.sh              # run all models
#   ./experiment/run_pipeline_best_models.sh --force       # rerun completed steps
#   DRY_RUN=1 ./experiment/run_pipeline_best_models.sh    # print commands only
#
# All pipeline env vars are forwarded (SEEDS, ENABLE_TUNED_LENS, etc.).
set -uo pipefail

cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
  FORCE_FLAG="--force"
fi

MODEL_A="${MODEL_A:-EleutherAI/deep-ignorance-unfiltered}"

# Models from: https://huggingface.co/collections/girishgupta/best-unlearned-models
MODELS=(
  "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga"
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga_simple"
  "girishgupta/deep-ignorance-unfiltered_unlearned_grad_diff"
  "girishgupta/deep-ignorance-unfiltered_unlearned_npo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_simnpo"
  "girishgupta/deep-ignorance-unfiltered_unlearned_wt_dist"
  "girishgupta/deep-ignorance-unfiltered_unlearned_wt_dist_reg"
)

# ---- Helper: run pipeline on a list of MODEL_B paths ----
run_pass() {
  local pass_name="$1"
  shift
  local models=("$@")
  local total=${#models[@]}
  local passed=0
  local failed=0
  local failed_models=()

  echo ""
  echo "=========================================="
  echo "  $pass_name"
  echo "=========================================="
  echo "Base model:  $MODEL_A"
  echo "Models:      $total"
  echo ""

  for i in "${!models[@]}"; do
    local model_b="${models[$i]}"
    local n=$((i + 1))

    echo ""
    echo "========================================"
    echo "[$n/$total] $model_b"
    echo "========================================"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY RUN] MODEL_A=$MODEL_A MODEL_B=$model_b ./experiment/pipeline.sh $FORCE_FLAG"
      continue
    fi

    if MODEL_A="$MODEL_A" MODEL_B="$model_b" ./experiment/pipeline.sh $FORCE_FLAG; then
      echo "[$n/$total] ✓ $model_b complete"
      ((passed++))
    else
      echo "[$n/$total] ✗ $model_b failed (exit $?). Continuing..."
      ((failed++))
      failed_models+=("$model_b")
    fi
  done

  echo ""
  echo "=========================================="
  echo "  $pass_name — SUMMARY"
  echo "  Passed: $passed / $total"
  echo "  Failed: $failed / $total"
  if [[ ${#failed_models[@]} -gt 0 ]]; then
    echo "  Failed:"
    for m in "${failed_models[@]}"; do
      echo "    - $m"
    done
  fi
  echo "=========================================="

  # Store failure count in a global variable (bash functions can't return > 255)
  PASS_FAILURES=$failed
}

# ==================================================================
# Pass 1: Run pipeline on each best unlearned model (from HF).
#         Step 3b in the pipeline trains a norm-controlled variant.
# ==================================================================
PASS_FAILURES=0
run_pass "Pass 1: Best unlearned models" "${MODELS[@]}"
PASS1_FAILURES=$PASS_FAILURES

# ==================================================================
# Pass 2: Run pipeline on the norm-controlled variants produced by
#         Pass 1.  These are local directories under unlearned_models/
#         whose names contain _nrl.  The pipeline's Step 3b is
#         automatically skipped for these (IS_NORM_CONTROLLED guard).
# ==================================================================
MODEL_A_SAFE="${MODEL_A//\//_}"
NRL_MODELS=()
while IFS= read -r dir; do
  # Each dir is a local path like unlearned_models/EleutherAI_.../ga__..._nrl1.0
  # Pass it directly as MODEL_B — the pipeline handles local paths fine.
  NRL_MODELS+=("$dir")
done < <(find "unlearned_models/${MODEL_A_SAFE}" -maxdepth 1 -type d -name "*_nrl*" 2>/dev/null | sort)

if [[ ${#NRL_MODELS[@]} -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "  Pass 2: No norm-controlled models found — skipping"
  echo "=========================================="
else
  run_pass "Pass 2: Norm-controlled variants" "${NRL_MODELS[@]}"
fi

# Exit with failure if either pass had failures
if [[ $PASS1_FAILURES -gt 0 ]]; then
  exit 1
fi
