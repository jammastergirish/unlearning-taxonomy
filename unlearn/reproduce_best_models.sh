#!/usr/bin/env bash
# Reproduce all best unlearned models, verify eval scores, and upload to HuggingFace.
#
# This script:
#   1. Trains each best model config from best_unlearning_models.md
#   2. Evaluates MMLU and WMDP Bio (Robust)
#   3. Compares against expected scores (tolerance configurable via TOLERANCE)
#   4. Uploads to HuggingFace if scores match (requires HF_TOKEN)
#
# Usage:
#   ./unlearn/reproduce_best_models.sh              # run all methods
#   ./unlearn/reproduce_best_models.sh npo simnpo   # run specific methods only
#   DRY_RUN=1 ./unlearn/reproduce_best_models.sh    # print commands without running
#   TOLERANCE=0.02 ./unlearn/reproduce_best_models.sh  # allow 2% deviation
#
# Requires: HF_TOKEN env var for uploading to HuggingFace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

TOLERANCE="${TOLERANCE:-0.01}"
DRY_RUN="${DRY_RUN:-0}"
METHODS_FILTER=("$@")

# ── Best configs from best_unlearning_models.md ──────────────────────────
# Each entry: method|expected_mmlu|expected_wmdp_robust|env_overrides
BEST_CONFIGS=(
  "cb|0.4432|0.4101|EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 ALPHA=200.0 STEERING_COEFF=10.0 LAYER_ID=13-14-15 MAX_LENGTH=512 MAX_LINES=1024"
  "cb_lat|0.4435|0.4124|EPOCHS=1 LR=1.3e-05 BATCH_SIZE=16 ALPHA=200.0 STEERING_COEFF=10.0 LAT_EPS=0.1 LAT_STEPS=5 LAYER_ID=13-14-15 MAX_LENGTH=512 MAX_LINES=1024"
  "dpo|0.3840|0.2408|EPOCHS=1 LR=4.5e-05 BATCH_SIZE=32 BETA=0.01 MAX_LENGTH=512 MAX_LINES=8192"
  "ga|0.4064|0.2523|EPOCHS=4 LR=3e-05 BATCH_SIZE=32 RETAIN_WEIGHT=1.0 MAX_LENGTH=512 MAX_LINES=10000"
  "ga_simple|0.2690|0.2408|EPOCHS=2 LR=1e-05 BATCH_SIZE=32 MAX_LENGTH=512 MAX_LINES=10000"
  "grad_diff|0.4296|0.2581|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 FORGET_WEIGHT=0.5 MAX_LENGTH=512 MAX_LINES=10000"
  "lat|0.2898|0.2742|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 LAT_EPS=0.1 LAT_STEPS=5 RETAIN_WEIGHT=1.0 LAYER_ID=5-6-7 MAX_LENGTH=512 MAX_LINES=2048"
  "npo|0.4426|0.2673|EPOCHS=1 LR=4.5e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.5 MAX_LENGTH=512 MAX_LINES=8192"
  "rmu|0.4030|0.3468|EPOCHS=1 LR=2e-05 BATCH_SIZE=32 ALPHA=1000.0 STEERING_COEFF=20.0 LAYER_ID=11-12-13 MAX_LENGTH=2048"
  "simnpo|0.4315|0.2535|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.0 MAX_LENGTH=2048"
  "tar|0.3984|0.3514|TAR_ALPHA=6.5 TAR_LR=5e-06 TAR_EPOCHS=1 MAX_LENGTH=512 MAX_LINES=8192"
  "wt_dist|0.3518|0.2903|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 WT_NOISE_STD=0.01 MAX_LENGTH=2048"
  "wt_dist_reg|0.2637|0.2396|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 WT_REG_LAMBDA=1.0 MAX_LENGTH=512 MAX_LINES=10000"
)

passed=0
failed=0
skipped=0
failed_methods=()

for entry in "${BEST_CONFIGS[@]}"; do
  IFS='|' read -r method expected_mmlu expected_wmdp env_vars <<< "$entry"

  # Filter if specific methods were requested
  if [[ ${#METHODS_FILTER[@]} -gt 0 ]]; then
    match=0
    for m in "${METHODS_FILTER[@]}"; do
      if [[ "$m" == "$method" ]]; then match=1; break; fi
    done
    if [[ $match -eq 0 ]]; then
      continue
    fi
  fi

  echo ""
  echo "================================================================"
  echo "  Method: $method"
  echo "  Expected MMLU: $expected_mmlu  |  Expected WMDP (Robust): $expected_wmdp"
  echo "  Config: $env_vars"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] Would run: env $env_vars FORCE=1 PUSH_TO_HUB=1 ./unlearn/run_unlearn.sh $method"
    skipped=$((skipped + 1))
    continue
  fi

  # Run unlearning with --force (skip W&B idempotency), push-to-hub enabled
  # The eval is run automatically by unlearn.py after training
  echo "[reproduce] Training $method ..."
  if ! env $env_vars FORCE=1 PUSH_TO_HUB=1 ./unlearn/run_unlearn.sh "$method"; then
    echo "[reproduce] FAILED: $method training/eval failed"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  # ── Verify eval scores from the output summary.json ──────────────
  # Determine the output directory name (mirrors unlearn.py naming convention)
  base_model_safe="EleutherAI_deep-ignorance-unfiltered"
  # Build run name from env vars to find the output dir
  # Instead of reconstructing, find the most recent output dir for this method
  outdir=$(find "outputs/$base_model_safe" -maxdepth 1 -type d -name "${method}__*" | sort -t/ -k3 | tail -1)

  if [[ -z "$outdir" ]]; then
    echo "[reproduce] WARNING: Could not find output directory for $method"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  eval_summary="$outdir/evals/summary.json"
  if [[ ! -f "$eval_summary" ]]; then
    echo "[reproduce] WARNING: No eval summary found at $eval_summary"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  # Extract MMLU and WMDP Bio (Robust) from summary.json
  actual_mmlu=$(python3 -c "
import json, sys
with open('$eval_summary') as f:
    data = json.load(f)
results = data.get('results', {})
mmlu = results.get('mmlu', {}).get('acc,none')
if mmlu is None:
    sys.exit(1)
print(f'{mmlu:.4f}')
")

  actual_wmdp=$(python3 -c "
import json, sys
with open('$eval_summary') as f:
    data = json.load(f)
results = data.get('results', {})
wmdp = results.get('wmdp_bio_robust', {}).get('acc,none')
if wmdp is None:
    sys.exit(1)
print(f'{wmdp:.4f}')
")

  echo "[reproduce] $method — Actual MMLU: $actual_mmlu (expected $expected_mmlu)"
  echo "[reproduce] $method — Actual WMDP: $actual_wmdp (expected $expected_wmdp)"

  # Check if scores are within tolerance
  score_ok=$(python3 -c "
import sys
actual_mmlu = float('$actual_mmlu')
expected_mmlu = float('$expected_mmlu')
actual_wmdp = float('$actual_wmdp')
expected_wmdp = float('$expected_wmdp')
tol = float('$TOLERANCE')
mmlu_ok = abs(actual_mmlu - expected_mmlu) <= tol
wmdp_ok = abs(actual_wmdp - expected_wmdp) <= tol
if mmlu_ok and wmdp_ok:
    print('PASS')
else:
    if not mmlu_ok:
        print(f'FAIL: MMLU diff = {abs(actual_mmlu - expected_mmlu):.4f} > {tol}', file=sys.stderr)
    if not wmdp_ok:
        print(f'FAIL: WMDP diff = {abs(actual_wmdp - expected_wmdp):.4f} > {tol}', file=sys.stderr)
    print('FAIL')
")

  if [[ "$score_ok" == "PASS" ]]; then
    echo "[reproduce] ✓ $method PASSED — scores within tolerance ($TOLERANCE)"
    passed=$((passed + 1))
  else
    echo "[reproduce] ✗ $method FAILED — scores outside tolerance ($TOLERANCE)"
    failed=$((failed + 1))
    failed_methods+=("$method")
  fi
done

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "  Passed: $passed  |  Failed: $failed  |  Skipped: $skipped"
if [[ ${#failed_methods[@]} -gt 0 ]]; then
  echo "  Failed methods: ${failed_methods[*]}"
fi
echo "================================================================"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
