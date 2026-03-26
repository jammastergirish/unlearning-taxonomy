#!/usr/bin/env bash
# Re-run all best unlearning configs with the Muon optimizer instead of AdamW.
#
# Uses the same hyperparameters from best_unlearning_models.md but adds
# OPTIMIZER=muon. Results are evaluated and logged to W&B (tagged with
# _optmuon in the run name by unlearn.py). No HuggingFace upload — review
# W&B metrics first.
#
# Usage:
#   ./unlearn/reproduce_best_models_muon.sh              # run all methods
#   ./unlearn/reproduce_best_models_muon.sh npo simnpo   # run specific methods only
#   DRY_RUN=1 ./unlearn/reproduce_best_models_muon.sh    # print commands without running
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN="${DRY_RUN:-0}"
METHODS_FILTER=("$@")

# Methods to skip — TAR has its own optimizer path that doesn't use --optimizer,
# and cb/cb_lat/lat are excluded per project decision.
EXCLUDE_METHODS=(cb cb_lat lat tar)

# ── Best configs from best_unlearning_models.md ──────────────────────────
# Each entry: method|env_overrides (same hyperparams as adamw, Muon added at invocation)
BEST_CONFIGS=(
  "dpo|EPOCHS=1 LR=4.5e-05 BATCH_SIZE=32 BETA=0.01 MAX_LENGTH=512 MAX_LINES=8192"
  "ga|EPOCHS=4 LR=3e-05 BATCH_SIZE=32 RETAIN_WEIGHT=1.0 MAX_LENGTH=512 MAX_LINES=10000"
  "ga_simple|EPOCHS=2 LR=1e-05 BATCH_SIZE=32 MAX_LENGTH=512 MAX_LINES=10000"
  "grad_diff|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 FORGET_WEIGHT=0.5 MAX_LENGTH=512 MAX_LINES=10000"
  "npo|EPOCHS=1 LR=4.5e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.5 MAX_LENGTH=512 MAX_LINES=8192"
  "rmu|EPOCHS=1 LR=2e-05 BATCH_SIZE=32 ALPHA=1000.0 STEERING_COEFF=20.0 LAYER_ID=11-12-13 MAX_LENGTH=2048"
  "simnpo|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 BETA=0.01 RETAIN_WEIGHT=1.0 MAX_LENGTH=2048"
  "wt_dist|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 WT_NOISE_STD=0.01 MAX_LENGTH=2048"
  "wt_dist_reg|EPOCHS=3 LR=3e-05 BATCH_SIZE=32 WT_REG_LAMBDA=1.0 MAX_LENGTH=512 MAX_LINES=10000"
)

passed=0
failed=0
skipped=0
failed_methods=()

for entry in "${BEST_CONFIGS[@]}"; do
  IFS='|' read -r method env_vars <<< "$entry"

  # Skip excluded methods
  for excluded in "${EXCLUDE_METHODS[@]}"; do
    if [[ "$excluded" == "$method" ]]; then
      echo "[muon] Skipping excluded method: $method"
      skipped=$((skipped + 1))
      continue 2
    fi
  done

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
  echo "  Method: $method (Muon optimizer)"
  echo "  Config: $env_vars"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] Would run: env $env_vars OPTIMIZER=muon FORCE=1 ./unlearn/run_unlearn.sh $method"
    skipped=$((skipped + 1))
    continue
  fi

  # Run unlearning with Muon optimizer — no --push-to-hub, just train + eval + W&B
  echo "[muon] Training $method with Muon ..."
  if ! env $env_vars OPTIMIZER=muon FORCE=1 ./unlearn/run_unlearn.sh "$method"; then
    echo "[muon] FAILED: $method training/eval failed"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  # Find the output directory (Muon runs have _optmuon suffix in dir name)
  base_model_safe="EleutherAI_deep-ignorance-unfiltered"
  outdir=$(find "unlearned_models/$base_model_safe" -maxdepth 1 -type d -name "${method}__*_optmuon" | sort -t/ -k3 | tail -1)

  if [[ -z "$outdir" ]]; then
    echo "[muon] WARNING: Could not find output directory for $method (Muon)"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  eval_summary="$outdir/evals/summary.json"
  if [[ ! -f "$eval_summary" ]]; then
    echo "[muon] WARNING: No eval summary found at $eval_summary"
    failed=$((failed + 1))
    failed_methods+=("$method")
    continue
  fi

  # Extract and display MMLU and WMDP Bio (Robust) from summary.json
  actual_mmlu=$(python3 -c "
import json, sys
with open('$eval_summary') as f:
    data = json.load(f)
results = data.get('results', {})
mmlu = results.get('mmlu', {}).get('acc,none')
if mmlu is None:
    print('N/A'); sys.exit(0)
print(f'{mmlu:.4f}')
")

  actual_wmdp=$(python3 -c "
import json, sys
with open('$eval_summary') as f:
    data = json.load(f)
results = data.get('results', {})
wmdp = results.get('wmdp_bio_robust', {}).get('acc,none')
if wmdp is None:
    print('N/A'); sys.exit(0)
print(f'{wmdp:.4f}')
")

  echo "[muon] $method — MMLU: $actual_mmlu  |  WMDP (Robust): $actual_wmdp"
  echo "[muon] ✓ $method complete — check W&B for full results"
  passed=$((passed + 1))
done

echo ""
echo "================================================================"
echo "  SUMMARY (Muon optimizer)"
echo "  Completed: $passed  |  Failed: $failed  |  Skipped: $skipped"
if [[ ${#failed_methods[@]} -gt 0 ]]; then
  echo "  Failed methods: ${failed_methods[*]}"
fi
echo "================================================================"

if [[ $failed -gt 0 ]]; then
  exit 1
fi
