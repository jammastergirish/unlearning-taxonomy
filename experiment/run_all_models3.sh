#!/usr/bin/env bash
# Run collect_activation_comparison.py for the three specified unlearned models.
# Uses the same multi-seed runner and output layout as pipeline.sh.
# Run from the project root:
#   ./experiment/run_all_models3.sh [--force]

set -uo pipefail

# Always run from the project root (parent of experiment/)
cd "$(dirname "$0")/.."

# ---- Force flag ----
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[run_all_models3] --force: will rerun all steps regardless of existing results"
fi

# Load credentials
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# ---- Config (mirrors pipeline.sh defaults) ----
OUTROOT="${OUTROOT:-outputs}"
ACTIVATION_DEVICE="${ACTIVATION_DEVICE:-auto}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-auto}"
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"
SEEDS="${SEEDS:-42 123 456}"

MODEL_A="EleutherAI/deep-ignorance-unfiltered"

MODELS=(
  "girishgupta/deep-ignorance-unfiltered_unlearned_ga"
  "girishgupta/deep-ignorance-unfiltered_unlearned_grad_diff"
  "girishgupta/deep-ignorance-unfiltered_unlearned_simnpo"
)

# ---- Helpers (mirrors pipeline.sh) ----
step_complete() {
  local dir="$1" sentinel="$2"
  if [[ "$FORCE" == "1" ]]; then return 1; fi
  [[ -f "${dir}/${sentinel}" ]]
}

run_multiseed_experiment() {
  local base_outdir="$1" sentinel="$2" script="$3"
  shift 3
  local extra_args=("$@")

  if step_complete "$base_outdir" "$sentinel"; then
    echo "  ✓ Already complete — skipping"
    return
  fi

  local seed_dirs=()
  for seed in $SEEDS; do
    local seed_outdir="${base_outdir}/seed_${seed}"
    mkdir -p "$seed_outdir"
    echo "    Running with seed $seed..."
    uv run "$script" "${extra_args[@]}" --seed "$seed" --outdir "$seed_outdir"
    seed_dirs+=("$seed_outdir")
  done

  echo "    Aggregating results across seeds..."
  uv run experiment/aggregate_multiseed_results.py \
    --seed-dirs "${seed_dirs[@]}" \
    --output-dir "$base_outdir" \
    --sentinel-file "$sentinel"
}

# ---- Main loop ----
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
  echo "[$N/$TOTAL] Comparing: $MODEL_A → $MODEL_B"
  echo "----------------------------------------"

  if run_multiseed_experiment "${OUTROOT}/${COMP}/activation_comparison" "activation_comparison.csv" \
    "experiment/collect_activation_comparison.py" \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --title "${MODEL_B##*/}: Activation Norms"; then
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
