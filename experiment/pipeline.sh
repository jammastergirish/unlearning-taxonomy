#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root (parent of experiment/)
cd "$(dirname "$0")/.."

clear && printf '\e[3J'

# ---- Force flag: pass --force to rerun completed steps ----
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  echo "[pipeline] --force: will rerun all steps regardless of existing results"
fi

# Load .env (HF_TOKEN, WANDB_API_KEY, etc.) if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Group all W&B runs from this pipeline invocation together
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-pipeline_$(date +%s)}"

# Configuration — single output root for all results
OUTROOT="${OUTROOT:-outputs}"

# Models (UNLEARNED can be overridden: UNLEARNED=user/model ./experiment/pipeline.sh)
BASE="EleutherAI/deep-ignorance-unfiltered"
FILTERED="EleutherAI/deep-ignorance-e2e-strong-filter"
UNLEARNED="${UNLEARNED:-EleutherAI/deep-ignorance-unfiltered-cb-lat}"
CB_ONLY="EleutherAI/deep-ignorance-unfiltered-cb"
PRETRAIN="EleutherAI/deep-ignorance-pretraining-stage-unfiltered"

# Opt-in flags for extra comparisons (set to 1 to enable).
# By default only COMP1 (Base→Filtered) and COMP2 (Base→Unlearned) run.
# COMP3: Base→Pretraining checkpoint
# COMP4: Base→CB-only, COMP5: CB-only→Unlearned, COMP6: Unlearned→Filtered
ENABLE_PRETRAIN_COMPARISON="${ENABLE_PRETRAIN_COMPARISON:-0}"
ENABLE_CB_COMPARISONS="${ENABLE_CB_COMPARISONS:-0}"
# Tuned lens is slow (~1hr per model). Logit lens runs by default; set to 1 to also run tuned lens.
ENABLE_TUNED_LENS="${ENABLE_TUNED_LENS:-0}"
# Multiple seeds for statistical robustness (space-separated list)
SEEDS="${SEEDS:-42 123 456}"

# Comparison names (derived from model IDs: / → _)
COMP1="EleutherAI_deep-ignorance-unfiltered__to__EleutherAI_deep-ignorance-e2e-strong-filter"
COMP2="EleutherAI_deep-ignorance-unfiltered__to__${UNLEARNED//\//_}"
COMP3="EleutherAI_deep-ignorance-unfiltered__to__${PRETRAIN//\//_}"
COMP4="EleutherAI_deep-ignorance-unfiltered__to__${CB_ONLY//\//_}"
COMP5="${CB_ONLY//\//_}__to__${UNLEARNED//\//_}"
COMP6="${UNLEARNED//\//_}__to__EleutherAI_deep-ignorance-e2e-strong-filter"

# Per-model folder names (for analyses that run once per model, not per comparison)
MODEL_BASE="EleutherAI_deep-ignorance-unfiltered"
MODEL_FILTERED="EleutherAI_deep-ignorance-e2e-strong-filter"
MODEL_UNLEARNED="${UNLEARNED//\//_}"

# Device and dtype settings
PARAM_DEVICE="${PARAM_DEVICE:-auto}"  # auto = cuda > mps > cpu
PARAM_DTYPE="${PARAM_DTYPE:-fp16}"
ACTIVATION_DEVICE="${ACTIVATION_DEVICE:-auto}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-auto}"

# Data paths
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

# ---- Skip-if-complete helper ----
# Usage: step_complete <dir> <sentinel_file>
# Returns 0 (true) if the sentinel exists and --force was not passed.
step_complete() {
  local dir="$1" sentinel="$2"
  if [[ "$FORCE" == "1" ]]; then return 1; fi
  [[ -f "${dir}/${sentinel}" ]]
}

# ---- Multi-seed experiment runner ----
# Usage: run_multiseed_experiment <base_outdir> <sentinel_file> <script> [args...]
run_multiseed_experiment() {
  local base_outdir="$1" sentinel="$2" script="$3"
  shift 3
  local extra_args=("$@")

  if step_complete "$base_outdir" "$sentinel"; then
    echo "  ✓ Already complete — skipping"
    return
  fi

  # Run experiment for each seed
  local seed_dirs=()
  for seed in $SEEDS; do
    local seed_outdir="${base_outdir}/seed_${seed}"
    mkdir -p "$seed_outdir"

    echo "    Running with seed $seed..."
    uv run "$script" "${extra_args[@]}" --seed "$seed" --outdir "$seed_outdir"
    seed_dirs+=("$seed_outdir")
  done

  # Aggregate results across seeds
  echo "    Aggregating results across seeds..."
  uv run experiment/aggregate_multiseed_results.py \
    --seed-dirs "${seed_dirs[@]}" \
    --output-dir "$base_outdir" \
    --sentinel-file "$sentinel"
}

echo "=========================================="
echo "      MODEL DIFFS ANALYSIS PIPELINE"
echo "=========================================="
echo ""
echo "Base model:    $BASE"
echo "Comparison 1:  Base → $FILTERED"
echo "Comparison 2:  Base → $UNLEARNED"
if [[ "$ENABLE_PRETRAIN_COMPARISON" == "1" ]]; then
  echo "Comparison 3:  Base → $PRETRAIN  [opt-in]"
else
  echo "Comparison 3:  (disabled — set ENABLE_PRETRAIN_COMPARISON=1 to enable)"
fi
if [[ "$ENABLE_CB_COMPARISONS" == "1" ]]; then
  echo "Comparison 4:  Base → $CB_ONLY  [opt-in]"
  echo "Comparison 5:  $CB_ONLY → $UNLEARNED  [opt-in]"
  echo "Comparison 6:  $UNLEARNED → $FILTERED  [opt-in]"
else
  echo "Comparisons 4–6: (disabled — set ENABLE_CB_COMPARISONS=1 to enable)"
fi
echo "Output root:   $OUTROOT"
echo "Seeds:         $SEEDS  (for statistical robustness)"
echo ""

# ============================================
# STEP 0: Benchmark Evaluation (per-model)
# ============================================
echo "=========================================="
echo "STEP 0: Benchmark Evaluation (MMLU, WMDP, HellaSwag, TruthfulQA)"
echo "=========================================="
echo "Quick sanity check — identifies collapsed models before expensive diagnostics."
echo "(Results stored per-model, not per-comparison)"

echo ""
echo "Model: $BASE"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_BASE}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$BASE" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $FILTERED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_FILTERED}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$FILTERED" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

echo ""
echo "Model: $UNLEARNED"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${MODEL_UNLEARNED}/evals" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/eval.py \
    --model "$UNLEARNED" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
fi

# ============================================
# STEP 1: Parameter Statistics & Weight Comparison
# ============================================
echo "=========================================="
echo "STEP 1: Parameter Statistics & Weight Comparison"
echo "=========================================="
echo "Computing per-component metrics (Frobenius, spectral, stable rank, cosine sim, etc.)"

echo ""
echo "Comparison 1: Base → Filtered"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP1}/weight_comparison" "per_matrix.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/collect_weight_comparison.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --outdir "${OUTROOT}/${COMP1}/weight_comparison" \
    --plot-outdir "${OUTROOT}/${COMP1}/param_plots" \
    --title "$BASE → $FILTERED"
fi

echo ""
echo "Comparison 2: Base → Unlearned"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP2}/weight_comparison" "per_matrix.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/collect_weight_comparison.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --outdir "${OUTROOT}/${COMP2}/weight_comparison" \
    --plot-outdir "${OUTROOT}/${COMP2}/param_plots" \
    --title "$BASE → $UNLEARNED"
fi

if [[ "$ENABLE_PRETRAIN_COMPARISON" == "1" ]]; then
  echo ""
  echo "Comparison 3: Base → Pretraining"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP3}/weight_comparison" "per_matrix.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/collect_weight_comparison.py \
      --model-a "$BASE" \
      --model-b "$PRETRAIN" \
      --device "$PARAM_DEVICE" \
      --dtype "$PARAM_DTYPE" \
      --outdir "${OUTROOT}/${COMP3}/weight_comparison" \
      --plot-outdir "${OUTROOT}/${COMP3}/param_plots" \
      --title "$BASE → $PRETRAIN"
  fi
else
  echo ""
  echo "Comparison 3: Base → Pretraining (skipped — set ENABLE_PRETRAIN_COMPARISON=1 to enable)"
fi

if [[ "$ENABLE_CB_COMPARISONS" == "1" ]]; then
  echo ""
  echo "Comparison 4: Base → CB-only"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP4}/weight_comparison" "per_matrix.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/collect_weight_comparison.py \
      --model-a "$BASE" \
      --model-b "$CB_ONLY" \
      --device "$PARAM_DEVICE" \
      --dtype "$PARAM_DTYPE" \
      --outdir "${OUTROOT}/${COMP4}/weight_comparison" \
      --plot-outdir "${OUTROOT}/${COMP4}/param_plots" \
      --title "$BASE → $CB_ONLY"
  fi

  echo ""
  echo "Comparison 5: CB-only → CB+LAT"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP5}/weight_comparison" "per_matrix.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/collect_weight_comparison.py \
      --model-a "$CB_ONLY" \
      --model-b "$UNLEARNED" \
      --device "$PARAM_DEVICE" \
      --dtype "$PARAM_DTYPE" \
      --outdir "${OUTROOT}/${COMP5}/weight_comparison" \
      --plot-outdir "${OUTROOT}/${COMP5}/param_plots" \
      --title "$CB_ONLY → $UNLEARNED"
  fi

  echo ""
  echo "Comparison 6: CB+LAT → Filtered"
  echo "----------------------------------------"
  if step_complete "${OUTROOT}/${COMP6}/weight_comparison" "per_matrix.csv"; then
    echo "  ✓ Already complete — skipping"
  else
    uv run experiment/collect_weight_comparison.py \
      --model-a "$UNLEARNED" \
      --model-b "$FILTERED" \
      --device "$PARAM_DEVICE" \
      --dtype "$PARAM_DTYPE" \
      --outdir "${OUTROOT}/${COMP6}/weight_comparison" \
      --plot-outdir "${OUTROOT}/${COMP6}/param_plots" \
      --title "$UNLEARNED → $FILTERED"
  fi
else
  echo ""
  echo "Comparisons 4–6: CB-only chain (skipped — set ENABLE_CB_COMPARISONS=1 to enable)"
fi

# ============================================
# STEP 2: Generate Test Datasets
# ============================================
echo ""
echo "=========================================="
echo "STEP 2: Generating Test Datasets"
echo "=========================================="
if [[ -f "$FORGET" && -f "$RETAIN" && "$FORCE" != "1" ]]; then
  echo "  ✓ Datasets already exist — skipping"
else
  uv run create_datasets.py
fi

# ============================================
# STEP 3: Activation Norms
# ============================================
echo ""
echo "=========================================="
echo "STEP 3: Activation Norms"
echo "=========================================="

if [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Warning: Activation files missing; skipping activation analysis."
else
  echo ""
  echo "Comparison 1: Base → Filtered"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${COMP1}/activation_comparison" "activation_comparison.csv" \
    "experiment/collect_activation_comparison.py" \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --title "E2E Strong Filter: Activation Norms"

  echo ""
  echo "Comparison 2: Base → Unlearned"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${COMP2}/activation_comparison" "activation_comparison.csv" \
    "experiment/collect_activation_comparison.py" \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --title "${UNLEARNED##*/}: Activation Norms"
fi

# ============================================
# STEP 4: MLP vs Attention Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 4: MLP vs Attention Analysis"
echo "=========================================="

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/analyze_mlp_vs_attn.py \
    --per-layer-csv "${OUTROOT}/${COMP1}/weight_comparison/per_coarse_layer.csv" \
    --per-matrix-csv "${OUTROOT}/${COMP1}/weight_comparison/per_matrix.csv" \
    --outdir "${OUTROOT}/${COMP1}/mlp_attn_analysis" \
    --title "E2E Strong Filter: MLP vs Attention"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/analyze_mlp_vs_attn.py \
    --per-layer-csv "${OUTROOT}/${COMP2}/weight_comparison/per_coarse_layer.csv" \
    --per-matrix-csv "${OUTROOT}/${COMP2}/weight_comparison/per_matrix.csv" \
    --outdir "${OUTROOT}/${COMP2}/mlp_attn_analysis" \
    --title "${UNLEARNED##*/}: MLP vs Attention"
fi

# ============================================
# STEP 6: Layer-wise WMDP Accuracy (per-model)
# ============================================
echo ""
echo "=========================================="
echo "STEP 5: Layer-wise WMDP Accuracy (Logit Lens)"
echo "=========================================="
echo "Measuring WMDP-Bio MCQ accuracy at every transformer layer..."
echo "(Results stored per-model, not per-comparison)"
if [[ "$ENABLE_TUNED_LENS" == "1" ]]; then
  echo "Tuned lens enabled — will also train per-layer affine probes (~1hr per model)"
else
  echo "Tuned lens disabled — logit lens only (set ENABLE_TUNED_LENS=1 to enable)"
fi

LENS_MODES="logit"
if [[ "$ENABLE_TUNED_LENS" == "1" ]]; then
  LENS_MODES="logit tuned"
fi

for LENS in $LENS_MODES; do
  echo ""
  echo "--- Lens: ${LENS} ---"

  echo ""
  echo "Model: $BASE"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${MODEL_BASE}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$BASE" \
    --lens "$LENS" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"

  echo ""
  echo "Model: $FILTERED"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${MODEL_FILTERED}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$FILTERED" \
    --lens "$LENS" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"

  echo ""
  echo "Model: $UNLEARNED"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${MODEL_UNLEARNED}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$UNLEARNED" \
    --lens "$LENS" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"
done

# ============================================
# STEP 7: Null Space & Subspace Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 7: Null Space & Subspace Analysis"
echo "=========================================="
echo "Note: This is computationally intensive (SVD on 50 weight matrices)"

echo ""
echo "Analyzing Comparison 1..."
run_multiseed_experiment "${OUTROOT}/${COMP1}/null_space_analysis" "null_space_visualization.png" \
  "experiment/null_space_analysis.py" \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --num-samples 50

echo ""
echo "Analyzing Comparison 2..."
run_multiseed_experiment "${OUTROOT}/${COMP2}/null_space_analysis" "null_space_visualization.png" \
  "experiment/null_space_analysis.py" \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --num-samples 50

# ============================================
# STEP 8: Activation Separation Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 8: Activation Separation Analysis"
echo "=========================================="
echo "Analyzing how well forget/retain activations are separated..."

echo ""
echo "Analyzing Comparison 1..."
run_multiseed_experiment "${OUTROOT}/${COMP1}/activation_separation" "summary.json" \
  "experiment/activation_separation_analysis.py" \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

echo ""
echo "Analyzing Comparison 2..."
run_multiseed_experiment "${OUTROOT}/${COMP2}/activation_separation" "summary.json" \
  "experiment/activation_separation_analysis.py" \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

# ============================================
# STEP 9: Activation Covariance Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 9: Activation Covariance Analysis"
echo "=========================================="
echo "Analyzing covariance spectrum changes..."

echo ""
echo "Analyzing Comparison 1..."
run_multiseed_experiment "${OUTROOT}/${COMP1}/activation_covariance" "summary.json" \
  "experiment/activation_covariance_analysis.py" \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

echo ""
echo "Analyzing Comparison 2..."
run_multiseed_experiment "${OUTROOT}/${COMP2}/activation_covariance" "summary.json" \
  "experiment/activation_covariance_analysis.py" \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

# ============================================
# STEP 10: MLP Nullspace Alignment
# ============================================
echo ""
echo "=========================================="
echo "STEP 10: MLP Nullspace Alignment Analysis"
echo "=========================================="
echo "Analyzing if MLP updates align with nullspace..."

echo ""
echo "Analyzing Comparison 1..."
if step_complete "${OUTROOT}/${COMP1}/mlp_nullspace_alignment" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$FILTERED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE"
fi

echo ""
echo "Analyzing Comparison 2..."
if step_complete "${OUTROOT}/${COMP2}/mlp_nullspace_alignment" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/mlp_nullspace_alignment.py \
    --model-a "$BASE" \
    --model-b "$UNLEARNED" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE"
fi

# ============================================
# STEP 11: Row Space Projection Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 11: Row Space Projection Analysis"
echo "=========================================="
echo "Analyzing how activations project onto update directions..."

echo ""
echo "Analyzing Comparison 1..."
run_multiseed_experiment "${OUTROOT}/${COMP1}/row_space_projection" "summary.json" \
  "experiment/row_space_projection_analysis.py" \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

echo ""
echo "Analyzing Comparison 2..."
run_multiseed_experiment "${OUTROOT}/${COMP2}/row_space_projection" "summary.json" \
  "experiment/row_space_projection_analysis.py" \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

# ============================================
# STEP 12: Local Lipschitzness Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 12: Local Lipschitzness Analysis"
echo "=========================================="
echo "Analyzing local smoothness changes..."

echo ""
echo "Analyzing Comparison 1..."
run_multiseed_experiment "${OUTROOT}/${COMP1}/lipschitzness_analysis" "summary.json" \
  "experiment/local_lipschitzness_analysis.py" \
  --model-a "$BASE" \
  --model-b "$FILTERED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

echo ""
echo "Analyzing Comparison 2..."
run_multiseed_experiment "${OUTROOT}/${COMP2}/lipschitzness_analysis" "summary.json" \
  "experiment/local_lipschitzness_analysis.py" \
  --model-a "$BASE" \
  --model-b "$UNLEARNED" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

# ============================================
# COMPLETION
# ============================================
echo ""
echo "=========================================="
echo "        PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "All results saved under: ${OUTROOT}/"
echo ""
echo "  <comparison>/"
echo "    weight_comparison/      per_matrix.csv, per_component.csv, per_layer.csv, per_coarse_layer.csv"
echo "    param_plots/            Layer locality, stable rank, spectral norm PNGs"
echo "    activation_comparison/  activation_comparison.csv + _std columns (multi-seed aggregated)"
echo "    activation_plots/       Activation norms, diffs PNGs"
echo "    mlp_attn_analysis/      summary CSV + plots"
echo "    null_space_analysis/    null_space_results.csv + plots (multi-seed aggregated)"
echo "    activation_separation/  separation metrics + plots (multi-seed aggregated)"
echo "    activation_covariance/  covariance spectra + plots (multi-seed aggregated)"
echo "    mlp_nullspace_alignment/ alignment metrics + plots"
echo "    row_space_projection/   projection metrics + plots (multi-seed aggregated)"
echo "    lipschitzness_analysis/ Lipschitz estimates + plots (multi-seed aggregated)"
echo "        └── seed_*/         Individual seed results (for debugging)"
echo ""
echo "  <model>/"
echo "    evals/                  summary.json (MMLU, WMDP, HellaSwag, TruthfulQA)"
echo "    linear_probes/          probe_results.csv, summary.json + plot"
echo "    wmdp_logit_lens/        wmdp_lens_results.csv, summary.json + _std fields (multi-seed)"
echo "    wmdp_tuned_lens/        wmdp_lens_results.csv, summary.json + _std fields (multi-seed)"
echo "        └── seed_*/         Individual seed results"
echo ""
echo "Statistical Robustness:"
echo "  • Multi-seed experiments (${SEEDS}) provide error bars for stochastic analyses"
echo "  • Results include mean ± std across seeds in CSV/JSON files"
echo "  • Individual seed results preserved under seed_*/ subdirectories"
echo ""
echo "Tip: rerun with --force to regenerate all results."
echo "Tip: set SEEDS=\"42 123 456 789 999\" for more robust statistics."
echo ""