#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root (parent of experiment/)
cd "$(dirname "$0")/.."

clear && printf '\e[3J'

# Ensure datasets exist before starting
if [[ ! -f "data/forget.txt" || ! -f "data/retain.txt" ]]; then
  echo "[pipeline] Data files missing — running create_datasets.py..."
  uv run create_datasets.py
fi

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

# Models — set MODEL_A and MODEL_B to compare any two models.
# Override via environment variables:
#   MODEL_A=org/model-a MODEL_B=org/model-b ./experiment/pipeline.sh
MODEL_A="${MODEL_A:-EleutherAI/deep-ignorance-unfiltered}"
MODEL_B="${MODEL_B:-EleutherAI/deep-ignorance-e2e-strong-filter}"

# Tuned lens is slow (~1hr per model). Logit lens runs by default; set to 1 to also run tuned lens.
ENABLE_TUNED_LENS="${ENABLE_TUNED_LENS:-0}"
# Multiple seeds for statistical robustness (space-separated list)
SEEDS="${SEEDS:-42 123 456}"

# Output directory names derived from model IDs (/ → _)
MODEL_A_DIR="${MODEL_A//\//_}"
MODEL_B_DIR="${MODEL_B//\//_}"
COMP="${MODEL_A_DIR}__to__${MODEL_B_DIR}"

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
echo "Model A:  $MODEL_A"
echo "Model B:  $MODEL_B"
echo "Output root:   $OUTROOT"
echo "Seeds:         $SEEDS  (for statistical robustness)"
echo ""

# # ============================================
# # STEP 0: Benchmark Evaluation (per-model)
# # ============================================
# echo "=========================================="
# echo "STEP 0: Benchmark Evaluation (MMLU, WMDP, HellaSwag, TruthfulQA)"
# echo "=========================================="
# echo "Quick sanity check — identifies collapsed models before expensive diagnostics."
# echo "(Results stored per-model, not per-comparison)"

# echo ""
# echo "Model A: $MODEL_A"
# echo "----------------------------------------"
# if step_complete "${OUTROOT}/${MODEL_A_DIR}/evals" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run experiment/eval.py \
#     --model "$MODEL_A" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE"
# fi

# echo ""
# echo "Model B: $MODEL_B"
# echo "----------------------------------------"
# if step_complete "${OUTROOT}/${MODEL_B_DIR}/evals" "summary.json"; then
#   echo "  ✓ Already complete — skipping"
# else
#   uv run experiment/eval.py \
#     --model "$MODEL_B" \
#     --device "$ACTIVATION_DEVICE" \
#     --dtype "$ACTIVATION_DTYPE"
# fi

# ============================================
# STEP 1: Parameter Statistics & Weight Comparison
# ============================================
echo "=========================================="
echo "STEP 1: Parameter Statistics & Weight Comparison"
echo "=========================================="
echo "Computing per-component metrics (Frobenius, spectral, stable rank, cosine sim, etc.)"

echo ""
echo "Comparing: $MODEL_A → $MODEL_B"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP}/weight_comparison" "per_matrix.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/collect_weight_comparison.py \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --device "$PARAM_DEVICE" \
    --dtype "$PARAM_DTYPE" \
    --outdir "${OUTROOT}/${COMP}/weight_comparison" \
    --plot-outdir "${OUTROOT}/${COMP}/param_plots" \
    --title "$MODEL_A → $MODEL_B"
fi

# ============================================
# STEP 1.5: Singular Value Spectrum Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 1.5: Singular Value Spectrum Analysis"
echo "=========================================="
echo "Plotting full SV spectra for mlp_expand, mlp_contract, proj (early/mid/late layers)."
echo "No text data required — weight-only, deterministic."

echo ""
echo "Analyzing: $MODEL_A → $MODEL_B"
echo "----------------------------------------"
if step_complete "${OUTROOT}/${COMP}/sv_spectrum" "sv_spectrum.png"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/singular_value_spectrum_analysis.py \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --device "$PARAM_DEVICE" \
    --dtype "fp32" \
    --outdir "${OUTROOT}/${COMP}/sv_spectrum" \
    --title "$MODEL_A → $MODEL_B"
fi

# # ============================================
# # STEP 2: Generate Test Datasets
# # ============================================
# echo ""
# echo "=========================================="
# echo "STEP 2: Generating Test Datasets"
# echo "=========================================="
# if [[ -f "$FORGET" && -f "$RETAIN" && "$FORCE" != "1" ]]; then
#   echo "  ✓ Datasets already exist — skipping"
# else
#   uv run create_datasets.py
# fi

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
  echo "Comparing: $MODEL_A → $MODEL_B"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${COMP}/activation_comparison" "activation_comparison.csv" \
    "experiment/collect_activation_comparison.py" \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --forget-text "$FORGET" \
    --retain-text "$RETAIN" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE" \
    --title "${MODEL_B##*/}: Activation Norms"
fi

# ============================================
# STEP 4: MLP vs Attention Analysis
# ============================================
echo ""
echo "=========================================="
echo "STEP 4: MLP vs Attention Analysis"
echo "=========================================="

echo ""
echo "Analyzing: $MODEL_A → $MODEL_B"
if step_complete "${OUTROOT}/${COMP}/mlp_attn_analysis" "mlp_attn_summary.csv"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/analyze_mlp_vs_attn.py \
    --per-layer-csv "${OUTROOT}/${COMP}/weight_comparison/per_coarse_layer.csv" \
    --per-matrix-csv "${OUTROOT}/${COMP}/weight_comparison/per_matrix.csv" \
    --outdir "${OUTROOT}/${COMP}/mlp_attn_analysis" \
    --title "${MODEL_B##*/}: MLP vs Attention"
fi

# ============================================
# STEP 5: Layer-wise WMDP Accuracy (per-model)
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
  echo "Model A: $MODEL_A"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${MODEL_A_DIR}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$MODEL_A" \
    --lens "$LENS" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"

  echo ""
  echo "Model B: $MODEL_B"
  echo "----------------------------------------"
  run_multiseed_experiment "${OUTROOT}/${MODEL_B_DIR}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
run_multiseed_experiment "${OUTROOT}/${COMP}/null_space_analysis" "null_space_visualization.png" \
  "experiment/null_space_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
run_multiseed_experiment "${OUTROOT}/${COMP}/activation_separation" "summary.json" \
  "experiment/activation_separation_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
run_multiseed_experiment "${OUTROOT}/${COMP}/activation_covariance" "summary.json" \
  "experiment/activation_covariance_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
if step_complete "${OUTROOT}/${COMP}/mlp_nullspace_alignment" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  uv run experiment/mlp_nullspace_alignment.py \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
run_multiseed_experiment "${OUTROOT}/${COMP}/row_space_projection" "summary.json" \
  "experiment/row_space_projection_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
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
echo "Analyzing: $MODEL_A → $MODEL_B"
run_multiseed_experiment "${OUTROOT}/${COMP}/lipschitzness_analysis" "summary.json" \
  "experiment/local_lipschitzness_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
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
echo "  ${COMP}/"
echo "    weight_comparison/      per_matrix.csv, per_component.csv, per_layer.csv, per_coarse_layer.csv"
echo "    param_plots/            Layer locality, stable rank, spectral norm PNGs"
echo "    sv_spectrum/            SV spectra per component+layer, dW spectra, elbow_summary.csv"
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