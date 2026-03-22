#!/usr/bin/env bash
set -uo pipefail

# Always run from the project root (parent of experiment/)
cd "$(dirname "$0")/.."

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
# (tagging deferred until MODEL_B is known — see below)

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

# Detect if MODEL_B is already a norm-controlled variant (contains _nrl)
IS_NORM_CONTROLLED=0
if [[ "$MODEL_B" == *"_nrl"* ]]; then
  IS_NORM_CONTROLLED=1
fi

# Group all W&B runs from this pipeline invocation together.
# When MODEL_B is a norm-controlled variant, prefix the group and add a W&B tag
# so every run in this invocation is filterable in the dashboard.
PIPELINE_TAG="pipeline_$(date +%s)"
if [[ "$IS_NORM_CONTROLLED" == "1" ]]; then
  PIPELINE_TAG="norm_controlled_${PIPELINE_TAG}"
  export WANDB_TAGS="${WANDB_TAGS:+${WANDB_TAGS},}norm_controlled"
fi
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$PIPELINE_TAG}"

# Device and dtype settings
PARAM_DEVICE="${PARAM_DEVICE:-auto}"  # auto = cuda > mps > cpu
PARAM_DTYPE="${PARAM_DTYPE:-fp16}"
ACTIVATION_DEVICE="${ACTIVATION_DEVICE:-auto}"
ACTIVATION_DTYPE="${ACTIVATION_DTYPE:-auto}"

# Data paths
FORGET="${FORGET_TEXT:-data/forget.txt}"
RETAIN="${RETAIN_TEXT:-data/retain.txt}"

# ---- W&B completion check ----
# At startup, fetch all finished runs from W&B into a local cache file.
# Subsequent step_complete checks grep this cache — no network calls per step.
WANDB_CACHE="/tmp/wandb_finished_runs_${$}.txt"
echo "[pipeline] Fetching completed runs from W&B..."
if uv run experiment/check_wandb_complete.py --fetch --cache-file "$WANDB_CACHE" 2>/dev/null; then
  WANDB_AVAILABLE=1
  echo "[pipeline] W&B cache ready."
else
  WANDB_AVAILABLE=0
  echo "[pipeline] W&B unavailable — using local sentinel files as fallback."
fi

# Derive the W&B run name from an outdir (matches _derive_run_name in utils.py).
_wandb_run_name_from_outdir() {
  local outdir="$1"
  local stripped="${outdir#outputs/}"
  stripped="${stripped#unlearned_models/}"
  stripped="${stripped#plots/}"
  echo "$stripped" | awk -F/ '{if(NF>=2) print $(NF-1)"/"$NF; else print $NF}'
}

# Fast local check against the cached W&B data.
wandb_step_complete() {
  local outdir="$1"
  if [[ "$WANDB_AVAILABLE" != "1" ]]; then return 2; fi
  local model_a_arg="--model-a $MODEL_A"
  local model_b_arg=""
  if [[ "$outdir" != *"__to__"* ]]; then
    if [[ "$outdir" == *"${MODEL_B_DIR}"* ]]; then
      model_a_arg="--model-a $MODEL_B"
    fi
  else
    model_b_arg="--model-b $MODEL_B"
  fi
  local run_name
  run_name=$(_wandb_run_name_from_outdir "$outdir")
  python3 experiment/check_wandb_complete.py \
    --check --cache-file "$WANDB_CACHE" \
    --run-name "$run_name" $model_a_arg $model_b_arg
}

# ---- Skip-if-complete helper ----
# Usage: step_complete <dir> <sentinel_file>
# W&B is the authority. Local sentinel is only used when W&B is unavailable.
# Returns 0 (true) if the step has already completed.
step_complete() {
  local dir="$1" sentinel="$2"
  if [[ "$FORCE" == "1" ]]; then return 1; fi
  # W&B is authoritative — check it first
  wandb_step_complete "$dir"
  local rc=$?
  if [[ $rc -eq 0 ]]; then return 0; fi
  if [[ $rc -eq 1 ]]; then return 1; fi
  # rc=2: W&B unavailable — fall back to local sentinel
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

  # Run experiment for each seed, skipping seeds that already completed in W&B
  local seed_dirs=()
  local all_seeds_done=true
  for seed in $SEEDS; do
    local seed_outdir="${base_outdir}/seed_${seed}"
    seed_dirs+=("$seed_outdir")

    # Check if this specific seed run already finished
    if [[ "$FORCE" != "1" ]] && wandb_step_complete "$seed_outdir"; then
      echo "    Seed $seed: ✓ already complete in W&B — skipping"
      continue
    fi

    all_seeds_done=false
    mkdir -p "$seed_outdir"
    echo "    Running with seed $seed..."
    uv run "$script" "${extra_args[@]}" --seed "$seed" --outdir "$seed_outdir"
  done

  # Aggregate results across seeds
  echo "    Aggregating results across seeds..."
  uv run experiment/aggregate_multiseed_results.py \
    --seed-dirs "${seed_dirs[@]}" \
    --output-dir "$base_outdir" \
    --sentinel-file "$sentinel"
}

# ---- Resilient step runner ----
# Runs a command; on failure, logs a warning and continues instead of aborting.
# Usage: run_step "Step N: description" command arg1 arg2 ...
STEP_FAILURES=0
run_step() {
  local step_name="$1"
  shift
  if "$@"; then
    return 0
  else
    local rc=$?
    echo ""
    echo "  ✗ $step_name FAILED (exit $rc) — continuing to next step"
    echo ""
    ((STEP_FAILURES++))
    return 0  # swallow the error so the pipeline continues
  fi
}

echo "=========================================="
echo "      MODEL DIFFS ANALYSIS PIPELINE"
echo "=========================================="
echo ""
echo "Model A:  $MODEL_A"
echo "Model B:  $MODEL_B"
if [[ "$IS_NORM_CONTROLLED" == "1" ]]; then
  echo "          ^^^ norm-controlled variant (activation-norm regularised)"
fi
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
  run_step "Step 1" uv run experiment/collect_weight_comparison.py \
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
  run_step "Step 1.5" uv run experiment/singular_value_spectrum_analysis.py \
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
  run_step "Step 3" run_multiseed_experiment "${OUTROOT}/${COMP}/activation_comparison" "activation_comparison.csv" \
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
# STEP 3b: Norm-Controlled Unlearning Experiment
# ============================================
echo ""
echo "=========================================="
echo "STEP 3b: Norm-Controlled Unlearning"
echo "=========================================="
echo "Gradient-ascent unlearning with activation-norm regularisation."
echo "Tests whether unlearning can avoid the characteristic norm drops."

if [[ "$IS_NORM_CONTROLLED" == "1" ]]; then
  echo "  ✓ MODEL_B is already norm-controlled — skipping to avoid infinite loop"
fi

# Norm-controlled unlearning re-runs the SAME method used to produce MODEL_B,
# but with --norm-reg-lambda to anchor activation norms to the base model.
# This lets us compare the original unlearned model (MODEL_B) against a
# norm-controlled variant to see if the norm drops are avoidable.
#
# NORM_CTRL_METHOD is inferred from MODEL_B's name by default (matching the
# same slug logic as utils.infer_method_from_model_name).
NORM_CTRL_LAMBDA="${NORM_CTRL_LAMBDA:-1.0}"
NORM_CTRL_METHOD="${NORM_CTRL_METHOD:-}"

if [[ "$IS_NORM_CONTROLLED" == "1" ]]; then
  :  # already printed skip message above
elif [[ ! -f "$FORGET" || ! -f "$RETAIN" ]]; then
  echo "Warning: Data files missing; skipping norm-controlled unlearning."
elif [[ -z "$NORM_CTRL_METHOD" ]]; then
  # Try to infer the method from MODEL_B's name
  NORM_CTRL_METHOD=$(python3 -c "
import sys; sys.path.insert(0,'.')
from utils import infer_method_from_model_name
m = infer_method_from_model_name('$MODEL_B')
print(m or '')
")
  if [[ -z "$NORM_CTRL_METHOD" ]]; then
    echo "  Could not infer unlearning method from MODEL_B name."
    echo "  Set NORM_CTRL_METHOD=<method> to enable this step."
  else
    echo "  Inferred method: $NORM_CTRL_METHOD (from MODEL_B name)"
  fi
fi

if [[ "$IS_NORM_CONTROLLED" != "1" && -n "$NORM_CTRL_METHOD" && -f "$FORGET" && -f "$RETAIN" ]]; then
  echo ""
  echo "Re-running $NORM_CTRL_METHOD with norm regularisation (λ=$NORM_CTRL_LAMBDA)"
  echo "  Base model: $MODEL_A"
  echo "----------------------------------------"

  # Let unlearn.py auto-generate its outdir; we just need to know the path
  # for the comparison step. build_outdir appends _nrl<lambda> when non-zero.
  NORM_CTRL_OUTDIR=$(python3 -c "
import sys, types; sys.path.insert(0,'.')
from utils import build_outdir
a = types.SimpleNamespace(
    model='$MODEL_A', method='$NORM_CTRL_METHOD',
    epochs=1, lr=1e-5, batch_size=4, max_length=512, max_lines=1024,
    retain_weight=1.0, forget_weight=1.0, beta=0.1, alpha=100.0,
    steering_coeff=20.0, layer_id='5,6,7', lat_eps=0.1, lat_steps=5,
    tar_alpha=1.0, tar_lr=1e-5, tar_epochs=1,
    wt_noise_std=0.02, wt_reg_lambda=0.1,
    norm_reg_lambda=$NORM_CTRL_LAMBDA, optimizer='adamw', grad_accum_steps=1,
)
print(build_outdir(a))
")
  NORM_CTRL_COMP="${MODEL_A_DIR}__to__$(basename "$NORM_CTRL_OUTDIR")"

  if step_complete "$NORM_CTRL_OUTDIR" "config.json"; then
    echo "  ✓ Norm-controlled model already trained — skipping"
  else
    run_step "Step 3b" uv run unlearn/unlearn.py \
      --model "$MODEL_A" \
      --method "$NORM_CTRL_METHOD" \
      --forget-data "$FORGET" \
      --retain-data "$RETAIN" \
      --norm-reg-lambda "$NORM_CTRL_LAMBDA" \
      --device "$ACTIVATION_DEVICE" \
      --dtype "$ACTIVATION_DTYPE" \
      --no-eval
  fi

  echo ""
  echo "Norm-controlled model saved to: $NORM_CTRL_OUTDIR"
  echo "To run the full distinguishability analysis, re-run the pipeline with:"
  echo "  MODEL_B=$NORM_CTRL_OUTDIR ./experiment/pipeline.sh"
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
  run_step "Step 4" uv run experiment/analyze_mlp_vs_attn.py \
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
  run_step "Step 5 (${LENS}, Model A)" run_multiseed_experiment "${OUTROOT}/${MODEL_A_DIR}/wmdp_${LENS}_lens" "summary.json" \
    "experiment/layerwise_wmdp_accuracy.py" \
    --model "$MODEL_A" \
    --lens "$LENS" \
    --device "$ACTIVATION_DEVICE" \
    --dtype "$ACTIVATION_DTYPE"

  echo ""
  echo "Model B: $MODEL_B"
  echo "----------------------------------------"
  run_step "Step 5 (${LENS}, Model B)" run_multiseed_experiment "${OUTROOT}/${MODEL_B_DIR}/wmdp_${LENS}_lens" "summary.json" \
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
run_step "Step 7" run_multiseed_experiment "${OUTROOT}/${COMP}/null_space_analysis" "null_space_visualization.png" \
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
run_step "Step 8" run_multiseed_experiment "${OUTROOT}/${COMP}/activation_separation" "summary.json" \
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
run_step "Step 9" run_multiseed_experiment "${OUTROOT}/${COMP}/activation_covariance" "summary.json" \
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
  run_step "Step 10" uv run experiment/mlp_nullspace_alignment.py \
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
run_step "Step 11" run_multiseed_experiment "${OUTROOT}/${COMP}/row_space_projection" "summary.json" \
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
run_step "Step 12" run_multiseed_experiment "${OUTROOT}/${COMP}/lipschitzness_analysis" "summary.json" \
  "experiment/local_lipschitzness_analysis.py" \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --forget-text "$FORGET" \
  --retain-text "$RETAIN" \
  --device "$ACTIVATION_DEVICE" \
  --dtype "$ACTIVATION_DTYPE"

# ============================================
# STEP 13: Basin Analysis (Goldilocks Distance)
# ============================================
echo ""
echo "=========================================="
echo "STEP 13: Basin Analysis (Goldilocks Distance)"
echo "=========================================="
echo "Correlating per-layer weight distance with unlearning effectiveness."
echo "Requires Steps 1, 3, and 5 (logit lens) to have completed."

BASIN_WEIGHT_CSV="${OUTROOT}/${COMP}/weight_comparison/per_coarse_layer.csv"
BASIN_ACTIVATION_CSV="${OUTROOT}/${COMP}/activation_comparison/activation_comparison.csv"
BASIN_WMDP_A_CSV="${OUTROOT}/${MODEL_A_DIR}/wmdp_logit_lens/wmdp_lens_results.csv"
BASIN_WMDP_B_CSV="${OUTROOT}/${MODEL_B_DIR}/wmdp_logit_lens/wmdp_lens_results.csv"

BASIN_MISSING=""
for f in "$BASIN_WEIGHT_CSV" "$BASIN_ACTIVATION_CSV" "$BASIN_WMDP_A_CSV" "$BASIN_WMDP_B_CSV"; do
  if [[ ! -f "$f" ]]; then
    BASIN_MISSING="${BASIN_MISSING}  missing: $f\n"
  fi
done

if [[ -n "$BASIN_MISSING" ]]; then
  echo "  Skipping — prerequisite CSVs not found:"
  printf "$BASIN_MISSING"
  echo "  Run Steps 1, 3, and 5 first."
elif step_complete "${OUTROOT}/${COMP}/basin_analysis" "summary.json"; then
  echo "  ✓ Already complete — skipping"
else
  run_step "Step 13" uv run experiment/basin_analysis.py \
    --weight-csv "$BASIN_WEIGHT_CSV" \
    --activation-csv "$BASIN_ACTIVATION_CSV" \
    --wmdp-a-csv "$BASIN_WMDP_A_CSV" \
    --wmdp-b-csv "$BASIN_WMDP_B_CSV" \
    --outdir "${OUTROOT}/${COMP}/basin_analysis" \
    --title "${MODEL_B##*/}: Basin Analysis"
fi

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
echo "    basin_analysis/         basin_summary.csv, summary.json, Goldilocks scatter + profile PNGs"
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
if [[ $STEP_FAILURES -gt 0 ]]; then
  echo "WARNING: $STEP_FAILURES step(s) failed during this run."
  echo "  Rerun with --force to retry, or check the logs above for details."
  echo ""
fi
echo "Tip: rerun with --force to regenerate all results."
echo "Tip: set SEEDS=\"42 123 456 789 999\" for more robust statistics."

# Clean up W&B cache
rm -f "$WANDB_CACHE"
echo ""