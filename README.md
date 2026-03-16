# What Makes Unlearning Brittle? A Mechanistic Study of Parameter-Space Interventions

This repository contains a pipeline for creating unlearned large language models using various unlearning algorihtms, and then analyzing how these methods alter models in both parameter and activation spaces. The goal is to identify mechanistic signatures that distinguish deep representational change from shallow parameter patching.

## Initial Proposal

The field of machine unlearning proposes a range of post-training interventions intended to remove sensitive or harmful information from large language models. However, these methods are brittle under further fine-tuning or adversarial tampering, and there is very limited mechanistic understanding of why unlearning approaches are so shallow. I would like to study unlearning methods through the lens of mechanistic interpretability, treating them as targeted parameter-space interventions whose internal effects can be directly analyzed. 

I analyze unlearning’s internal effects using various model-agnostic diagnostics that characterize how much models change, where those changes occur, and how high-dimensional they are, using layer-wise norms and effective rank, alongside activation-level diagnostics on “forget” and “retain” datasets. Initial results compare baseline, pretraining-filtered (demonstrating actual ignorance), and post-training unlearned models. This reveals distinct regimes: pretraining-time filtering induces large, high-rank, distributed updates consistent with deep representational change, while post-training methods such as Circuit Breaking with Latent Adversarial Training produce small, low-rank, attention-localized edits that leave feature computation largely intact. Activation norms appear to show that these differences are not explained by global suppression.

The project will expand across a broader set of unlearning algorithms (e.g., RMU, gradient-based, latent adversarial), and will also study how training dynamics—such as optimizer choice—shape the geometry of unlearning updates. Additional diagnostics may include representation-level analyses such as sparse autoencoders. By identifying mechanistic signatures associated with brittleness and partial robustness—such as update rank, depth, and distribution—this work aims to equip researchers with the information needed to design more effective approaches to tamper-resistant unlearning, rather than claiming success for existing methods.

---

## Quick Overview

We essentially have three workflows in this repository:

### Experiment

This is governed by `experiment/pipeline.sh`, which runs experiments on one or more models. We should ensure all experiments run via this file, with the general parameters already in place, following its outputs, etc.

We have steps, each of which corresponds to a file in the `experiment` folder.

See [Experimental Pipeline](#experimental-pipeline) for details.

### Unlearning

This is governed by `unlearn/run_unlearn.sh`, which creates a new unlearned model from scratch. We should similarly ensure all unlearning runs use this workflow. 

See [Unlearning](#unlearning) for available methods and tuning guidance.

### Inference

This is handled in the `infer/` folder with both command-line and web UI options. See [Inference](#inference) for usage examples.

## Style

Let's keep code clean, DRY, not littered with acronyms, clearly commented, etc. This is a learning resource for us as much as an experimental workflow. Prefer descriptive variable and function names (`dataset`, `example`, `question`) over abbreviations (`ds`, `ex`, `q`), for example.

## Testing

Please add tests for any new functionality. You can run all tests using:

`uv run tests/run_tests.py`

---

## Datasets

`uv run create_datasets.py` creates two text datasets that serve in training for unlearning, and as *probes* for activation-level analyses:

| Dataset | Source | Purpose |
|---|---|---|
| `forget.txt` | WMDP-Bio questions | Text the model *should* have forgotten — the "target" of unlearning |
| `retain.txt` | WikiText-2 | Benign text the model *should* still handle well — the "control" |

These are analogous to stimulus and control conditions in an experiment. Every activation-level diagnostic (Steps 3, 5–6, 8–9, 11–12) runs on *both* datasets, measuring whether interventions selectively affect forget-domain processing while preserving retain-domain processing.

---

## Pre-Requisites For Experimenting and Unlearning


- Add `HF_TOKEN`and `WANDB_API_KEY` to `.env`.
- Ensure the `uv` package manager is installed. (`pip install uv`)

---

## Experiment

```bash
# Compare the two default models (unfiltered → e2e-strong-filter)
./experiment/pipeline.sh

# Compare any two models
MODEL_A=EleutherAI/deep-ignorance-e2e-strong-filter \
MODEL_B=EleutherAI/deep-ignorance-e2e-strong-filter-adversarial \
./experiment/pipeline.sh

# Also run the tuned lens in Step 5 (slow — ~1hr per model; logit lens only by default)
ENABLE_TUNED_LENS=1 ./experiment/pipeline.sh
```

Already-completed steps are automatically skipped (pass `--force` to rerun).

### Statistical Robustness & Error Bars

The pipeline includes **multi-seed support** for stochastic experiments. By default, analyses that depend on random sampling (Steps 3, 5, 7–12) run with 3 seeds and automatically compute error bars:

```bash
# Default: 3 seeds for robust statistics
./experiment/pipeline.sh

# Custom seeds for more robust estimates
SEEDS="42 123 456 789 999" ./experiment/pipeline.sh

# Single seed (faster, less robust)
SEEDS="42" ./experiment/pipeline.sh
```

**What gets multi-seed treatment:**
- **WMDP accuracy** (Step 5): Train/test splits and probe training
- **Activation analyses** (Steps 3, 8, 9, 11, 12): Text sampling variability — each seed shuffles the dataset with a different RNG and takes the first `--max-samples` lines, so each seed evaluates a different random subset of texts. Error bars reflect how stable the measured activation patterns are across different samples from the forget/retain sets.
- **Null space analysis** (Step 7): Weight matrix sampling robustness

Parameter comparison (Step 1) remains deterministic and doesn't use multiple seeds since it computes exact differences between model weights.

**Results structure:**
```
outputs/
  comparison/
    activation_separation/
      summary.json              # mean ± std aggregated across seeds
      separation_metrics.csv    # includes _std columns
      seed_42/                  # individual seed results (for debugging)
      seed_123/
      seed_456/
```

---

### Experimental Pipeline

The pipeline compares **any two models** (`MODEL_A` and `MODEL_B`) across a suite of parameter-space and activation-space diagnostics. Both models must share the same architecture.

```
MODEL_A  →  MODEL_B
```

Outputs are saved under `outputs/<MODEL_A>__to__<MODEL_B>/`. To compare a different pair, just set `MODEL_A` and `MODEL_B` — no other changes needed.

---

### Diagnostics

```mermaid
graph TD
    A["Parameter-Space"] --> C["How much did parameters change?<br/>Where? In what directions?"]
    B["Activation-Space"] --> D["How do those changes affect<br/>what the model computes?"]
    B --> G["At what layer is the forget-set knowledge<br/>encoded in each model?"]
    C --> E["Mechanistic Signature<br/>of the Intervention"]
    D --> E
    G --> E
```

---
<!-- 
#### Step 0: Benchmark Evaluation (`experiment/eval.py`)

**Question:** *Does the model still have general capabilities, and has it forgotten the target knowledge?*

Runs a suite of benchmarks on each model before the expensive mechanistic diagnostics. This immediately identifies models that have catastrophically collapsed during unlearning (e.g., repeating degenerate tokens), and checks whether the target WMDP-Bio knowledge has actually been suppressed.

The script automatically uses `device_map='auto'` with CPU offloading to handle large models and prevent OOM errors. When GPUs are available, it selects the one with the most free memory using `pick_best_gpu()`.

| Benchmark | Metric | What it tells you |
|---|---|---|
| **MMLU** | Overall + per-subject accuracy | Collapsed models score near random chance (0.25) or below |
| **WMDP-Bio** | MCQ accuracy | Should drop toward chance (0.25) in a well-unlearned model |
| **HellaSwag** | Accuracy | Checks commonsense reasoning is preserved |
| **TruthfulQA** | Accuracy | Checks general truthfulness is preserved |

> **Note:** Results are stored **per-model** (not per-comparison) since benchmarks evaluate a single model's capabilities.

--- -->

#### Step 1: Parameter Statistics (`experiment/collect_weight_comparison.py`)

**Question:** *How large is the intervention, and where is it concentrated?*

For every weight matrix `W` in the model, this computes:

| Metric | Formula | What it tells you |
|---|---|---|
| **Relative Frobenius norm** of $\Delta W$ | $\frac{\lVert \Delta W \rVert_F}{\lVert W \rVert_F}$ | Normalized magnitude of change — what fraction of the original weight moved? Comparable across layers regardless of matrix size. |
| **Frobenius norm** of $\Delta W$ | $\lVert \Delta W \rVert_F = \sqrt{\sum_{ij} \Delta W_{ij}^2}$ | Raw total magnitude (unnormalized; also recorded for completeness) |
| **Frobenius norm** of $W$ (baseline) | $\lVert W_a \rVert_F$ | Absolute weight scale of the baseline model per matrix |
| **Frobenius norm** of $W$ (unlearned) | $\lVert W_b \rVert_F$ | Absolute weight scale of the unlearned model per matrix — compare with baseline to see if norms grow or shrink |
| **Spectral norm** of $\Delta W$ (relative) | $\frac{\sigma_1(\Delta W)}{\sigma_1(W)}$ | Relative worst-case amplification — how much did the dominant singular direction shift? High spectral + low stable rank = a sharp rank-1 spike. |
| **Spectral norm** of $W$ (baseline) | $\sigma_1(W_a)$ | Dominant singular value of the baseline weight matrix |
| **Spectral norm** of $W$ (unlearned) | $\sigma_1(W_b)$ | Dominant singular value after unlearning — shift reveals changes to the leading computational direction |
| **Stable rank** of $\Delta W$ | $\frac{\lVert \Delta W \rVert_F^2}{\lVert \Delta W \rVert_2^2}$ | Effective dimensionality of the update. A rank-1 perturbation (e.g., LoRA-style) gives stable rank $\approx 1$. A full-rank rewrite gives stable rank $\approx \min(m,n)$. |
| **Stable rank** of $W$ (baseline) | $\frac{\lVert W_a \rVert_F^2}{\lVert W_a \rVert_2^2}$ | Baseline dimensionality for comparison |
| **Stable rank** of $W$ (unlearned) | $\frac{\lVert W_b \rVert_F^2}{\lVert W_b \rVert_2^2}$ | Whether unlearning increases or decreases the effective rank of the weight matrices |
| **Empirical rank** (opt-in: `--empirical-rank`) | $\min k$ s.t. $\sum_{i}^{k} \sigma_i^2 \geq 0.99 \cdot \sum \sigma_i^2$ | Discrete count of dimensions capturing 99% of variance (requires full SVD, off by default) |

These metrics are collected at the per-matrix level, then aggregated across three levels of granularity and saved as four CSVs:

| CSV | Granularity | Key use |
|---|---|---|
| `per_matrix.csv` | One row per weight matrix | Full detail; basis for all other aggregations |
| `per_component.csv` | One row per component type, averaged across all layers | High-level "which part of the model changed most?" |
| `per_layer.csv` | One row per (layer, component) pair | Layer-by-layer breakdown within each component |
| `per_coarse_layer.csv` | One row per (layer, coarse group) pair | Backward-compatible `attn` vs `mlp` split used by Step 4 |

The four **component** types map onto the internal structure of each transformer block:

| Component | What it covers | Coarse group |
|---|---|---|
| `qkv` | Query, key, and value projection weights | `attn` |
| `proj` | Attention output projection | `attn` |
| `mlp_expand` | First MLP linear (hidden → 4×hidden) | `mlp` |
| `mlp_contract` | Second MLP linear (4×hidden → hidden) | `mlp` |

Six PNG plots are generated per component (one panel per component type — `qkv`, `proj`, `mlp_expand`, `mlp_contract`) and automatically uploaded to W&B under the `weight_plots/` key:

| File | What it shows |
|---|---|
| `relative_frobenius.png` | Relative Frobenius norm of ΔW per layer — how much each layer changed, normalized by original weight magnitude |
| `stable_rank_deltaW.png` | Stable rank of ΔW per layer — effective dimensionality of the update |
| `relative_spectral_norm.png` | Relative spectral norm (σ₁(ΔW) / σ₁(W)) per layer — worst-case amplification |
| `absolute_frobenius.png` | ‖W‖_F per layer for **both** baseline and unlearned model overlaid — shows raw weight scale and how it shifts |
| `absolute_stable_rank.png` | Stable rank of W per layer for **both** models — shows how the baseline dimensionality changes after unlearning |
| `absolute_spectral_norm.png` | σ₁(W) per layer for **both** models — shows how the dominant singular value shifts across layers |

**Why this matters:** If unlearning produces low-rank, localized updates (small relative $\lVert \Delta W \rVert_F$ concentrated in a few layers) while filtering produces high-rank, distributed updates, that's direct evidence that unlearning is a *shallow patch* rather than a *deep restructuring*. The component breakdown further reveals *where* within each layer the changes land — e.g., if edits concentrate in `proj` (attention output) rather than `mlp_expand`/`mlp_contract` (knowledge storage), that suggests the intervention is redirecting routing rather than erasing stored associations.

> [!NOTE]
> **Kyungeun's addition.** The original script aggregated all attention weights together and all MLP weights together (`attn`/`mlp`). Kyungeun refactored the analysis to track four finer-grained components — `qkv`, `proj`, `mlp_expand`, `mlp_contract` — separately across layers, added cosine similarity and element-wise diff stats to every row, and introduced the `per_component.csv` and `per_layer.csv` (granular) outputs. The old coarse aggregation is preserved as `per_coarse_layer.csv` so that Step 4 still works unchanged.

---

#### Step 1.5: Singular Value Spectrum Analysis (`experiment/singular_value_spectrum_analysis.py`)

**Question:** *What is the effective rank of the weight update, and how does unlearning reshape the singular value spectrum?*

While Step 1's stable rank gives a single scalar summary of dimensionality, this step plots the **full sorted singular value spectrum** of each weight matrix — baseline $W_a$, unlearned $W_b$, and the update $\Delta W = W_b - W_a$ — revealing the shape of the rank structure rather than just a number.

All spectra are normalised by their leading singular value ($\sigma / \sigma_1$) so that matrices at different scales are directly comparable on a shared $[0, 1]$ axis.

For each matrix, the **elbow index** (effective rank at the spectrum drop-off) is computed using the maximum-distance-from-chord method, giving a discrete count of "meaningful" singular directions.

| Output | Description |
|---|---|
| `<component>_layer<N>.png` | 3-panel overlay for representative layers: (1) $W_a$ vs $W_b$ spectra with elbow markers, (2) $\Delta W$ spectrum, (3) log-scale view of all three |
| `dW_spectrum_<component>.png` | $\Delta W$ spectra overlaid across all layers for one component — shows how update rank varies by depth |
| `elbow_by_layer.png` | Line chart of elbow index vs layer for baseline, unlearned, and $\Delta W$ per component |
| `sv_spectrum.png` | Summary bar chart of elbow indices across representative layers |
| `elbow_summary.csv` | Per-matrix elbow indices and shift ($\text{elbow}_B - \text{elbow}_A$) |

**Why this matters:** A low-rank $\Delta W$ (sharp elbow, few dominant singular values) indicates the update lives in a small subspace — consistent with LoRA-style or shallow edits that could be easily reversed. A high-rank $\Delta W$ with a gradual decay suggests distributed restructuring. Comparing the spectra of $W_a$ and $W_b$ directly reveals whether unlearning preserves or reshapes the model's intrinsic dimensionality at each layer.

> [!NOTE]
> This step is weight-only and deterministic — no text data or forward passes are required.

---

#### Step 3: Activation Norms (`experiment/collect_activation_comparison.py`)

**Question:** *Does the intervention globally suppress or amplify activations?*

For each layer, computes the mean L1 and L2 norms of hidden states per token, plus the norm of the *difference* in activations ($\lVert h_{\text{modified}} - h_{\text{base}} \rVert$). Run on both forget and retain texts.

| Norm | Formula (per token) | What it captures |
|---|---|---|
| **L1** | $\sum_i \lvert h_i \rvert$ | Total activation mass—sensitive to diffuse, low-magnitude changes across many dimensions |
| **L2** | $\sqrt{\sum_i h_i^2}$ | Activation magnitude—sensitive to large spikes in individual dimensions |

Both are averaged across all tokens (weighted by attention mask). They are **not** divided by hidden dimension since all models share the same architecture, so they are directly comparable across models and layers.

**Why this matters:** If norms are similar between base and unlearned models but different for the filtered model, it means unlearning doesn't achieve suppression through reducing activation magnitudes—it's doing something more subtle (or less effective). L1 and L2 capture different aspects: L1 is more sensitive to many small changes spread across dimensions, while L2 is dominated by the largest components.

---

#### Step 4: MLP vs Attention Breakdown (`experiment/analyze_mlp_vs_attn.py`)

**Question:** *Are the changes concentrated in MLP (knowledge storage) or Attention (routing/composition)?*

Takes the per-matrix stats from Step 1 and computes the ratio of MLP change to Attention change at each layer. Addresses the mechanistic hypothesis that knowledge is primarily stored in MLP layers (the "key-value memory" view from [Geva et al](https://arxiv.org/pdf/2012.14913).), while attention layers handle routing.

**Why this matters:** If unlearning only modifies attention layers, it might be redirecting *routing around* the knowledge rather than erasing it — explaining why adversarial fine-tuning can recover the information.

---

#### Step 5: Layer-wise WMDP Accuracy (`experiment/layerwise_wmdp_accuracy.py`)

> [!NOTE]
> **Tuned lens is opt-in** (`ENABLE_TUNED_LENS=1`). By default, only the logit lens runs (~3 min total across all three models). The tuned lens trains a per-layer affine probe for all 33 layers of each model, costing ~1hr per model (~3hrs total). It is more accurate at early layers where the logit lens is unreliable, but the key signals for this project — the late-layer steady climb and the final-output suppression — are equally visible in the logit lens. Only enable the tuned lens if you specifically need reliable early-layer readings.

**Question:** *At which layer does the model "know" the answer to WMDP-Bio questions?*

Evaluates WMDP-Bio multiple-choice accuracy at every transformer layer using a **logit lens** (default) or **tuned lens**:

- **Logit lens:** Applies the model's own final LayerNorm + unembedding head (`embed_out`) to intermediate hidden states. Zero training cost.
- **Tuned lens:** Trains a per-layer affine transform (`nn.Linear`) mapping hidden states → vocab logits. More accurate at early layers but requires a training pass on held-out WMDP data (default: 30%).

> [!NOTE]
> **Logit lens reliability at early layers.** The logit lens applies the *final* model's LayerNorm and unembedding head to *intermediate* hidden states. At early layers, those hidden states live in a very different distribution than what the unembedding head was trained on, so accuracy readings at those layers are unreliable — they'll appear to hover near chance (0.25) regardless of what the model actually "knows" at that point. This isn't a bug; it's a fundamental limitation of the zero-cost approach. The **tuned lens** trains a lightweight affine correction per layer to compensate for this distribution mismatch, giving more interpretable early-layer results at the cost of a training pass.

For each question, the lens computes the average log-probability of each answer choice's tokens appended to the question, then picks the highest-scoring choice — the same MCQ scoring convention used by `lm-evaluation-harness`.

| Metric | What it tells you |
|---|---|
| **Accuracy** | Fraction of WMDP questions answered correctly using representations up to this layer |
| **Δ from final** | How much worse/better this layer is compared to reading from the model's final output |

**Why this matters:** A base model will show WMDP accuracy ramping up through mid-to-late layers — knowledge "crystallizes" as representations flow through the network. In a well-unlearned model, you'd expect accuracy to stay near chance (0.25 for 4-way MCQ) at *every* layer, not just the final one. If accuracy is high at intermediate layers but drops at the output, the knowledge is merely hidden, not erased — and a simple probe or fine-tuning attack could recover it.

> **Note:** Results are stored **per-model** (not per-comparison).

---

#### Step 7: Null Space & Subspace Analysis (`experiment/null_space_analysis.py`)

**Question:** *Is the update low-rank, and do the principal subspaces shift?*

For 50 sampled weight matrices, computes full SVD and measures:

| Metric | What it tells you |
|---|---|
| **Top-10 SV variance ratio** | What fraction of ΔW's energy is in its top 10 singular directions? High → very low-rank update. |
| **Effective rank** | How many singular values needed to capture 99% of variance |
| **Subspace alignment** (Grassmann distance) | Do the top-k singular vectors of W_base and W_modified span similar subspaces? High alignment → the intervention didn't change *what directions* the matrix uses, only *how much* it uses them. |

**Why this matters:** A low-rank ΔW with high subspace alignment means the unlearning intervention is a small perturbation within the existing computational manifold — it didn't rewire the representations, it just nudged the gains. This is precisely the geometric signature of brittleness: a small counter-perturbation (fine-tuning) can undo it.

---

#### Step 8: Activation Separation (`experiment/activation_separation_analysis.py`)

**Question:** *Can you tell forget-text activations apart from retain-text activations? Does the intervention change this?*

At each layer, extracts the centroid of forget-text activations and retain-text activations, then measures their separation via:

| Metric | What it captures |
|---|---|
| **Cosine distance** between centroids | Direction-based separation |
| **AUC** (linear classifier) | How linearly separable are the two distributions? |
| **Variance ratio** | Between-class vs. within-class variance (like Fisher's discriminant) |

**Why this matters:** If unlearning *increases* the separation between forget and retain activations (pushes them apart), that's evidence the model is actively routing forget-text to a different computational path. If separation stays similar, the model treats both text types the same way internally — it hasn't genuinely distinguished "knowledge to suppress."

---

#### Step 9: Activation Covariance Analysis (`experiment/activation_covariance_analysis.py`)

**Question:** *Does the intervention change the shape of the activation distribution?*

Computes the eigenvalue spectrum of the activation covariance matrix at each layer and measures:

| Metric | What it captures |
|---|---|
| **Effective rank** (of covariance) | How many dimensions do activations meaningfully occupy? |
| **Spectral entropy** | How uniform is the energy distribution across dimensions? |
| **Wasserstein distance** | How much did the spectrum change between base and modified model? |

A key output is the **selectivity ratio**: (Wasserstein distance on forget text) / (Wasserstein distance on retain text). High selectivity = the intervention specifically reshapes forget-domain representations while leaving retain-domain representations intact.

**Why this matters:** This captures something the norms miss — two distributions can have identical norms but completely different *shapes*. If filtering fundamentally restructures the covariance (high Wasserstein, changed effective rank) while unlearning barely disturbs it, that's evidence the representations aren't actually changing.

---

#### Step 10: MLP Nullspace Alignment (`experiment/mlp_nullspace_alignment.py`)

**Question:** *Does ΔW lie in the nullspace of the original W?*

Decomposes each MLP update ΔW into components that lie in the **column space** vs. **null space** of the original weight matrix W.

- **Nullspace component** ("off-manifold"): Changes orthogonal to what W originally computed. These add new directions without disrupting existing computations.
- **Column space component** ("on-manifold"): Changes that directly interfere with existing computations.

**Why this matters:** If unlearning updates are primarily in the nullspace, the model's existing computations are barely disturbed — the "unlearned" knowledge may still flow through the same channels, just with a small additive correction that's easy to remove. True knowledge erasure should require on-manifold changes that destroy the original computation.

---

#### Step 11: Row Space Projection (`experiment/row_space_projection_analysis.py`)

**Question:** *Do activations from forget-text align more with the directions the intervention modified?*

Computes the SVD of ΔW at each MLP layer and measures how much the *input activations* project onto the row space (input-side principal directions) of ΔW.

If forget-text activations have high projection onto ΔW's row space while retain-text activations don't, the update is *precisely targeted* — it modifies exactly the directions that forget-text activates. The **selectivity ratio** quantifies this.

**Why this matters:** This is perhaps the most mechanistically informative diagnostic. It directly tests whether the intervention is *geometrically aligned* with the specific input patterns it needs to suppress. High selectivity + low rank = a surgical intervention that only fires on forget-domain inputs. Low selectivity = a blunt instrument that affects everything equally. And crucially, high selectivity + low rank is also the easiest to undo: just learn a small correction in that same low-dimensional subspace.

---

#### Step 12: Local Lipschitz Analysis (`experiment/local_lipschitzness_analysis.py`)

**Question:** *Did the intervention make the model's output more or less sensitive to input perturbations?*

Estimates the local Lipschitz constant by perturbing input embeddings with small noise (ε-balls) and measuring how much the output changes. Also computes gradient norms and output variance under perturbation, separately for forget and retain texts.

| Outcome | Interpretation |
|---|---|
| Forget text becomes **rougher** (higher Lipschitz) | Model is unstable on forget inputs — outputs shift erratically |
| Forget text becomes **smoother** (lower Lipschitz) | Model learned to ignore/suppress forget-domain features |
| Retain text stays **similar** | Intervention didn't damage general capabilities |

**Why this matters:** A model that becomes rougher on forget text hasn't *learned to not know* something — it's in an unstable regime where small pushes (fine-tuning) can tip it back. Smoothness changes are a direct indicator of whether the loss landscape around forget-domain inputs is fundamentally reshaped or just locally perturbed.

---

### The Big Picture

The diagnostics answer an escalating series of questions:

| Level | Question | Steps |
|---|---|---|
| **Capabilities** | Does the model still work at all? | 0 |
| **Magnitude** | How much changed? | 1 |
| **Location** | Where — MLP or Attention? Which layers? | 3, 4 |
| **Knowledge Depth** | At which layer does the model know WMDP answers? | 5 |
| **Geometry** | What shape is ΔW? Low-rank? Nullspace-aligned? | 7, 10 |
| **Function** | Do activations actually change on target text? | 8–9 |
| **Precision** | Is the change *targeted* at forget-domain inputs? | 11 |
| **Stability** | Is the new behavior robust or fragile? | 12 |

The thesis prediction is that unlearning methods (CB-LAT) will show: small magnitude, attention-localized, low-rank, nullspace-aligned, minimal activation change, low selectivity, and increased roughness — the full mechanistic signature of a brittle intervention. While filtering will show the opposite across every dimension.

---

### Output Structure

All results are saved under a single root (default `outputs/`):

```
outputs/
  <comparison>/                        # Steps 1–4, 7–12: per model-pair
    weight_comparison/     per_matrix.csv, per_component.csv, per_layer.csv, per_coarse_layer.csv
    param_plots/           Layer locality, stable rank, rank comparison PNGs
    activation_comparison/ activation_comparison.csv
    activation_plots/      Activation norms, activation diffs PNGs
    mlp_attn_analysis/     summary CSV + plots
    null_space_analysis/   null_space_results.csv + plots
    activation_separation/ separation metrics + plots
    activation_covariance/ covariance spectra + plots
    mlp_nullspace/         alignment metrics + plots
    row_space_projection/  projection metrics + plots
    lipschitzness/         Lipschitz estimates + plots

  <model>/                             # Steps 0, 5: per individual model
    evals/                 summary.json, high_level_summary.md
    wmdp_logit_lens/       wmdp_lens_results.csv, summary.json + plot
    wmdp_tuned_lens/       wmdp_lens_results.csv, summary.json + plot
```

> **Tip:** The pipeline automatically skips steps whose output already exists. Use `./experiment/pipeline.sh --force` to regenerate everything.

---

## Unlearning

To create an unlearned model, you would use 

```bash
# Train a single method with defaults
./unlearn/run_unlearn.sh cb_lat
# -> unlearned_models/EleutherAI_deep-ignorance-unfiltered__cb_lat__ep1_lr1e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7/

# Override hyperparameters (automatically encoded into folder name)
EPOCHS=2 LR=5e-6 ./unlearn/run_unlearn.sh cb_lat
# -> unlearned_models/EleutherAI_deep-ignorance-unfiltered__cb_lat__ep2_lr5e-06_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7/

# Use the Muon optimizer instead of AdamW (appended to folder/W&B run name)
OPTIMIZER=muon EPOCHS=3 LR=3e-05 BATCH_SIZE=32 ./unlearn/run_unlearn.sh ga
# -> unlearned_models/...ga__ep3_lr3e-05_bs32_rw1.0_ml2048_optmuon/

# Create all 12 unlearned models with default hyperparameters
./unlearn/create_all_unlearning_models.sh
```

However, more likely, you are going to want to sweep through various hyperparameters. This is done by wrapping around the `run_unlearn.sh` script. For example:

```bash
# Run the same sweep in parallel across multiple GPUs (see Multi-GPU below)
./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn.sh

# Run a comprehensive hyperparameter sweep across all methods (96 models evaluated)
# This uses NO_SAVE=1 by default so the 14GB+ checkpoints aren't saved to disk.
./unlearn/sweep_unlearn.sh

# On managed GPU hosts (RunPod, Lambda, Vast.ai) where the system Python already
# has CUDA-enabled torch pre-installed, bypass uv's isolated venv entirely:
PYTHON=python3 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn_kl_cb32.sh
```

You can then [run the following](#evaluating-unlearned-models) which will create `best_unlearning_models.md` to help you work out which unlearned model is best (high MMLU, low WMDP).

```bash
uv run unlearn/analysis/analyze_runs.py
```

Once you've found that model, push it to HuggingFace:

```bash
PUSH_TO_HUB=1 NO_EVAL=1 EPOCHS=3 LR=5e-05 BATCH_SIZE=8 RETAIN_WEIGHT=1.0 BETA=0.1 \
    ./unlearn/run_unlearn.sh npo
```

Output directories are auto-generated by `unlearn.py` based on the method and all its relevant hyperparameters, so different runs never overwrite each other. After training, eval benchmarks (MMLU, WMDP Robust, WMDP Cloze) run automatically and results are logged to W&B. Add `--push-to-hub` (or `PUSH_TO_HUB=1` in the shell) to upload the finished model to HuggingFace. `sweep_unlearn.sh` runs with `NO_SAVE=1` by default to avoid filling the disk with checkpoints.

> [!NOTE]
> **Memory and Hardware Optimizations.** Both `unlearn.py` and `experiment/eval.py` contain several layers of optimizations designed to prevent OOM errors and make efficient use of available hardware:
>
> **GPU selection and sharding (`unlearn.py`).**  Before loading any weights, `filter_gpus_by_free_vram()` queries `nvidia-smi` and restricts `CUDA_VISIBLE_DEVICES` to GPUs with ≥ 10 GiB free. The model is then loaded with `device_map="auto"` (Accelerate / HuggingFace), which automatically shards layers across all visible GPUs. A `max_memory` budget is passed to Accelerate — computed by `compute_training_max_memory()` with a **6× model-size multiplier** (fp32 master weights + Adam `m`/`v` states) plus a per-GPU activation buffer that scales with batch size — so Accelerate spreads layers evenly rather than packing everything onto the first device.
>
> **Gradient checkpointing (`unlearn.py`).** `model.gradient_checkpointing_enable()` is called after loading. This trades a ~30% compute overhead for a significant VRAM reduction: activations are not stored during the forward pass and are recomputed on demand during backprop, cutting peak memory roughly in half for deep models.
>
> **Mixed-precision training (`unlearn.py`).** When `--dtype bf16` is used on CUDA, the HF Trainer runs forward/backward passes in bf16 (half the weights are live in VRAM at once) while keeping fp32 master copies for the Adam optimizer. This is strictly better than naive "everything in bf16" — the fp32 Adam buffers preserve small gradient updates that would round to zero in bf16.
>
> **Chunked cross-entropy (`unlearn.py`).** Standard `F.cross_entropy(logits.view(-1, V), ...)` materialises a full `(B × T, vocab_size)` intermediate tensor — roughly **3 GiB** on the GPU that holds the `lm_head` for a 7B model at batch=32, seq=512, vocab=50K. `chunked_cross_entropy()` processes `chunk_size` batch elements at a time, freeing each chunk before allocating the next. Peak memory scales with `chunk_size` (default 4 → ~0.4 GiB) instead of the full batch, while producing bit-for-bit identical results. The same chunking technique is applied in `log_probs_from_logits()` (used by DPO/NPO/SimNPO) to avoid materialising the full `(B, T, vocab_size)` log-softmax tensor.
>
> **TAR weight caching on CPU (`unlearn.py`).** The Task Arithmetic method must cache the original model weights before fine-tuning on the forget set. These are copied to CPU tensors (`non_blocking=True`) rather than cloned on GPU, keeping the full GPU budget free for the training forward/backward passes.
>
> **Explicit memory flushing (`unlearn.py`, `eval.py`).** After saving a model and before launching the evaluation subprocess, `unlearn.py` deletes the model object, calls `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`, `gc.collect()`, and sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation. `eval.py` performs the same flush before and after `lm_eval.simple_evaluate()`.
>
> **Dynamic GPU selection for evaluation (`eval.py`).** `pick_best_gpu()` reads live free-VRAM figures via `torch.cuda.mem_get_info` and pins `CUDA_VISIBLE_DEVICES` to the single GPU with the most headroom. If the best GPU has fewer than 10 GiB free, eval transparently falls back to CPU. On GPU, `device_map="auto"` with a `/tmp/offload` CPU-offload folder is used as a safety net so that models larger than a single GPU's VRAM can still be evaluated via host RAM.
>
> **Batch-size auto-tuning (`eval.py`).** `--batch-size auto` is passed to `lm-eval`, which uses the HuggingFace auto-batching heuristic to find the largest batch that fits in VRAM, maximising throughput without manual tuning.

> [!TIP]
> **Idempotent Sweeps**: The shell scripts (`run_unlearn.sh`, `sweep_unlearn.sh`, `sweep_unlearn2.sh`) are fully idempotent. Before loading any models into GPU memory, they invoke `unlearn.py --check-wandb-only`. This queries the Weights & Biases API to see if a run with the exact matching hyperparameter `run_name` has already finished successfully. If it has, the script skips the training entirely and moves to the next configuration.

> [!NOTE]
> **Evaluation with `--no-save`**: When `--no-save` is specified, the model is still saved *temporarily* to disk so that `eval.py` can load it and run benchmarks. Once evaluation completes, the heavy weight files (`*.safetensors`, `*.bin`, etc.) are automatically deleted to save disk space, while the evaluation logs and configuration are preserved.

> [!NOTE]
> **Skipping evals with `--no-eval` / `NO_EVAL=1`**: When pushing a previously-evaluated model to HuggingFace, you can skip the full benchmark re-run with `NO_EVAL=1`. This is useful when you have already verified MMLU/WMDP metrics via W&B and just want to materialize the weights and upload:
> ```bash
> PUSH_TO_HUB=1 NO_EVAL=1 EPOCHS=3 LR=5e-05 BATCH_SIZE=8 RETAIN_WEIGHT=1.0 BETA=0.1 \
>     ./unlearn/run_unlearn.sh npo
> ```
> The `--no-eval` flag is independent of `--no-save` — you can use both at once (e.g., in sweeps) or just `--no-eval` alone when you want to keep the weights after training.

> [!NOTE]
> **Multi-GPU Parallel Sweeps (`parallel_sweep.sh`)**: Wrap any sweep script with `parallel_sweep.sh` to fan its iterations out across GPU groups automatically — no changes to the sweep scripts needed:
> ```bash
> ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn4.sh
>
> # Override group size manually (useful if probe underestimates)
> GPUS_PER_JOB=4 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn4.sh
>
> # Skip probe entirely and run single job with all GPUs (for OOM debugging)
> SKIP_PROBE=1 ./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn4.sh
> ```
> The wrapper:
> 1. **Loads `.env`** — sources the project `.env` file so `WANDB_API_KEY`, `HF_TOKEN`, etc. are available to all background job subprocesses (not just the Python scripts that call `load_dotenv()`).
> 2. **Probes VRAM** (config-based, fast) — downloads only the model's `config.json` (a few KB), estimates parameter count from architecture fields, and divides training VRAM by per-GPU VRAM from `nvidia-smi`. Uses **10× model size** multiplier for batch_size=32 training (accounts for params + gradients + Adam states + activations + reference models for DPO/NPO methods). No model weights are downloaded.
> 3. **Picks group size** — finds the smallest group size ≥ minimum GPUs needed that divides evenly into total GPU count. Checks all divisors (not just powers of 2) to minimize waste. If the probe fails, defaults to **1 GPU per job** (maximum parallelism).
> 4. **Swaps in a shim** — temporarily replaces `run_unlearn.sh` with a lightweight dispatcher that assigns each sweep config to a free GPU group and returns immediately, so the sweep loop queues the next config without waiting.
> 5. **Waits for all jobs**, then restores the original `run_unlearn.sh`.
>
> **Troubleshooting OOM errors:**
> - If you see CUDA OOM even with multiple GPUs allocated, the probe may be underestimating. Use `GPUS_PER_JOB=8` to force more GPUs per job.
> - For batch_size=32 with methods like SimNPO/DPO that need reference models, you may need all 8 GPUs: `SKIP_PROBE=1`
> - The probe uses a 10× multiplier but actual requirements vary by method and batch size.
>
> Example: 8× A40s (48 GB each), model needs 2 GPUs for training → **4 configs run simultaneously**, quartering total wall-clock time.
> Example: 4× B200s (180 GB each), model fits on 1 GPU → **4 configs run simultaneously**.
>
> `--device auto` (the default) is handled differently by context: **`unlearn.py` training** uses `device_map="auto"` to shard the model across the GPUs in the assigned group. **`experiment/eval.py`** now also uses `device_map="auto"` with CPU offloading by default, combined with `pick_best_gpu()` to select the GPU with the most free VRAM. **Other inference scripts** use `pick_best_gpu()` alone to target whichever single GPU has the most free VRAM — avoiding OOM on already-busy GPUs.

### Evaluating Unlearned Models

After each unlearning run, `unlearn.py` automatically evaluates the model and logs metrics to Weights & Biases. You can fetch all finished runs, rank them, and save a summary using:

```bash
# Fetch finished runs, rank them by WMDP/MMLU, and save as markdown
# Output is automatically written to: unlearn/analysis/best_unlearning_models.md
uv run unlearn/analysis/analyze_runs.py
```

The script ranks models by **Score = MMLU − WMDP (Robust)**, which gives equal weight to retaining capability and suppressing forget-set knowledge. It reads two WMDP-Bio variants from W&B:

| Metric | W&B key | What it measures |
|---|---|---|
| **WMDP (Robust)** | `eval_bench/wmdp_bio_robust/acc` | Standard MCQ with distractor shuffling — harder than vanilla MCQ, less susceptible to positional bias. **Primary ranking metric.** |
| **WMDP (Cloze)** | `eval_bench/wmdp_bio_cloze_verified/acc_norm` | Fill-in-the-blank format: tests whether knowledge is suppressed in a *generative* setting, not just multiple-choice. A model that just learned to avoid MCQ answer tokens would still fail here. |

The cloze test in particular is the hardest to game: it requires the model to *generate the wrong answer* in a free-form setting, not just select the wrong MCQ option.

> [!NOTE]
> Baseline metrics for `deep-ignorance-unfiltered` and `deep-ignorance-e2e-strong-filter` are logged to W&B automatically when you run `unlearn.py` with `--wandb-project cambridge_era` — or you can run them directly:
>
> ```bash
> uv run unlearn/unlearn.py --model EleutherAI/deep-ignorance-unfiltered \
>     --method ga --epochs 0 --wandb-project cambridge_era \
>     --wandb-name EleutherAI/deep-ignorance-unfiltered
> ```
> (Zero epochs = eval only, no weight update.)


### Available Methods

| Method | Type | Key Params | Paper |
|---|---|---|---|
| `ga_simple` | Loss | — | [Jang et al. 2023](https://arxiv.org/abs/2210.01504) |
| `ga` | Loss | `--retain-weight` | [Jang et al. 2023](https://arxiv.org/abs/2210.01504) |
| `grad_diff` | Loss | `--forget-weight` | [Liu et al. 2022](https://arxiv.org/pdf/2203.12817) |
| `dpo` | Loss | `--beta` | [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290) |
| `npo` | Loss | `--beta`, `--retain-weight` | [Zhang et al. 2024](https://arxiv.org/abs/2404.05868) |
| `simnpo` | Loss | `--beta`, `--retain-weight` | [Meng et al. 2024](https://arxiv.org/abs/2405.14734) |
| `rmu` | Activation-Space | `--layer-id`, `--steering-coeff`, `--alpha` | [Li et al. 2024](https://arxiv.org/abs/2403.03218) |
| `cb` | Activation-Space | `--layer-id`, `--steering-coeff`, `--alpha` | [Zou et al. 2024](https://arxiv.org/abs/2406.04313) |
| `lat` | Activation-Space | `--layer-id`, `--lat-eps`, `--lat-steps`, `--retain-weight` | [Casper et al. 2024](https://arxiv.org/abs/2403.05030) |
| `cb_lat` | Activation-Space | `--layer-id`, `--steering-coeff`, `--alpha`, `--lat-eps`, `--lat-steps` | [Zou, et al. ](https://arxiv.org/abs/2406.04313) + [Casper](https://arxiv.org/abs/2403.05030) |
| `tar` | Parameter-Space | `--tar-alpha`, `--tar-lr`, `--tar-epochs` | [Ilharco et al. 2023](https://arxiv.org/abs/2212.04089) |
| `wt_dist` | Parameter-Space | `--wt-noise-std` | [Siddiqui et al. 2025](https://arxiv.org/abs/2505.22310) |
| `wt_dist_reg` | Parameter-Space | `--wt-reg-lambda` | [Siddiqui et al. 2025](https://arxiv.org/abs/2505.22310) |

See `uv run unlearn/unlearn.py --help` for full argument reference.

> [!NOTE]
> **PEFT / LoRA compatibility.** The current script does **full-parameter** fine-tuning (all weights receive gradients). However, every method is compatible with LoRA in principle — the optimizer trains whatever parameters have `requires_grad=True`, and all loss functions operate on **activations/logits**, not weight matrices directly. To use LoRA, wrap the model with a PEFT adapter before the training loop; only the adapter weights will be updated. The adversarial inner loop in LAT/CB-LAT perturbs hidden states (not weights), so it is unaffected. Note: `wt_dist` and `wt_dist_reg` operate directly on weight space, so they are **not** compatible with LoRA adaptors — they require full-parameter access by design.

> [!NOTE]
> **Mixed-precision bf16 (HF Trainer).** Training is handled by a subclassed `transformers.Trainer` (`unlearn/trainer.py`) rather than a raw PyTorch loop. When `--dtype bf16` is set on a CUDA device, the Trainer enables **mixed precision**: the forward pass runs in bf16 (fast), but the master weight copy and Adam `m`/`v` states are kept in fp32 (precise). This is strictly better than naïve "everything in bf16" — the fp32 Adam buffers preserve small gradient updates that would otherwise be rounded to zero in bf16. The Trainer also handles gradient accumulation, gradient clipping, cosine LR scheduling, and (optionally) multi-GPU training via Accelerate — all the same CLI flags apply, nothing in the sweep scripts changed.

---

#### Stability & Gradient Explosion

Some unlearning methods, particularly SimNPO and NPO, can suffer from gradient explosions during training due to numerical instability in their loss functions. This typically manifests as:

- Loss spikes from ~3 to 10+ during early training steps
- Gradient norms jumping from ~5 to 100-300+
- Training becoming unstable and potentially corrupting the model

**Root Cause:** SimNPO uses `logsigmoid(-beta * lp_forget)` where `lp_forget` can be large negative values (normal for language models). When `lp_forget ≈ -50`, the sigmoid argument becomes very positive, causing numerical issues.

**Stability Fixes:**

1. **Gradient Clipping:** Added `--grad-clip` parameter (default: 1.0, recommend: 0.5 for unstable methods)
2. **Lower Beta Values:** Use `--beta 0.01` instead of `0.1` for SimNPO/NPO/DPO to stabilize the sigmoid
3. **Conservative Learning Rates:** Use `3e-05` instead of `5e-05` for reference-free methods
4. **Monitor Early:** Watch for loss spikes in the first 50 training steps and stop immediately if detected

**Safe Hyperparameters for Unstable Methods:**

```bash
# SimNPO/NPO - stabilized
EPOCHS=3 LR=3e-05 BETA=0.01 GRAD_CLIP=0.5 RETAIN_WEIGHT=1.0 ./unlearn/run_unlearn.sh simnpo

# DPO - stabilized
EPOCHS=3 LR=3e-05 BETA=0.01 GRAD_CLIP=0.5 ./unlearn/run_unlearn.sh dpo
```

** Systematic Data Scaling:**

Hyperparameter sweeps should use systematic data scaling by default. Each promising hyperparameter configuration is tested at multiple dataset sizes (1024 → 2048 → 4096) to find the optimal data/compute trade-off before exploring parameter variations.

```bash
# All sweeps should include systematic data scaling
./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn4.sh   # Tests each config at 1024, 2048, 4096
./unlearn/parallel_sweep.sh ./unlearn/sweep_unlearn10.sh  # Same systematic approach for TAR

# Override for specific dataset sizes when needed
MAX_LINES=1024 ./unlearn/run_unlearn.sh simnpo  # Force specific size
MAX_LINES=0 ./unlearn/run_unlearn.sh simnpo     # Full dataset for final evaluation
```

**Methodology:**
1. **Start with 1024 samples** (32 steps × 32 batch size) - one clean epoch
2. **Double systematically** (2048, 4096) until results plateau
3. **Identify optimal dataset size** for each method/config
4. **Focus parameter exploration** at the optimal dataset size
5. **Significant time savings** (~5-10x faster than full dataset sweeps)

> **Note:** NPO is inherently more stable than SimNPO because it uses a reference model as an anchor, but both benefit from the above fixes.

---

#### Algorithm Details

##### GA Simple — Pure Gradient Ascent

The simplest unlearning baseline. Maximizes the cross-entropy loss on the forget set, pushing the model to produce *worse* predictions on forget data. No retain-set regularization.

$$L = -\text{NLL}_{\text{forget}}$$

**Risk:** Without retain loss, the model can degrade globally (catastrophic forgetting of general capabilities).

##### GA — Gradient Ascent with Retain Regularization

Adds a standard NLL term on the retain set to stabilize the model while performing gradient ascent on the forget set.

$$L = -\text{NLL}_{\text{forget}} + \text{NLL}_{\text{retain}}$$

##### GradDiff — Gradient Difference

Similar to GA but explicitly weights the forget-ascent and retain-descent terms separately. The `--forget-weight` parameter controls the trade-off.

$$L = - w \cdot \text{NLL}_{\text{forget}} + \text{NLL}_{\text{retain}}$$

When $w = 1$, this is equivalent to GA. Higher $w$ makes the model unlearn more aggressively at the cost of retain performance.

##### DPO — Direct Preference Optimization

Treats unlearning as a preference problem: retain texts are "chosen" (preferred) and forget texts are "rejected". Requires a frozen **reference model** (a copy of the original).

$$L = -\log \sigma\!\Big(\beta \cdot \big[\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big]\Big)$$

Where $y_w$ = retain (chosen), $y_l$ = forget (rejected), and $\beta$ is the inverse temperature.

##### NPO — Negative Preference Optimization

DPO-inspired but focuses the preference term only on the forget set. The model is penalized for assigning higher log-probability to forget data than the reference model does, plus a standard retain NLL term.

$$L = -\frac{2}{\beta} \cdot \mathbb{E}\!\big[\log \sigma\!\big(-\beta \cdot \log \frac{\pi_\theta}{\pi_{\text{ref}}}\big)\big]_{\text{forget}} + \text{NLL}_{\text{retain}}$$

##### SimNPO — Simple NPO (Reference-Free)

Removes the need for a reference model by directly penalizing the model's own log-probabilities on the forget set. Simpler and cheaper than NPO.

$$L = -\frac{2}{\beta} \cdot \mathbb{E}\!\big[\log \sigma\!\big(-\beta \cdot \text{avg\_logprob}_\theta\big)\big]_{\text{forget}} + \text{NLL}_{\text{retain}}$$

##### RMU — Representation Misdirection for Unlearning

Operates on **hidden-state activations** rather than output logits. At specified layers (`--layer-id`), it pushes forget-set activations toward a fixed random direction while anchoring retain-set activations to their original values (cached before training).

$$L = \sum_{\ell \in \text{layers}} \Big[ \text{MSE}\!\big(h_\ell^{\text{forget}},\; c \cdot \hat{r}_\ell\big) + \alpha \cdot \text{MSE}\!\big(h_\ell^{\text{retain}},\; h_\ell^{\text{cached}}\big) \Big]$$

Where $\hat{r}_\ell$ is a unit-norm random target per layer and $c$ is `--steering-coeff`.

##### CB — Circuit Breakers (Representation Rerouting)

Like RMU but uses **cosine similarity** instead of MSE. This makes the loss invariant to activation magnitude — it only cares about *direction*.

- **Forget:** Maximize cosine similarity between forget activations and the random target direction.
- **Retain:** Minimize `1 − cos(current, cached)` to keep retain activations directionally aligned with originals.

$$L = \sum_{\ell} \Big[ -\cos(h_\ell^{\text{forget}},\; c \cdot \hat{r}_\ell) + \alpha \cdot \big(1 - \cos(h_\ell^{\text{retain}},\; h_\ell^{\text{cached}})\big) \Big]$$

##### LAT — Latent Adversarial Training

Introduces a two-phase optimization to make unlearning robust to adversarial attacks:

1. **Inner loop** (adversary): Find a perturbation $\delta$ injected at the **middle target layer** that *minimizes* the forget-set NLL — i.e., helps the model recall forget data. Uses PGD (Projected Gradient Descent) for `--lat-steps` iterations, constrained to $\|\delta\|_\infty \leq$ `--lat-eps`. Gradients flow only into $\delta$, not the model.

2. **Outer loop** (defender): With $\delta$ frozen and injected, perform gradient ascent on forget NLL + gradient descent on retain NLL. This forces the model to unlearn **even under adversarial pressure**.

$$L_{\text{outer}} = -\text{NLL}_{\text{forget}}^{(\delta^*)} + \text{NLL}_{\text{retain}}$$

##### CB-LAT — Circuit Breakers + Latent Adversarial Training

The most robust method. Combines the adversarial robustness of LAT with the representation-level rerouting of CB:

1. **Inner loop (LAT):** Find adversarial $\delta$ at the middle target layer that helps the model recall forget data (same PGD procedure as LAT).

2. **Outer loop (CB):** With $\delta$ frozen and injected, compute the Circuit Breaker loss — rerouting forget-set activations toward random targets while preserving retain-set activations. The key difference from standalone CB is that the forget activations are collected **with the adversarial perturbation active**, so the model must reroute representations even when an adversary is trying to restore them.

$$L_{\text{outer}} = \sum_{\ell} \Big[ -\cos\!\big(h_\ell^{\text{forget}(\delta^*)},\; c \cdot \hat{r}_\ell\big) + \alpha \cdot \big(1 - \cos(h_\ell^{\text{retain}},\; h_\ell^{\text{cached}})\big) \Big]$$

##### Weight Distortion — Gaussian Noise + Retain Fine-Tuning

From [*From Dormant to Deleted*](https://arxiv.org/abs/2505.22310). Adds isotropic Gaussian noise (σ = `--wt-noise-std`, default 0.02) to **all** model weights before training, then fine-tunes on the retain set only. The noise displaces the model far from the pretrained basin in weight space, making it hard for an adversary to fine-tune back.

$$\theta_0 \leftarrow \theta_{\text{pretrained}} + \mathcal{N}(0, \sigma^2 I)$$
$$L = \text{NLL}_{\text{retain}}$$

##### Weight Distance Regularization — Maximize L2 from Pretrained

Also from [*From Dormant to Deleted*](https://arxiv.org/abs/2505.22310). Minimizes retain NLL while **explicitly maximizing** the L2 distance between the current and pretrained weights. This directly optimizes for tamper-resistance — the larger the distance, the harder it is to recover the original model via fine-tuning.

$$L = \text{NLL}_{\text{retain}} - \lambda \cdot \|\theta - \theta_{\text{pretrained}}\|_2^2$$

Where $\lambda$ is `--wt-reg-lambda` (default 0.1). The paper shows this produces the highest tamper-resistance of all methods tested, outperforming CB, RMU, SCRUB, and even CB-LAT under relearning attacks.

##### TAR — Task Arithmetic for Unlearning

From [*Editing Models with Task Arithmetic*](https://arxiv.org/abs/2212.04089). This method leverages task arithmetic to perform unlearning by subtracting a "forget task vector" from the pretrained model weights. The approach works by first fine-tuning a copy of the base model on the forget set to create a "forget model," then computing the difference vector and subtracting it from the original weights.

**Process:**
1. Fine-tune base model on forget set: $\theta_{\text{forget}} \leftarrow \text{finetune}(\theta_{\text{base}}, \mathcal{D}_{\text{forget}})$
2. Compute forget task vector: $\tau_{\text{forget}} = \theta_{\text{forget}} - \theta_{\text{base}}$
3. Apply negated task vector: $\theta_{\text{unlearned}} = \theta_{\text{base}} - \alpha \cdot \tau_{\text{forget}}$

The `--tar-alpha` parameter controls the strength of the subtraction (default varies by method), `--tar-lr` sets the learning rate for the forget model fine-tuning phase, and `--tar-epochs` determines how many epochs to train the forget model.

**Why this matters:** Task arithmetic provides a simple, training-free way to perform unlearning once the forget task vector is computed. However, the method's effectiveness depends heavily on the assumption that the forget knowledge forms a linearly separable subspace in parameter space, which may not hold for complex, distributed representations.

---

#### Hyperparameter Defaults & Tuning Guide

All methods use **full-parameter training** with a cosine-annealed optimizer, managed by the HF Trainer (`unlearn/trainer.py`). Shared defaults:

| Setting | Default | Flag |
|---|---|---|
| Optimizer | AdamW (or Muon via `OPTIMIZER=muon`) | `--optimizer` |
| Learning rate | 1e-5 | `--lr` |
| Epochs | 1 | `--epochs` |
| Batch size | 4 | `--batch-size` |
| Max sequence length | 512 | `--max-length` |
| Max dataset lines | 1024 (0 for unlimited) | `--max-lines` |
| Gradient clipping | 1.0 (⚠️ 0.5 for unstable methods) | `--grad-clip` |
| Gradient accumulation | 1 | `--grad-accum-steps` |
| Eval split | 10% | `--eval-split` |

**Muon optimizer.** Set `OPTIMIZER=muon` (or `--optimizer muon`) to use `MuonAdamW` (PyTorch's built-in [`torch.optim.Muon`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) + `AdamW`) instead of plain AdamW. Muon applies Newton-Schulz-orthogonalised gradient updates to hidden 2D weight matrices (faster convergence) while keeping standard AdamW for embeddings, biases, norms, and `lm_head`. Requires PyTorch ≥2.6, no external dependencies. The optimizer suffix `_optmuon` is automatically appended to the folder name, W&B run name, and HuggingFace repo name, keeping Muon and AdamW runs cleanly separated.

##### Per-Method Defaults

| Method | Key Param | Default | What to tune if collapsing | What to tune if not unlearning enough |
|---|---|---|---|---|
| `ga` | `--retain-weight` | 1.0 | ↑ to 2–10 | ↓ below 1.0 |
| `grad_diff` | `--forget-weight` | 1.0 | ↓ to 0.1–0.5 | ↑ to 2–5 |
| `dpo` | `--beta` | 0.1 ⚠️ | ↓ beta (0.01) + ↓ lr (3e-05) | ↑ beta (but risk instability) |
| `npo` | `--beta`, `--retain-weight` | 0.1 ⚠️, 1.0 | ↑ retain-weight or ↓ beta (0.01) | ↑ beta or ↓ retain-weight |
| `simnpo` | `--beta`, `--retain-weight` | 0.1 ⚠️, 1.0 | ↑ retain-weight or ↓ beta (0.01) | ↑ beta or ↓ retain-weight |
| `rmu` | `--alpha`, `--steering-coeff` | 100.0, 20.0 | ↑ alpha or ↓ steering-coeff | ↓ alpha or ↑ steering-coeff |
| `cb` | `--alpha`, `--steering-coeff` | 100.0, 20.0 | ↑ alpha or ↓ steering-coeff | ↓ alpha or ↑ steering-coeff |
| `lat` | `--lat-eps`, `--retain-weight` | 0.1, 1.0 | ↑ retain-weight or ↓ lat-eps | ↑ lat-eps or ↓ retain-weight |
| `cb_lat` | `--alpha`, `--lat-eps` | 100.0, 0.1 | ↑ alpha or ↓ lat-eps | ↓ alpha or ↑ lat-eps |
| `wt_dist` | `--wt-noise-std` | 0.02 | ↓ noise (less destruction) | ↑ noise (more disruption) |
| `wt_dist_reg` | `--wt-reg-lambda` | 0.1 | ↓ lambda (less weight push) | ↑ lambda (more distance) |

⚠️ **Gradient explosion risk:** Use `--grad-clip 0.5 --beta 0.01 --lr 3e-05` for stability

> [!TIP]
> **Preventing catastrophic collapse.** If MMLU drops well below the ~45% baseline, the two most effective levers are:
> 1. **Increase `--retain-weight`** (for ga/npo/simnpo/lat) or **`--alpha`** (for rmu/cb/cb_lat) — this strengthens the retain-preservation term relative to the forget-ascent term.
> 2. **Lower the learning rate** (`--lr 5e-6` or `1e-6`) — slower updates give the retain term more weight per step.
> 3. **More epochs with lower LR** — `EPOCHS=3 LR=5e-6` often works better than 1 epoch at 1e-5.
>
> The goal is MMLU within ~3% of the 45% baseline while still showing elevated forget-set NLL.

## Inference

Two inference options are available:

### Command Line Interface

```bash
# Single prompt inference (auto-picks the GPU with most free VRAM)
uv run infer/cli.py --model EleutherAI/deep-ignorance-unfiltered --prompt "What is biotin?"

# Pin to a specific GPU
uv run infer/cli.py --model EleutherAI/deep-ignorance-unfiltered --prompt "..." --device cuda:1

# Force a specific dtype or run on CPU
uv run infer/cli.py --model EleutherAI/deep-ignorance-unfiltered --prompt "..." --device cpu --dtype fp32
```

| Flag | Default | Description |
|---|---|---|
| `--max-tokens` | 200 | Maximum new tokens to generate |
| `--device` | `auto` | `auto` (best GPU), `cuda`, `cuda:N`, `mps`, `cpu` |
| `--dtype` | `auto` | `auto` (bf16 on CUDA, fp16 on MPS, fp32 on CPU), `fp32`, `fp16`, `bf16` |

With `--device auto` (the default), `pick_best_gpu()` from `utils.py` selects the GPU with the most free VRAM (`torch.cuda.mem_get_info`) and loads the model there — avoiding OOM on already-busy GPUs.

### Streamlit Web UI

```bash
# Launch interactive web interface
uv run infer/run.py
```

---

## Appendix A: CSV Column Reference

### `weight_comparison/per_matrix.csv`

One row per weight matrix. The primary output; all other CSVs are derived from this.

| Column | Description |
| :--- | :--- |
| `name` | Full parameter name (e.g., `gpt_neox.layers.10.mlp.dense_h_to_4h.weight`) |
| `layer` | Integer layer index extracted from name |
| `component` | Granular component: `qkv`, `proj`, `mlp_expand`, or `mlp_contract` |
| `shape0` | Matrix rows (output features) |
| `shape1` | Matrix columns (input features) |
| `elements` | Total number of parameters in this matrix |
| `cosine_sim` | Cosine similarity between $W$ and $W'$ (1.0 = unchanged direction) |
| `rel_frobenius` | $\frac{\lVert \Delta W \rVert_F}{\lVert W \rVert_F}$ — fraction of original weight that moved |
| `frobenius_norm` | $\lVert \Delta W \rVert_F$ — raw magnitude of change |
| `fro_norm_normalized` | $\lVert \Delta W \rVert_F / \sqrt{n}$ — size-normalized Frobenius norm |
| `W_fro` | $\lVert W \rVert_F$ — Frobenius norm of the original weight |
| `diff_mean` | Mean element-wise difference |
| `diff_std` | Std of element-wise differences |
| `diff_abs_mean` | Mean absolute element-wise difference |
| `diff_spectral_norm` | $\sigma_1(\Delta W)$ — spectral norm of the update |
| `W_spectral` | $\sigma_1(W)$ — spectral norm of the original weight |
| `dW_spectral_rel` | $\sigma_1(\Delta W) / \sigma_1(W)$ — relative spectral norm |
| `dW_stable_rank` | $\lVert \Delta W \rVert_F^2 / \sigma_1(\Delta W)^2$ — effective dimensionality of the update |
| `W_stable_rank` | Stable rank of the original weight |
| `dW_empirical_rank`* | Number of singular values of $\Delta W$ capturing 99% of variance |
| `W_empirical_rank`* | Number of singular values of $W$ capturing 99% of variance |

\* *Only present when `--empirical-rank` flag is passed (opt-in, requires full SVD).*

### `weight_comparison/per_component.csv`

One row per component type (`qkv`, `proj`, `mlp_expand`, `mlp_contract`), with each metric averaged/min/max/std across all layers. Quick summary of which part of the model changed most overall.

| Column | Description |
| :--- | :--- |
| `component` | Component type |
| `n_layers` | Number of layers with this component |
| `<metric>_mean/min/max/std` | Aggregated statistics for each metric in `per_matrix.csv` |

### `weight_comparison/per_layer.csv`

One row per `(layer, component)` pair — the most useful file for plotting layer-by-layer change profiles broken down by component.

| Column | Description |
| :--- | :--- |
| `layer` | Integer layer index |
| `component` | Granular component: `qkv`, `proj`, `mlp_expand`, or `mlp_contract` |
| `dW_fro_layer` | Root-sum-square of Frobenius norms: $\sqrt{\sum \lVert \Delta W_i \rVert_F^2}$ |
| `W_fro_layer` | Root-sum-square of original Frobenius norms |
| `dW_fro_layer_rel` | `dW_fro_layer / W_fro_layer` |
| `max_dW_spectral` | Max spectral norm of $\Delta W$ across matrices in this component/layer |
| `max_W_spectral` | Max spectral norm of $W$ across matrices in this component/layer |
| `max_dW_spectral_rel` | `max_dW_spectral / max_W_spectral` |
| `mean_dW_stable_rank` | Mean stable rank of $\Delta W$ across matrices |
| `count_mats` | Number of weight matrices aggregated |
| `mean_dW_empirical_rank`* | Mean empirical rank of $\Delta W$ |

\* *Only present when `--empirical-rank` flag is passed.*

### `weight_comparison/per_coarse_layer.csv`

Same aggregation as `per_layer.csv` but using the coarse `attn` / `mlp` grouping (`qkv`+`proj` → `attn`, `mlp_expand`+`mlp_contract` → `mlp`). Used by Step 4 (`analyze_mlp_vs_attn.py`).

| Column | Description |
| :--- | :--- |
| `layer` | Integer layer index |
| `group` | Coarse group: `attn` or `mlp` |
| *(remaining columns)* | Same as `per_layer.csv` above |

### `activation_comparison/activation_comparison.csv`


One row per (layer, split) combination.

| Column | Description |
| :--- | :--- |
| `layer` | Layer index (0 to N) |
| `split` | Dataset split: `forget` or `retain` |
| `model_a_norm_L1` | Mean L1 norm of hidden states for model A (baseline): $\mathbb{E}[\lVert h \rVert_1]$ |
| `model_a_norm_L2` | Mean L2 norm of hidden states for model A (baseline): $\mathbb{E}[\lVert h \rVert_2]$ |
| `model_b_norm_L1` | Mean L1 norm of hidden states for model B (target) |
| `model_b_norm_L2` | Mean L2 norm of hidden states for model B (target) |
| `mean_dh_L1` | Mean L1 norm of the activation difference: $\mathbb{E}[\lVert \Delta h \rVert_1]$ |
| `mean_dh_L2` | Mean L2 norm of the activation difference: $\mathbb{E}[\lVert \Delta h \rVert_2]$ |
