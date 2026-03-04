#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "scipy",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Activation covariance analysis between forget and retain datasets.

For selected transformer layers, this script:
  1. Extracts flattened hidden-state activations for forget and retain
     splits under both model A (baseline) and model B (target).
  2. Computes the activation covariance matrix and its eigenvalue spectrum.
  3. Derives per-layer metrics: effective rank (99% variance threshold),
     top-k concentration, spectral entropy, and Wasserstein distance
     between spectra to quantify how unlearning reshapes internal
     representations.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    init_wandb,
    log_csv_as_table,
    log_line_series,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_activations_batch(
    model,
    tokenizer,
    texts: List[str],
    layer_index: int,
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract flattened activations at *layer_index* for a list of texts.

    Hidden states are flattened over batch and sequence dimensions,
    returning an (N_tokens, hidden_dim) float32 array.
    """
    all_activations: List[np.ndarray] = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index]
            # Flatten batch and sequence dimensions
            flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
            all_activations.append(flattened.cpu().float().numpy())

    return np.vstack(all_activations)


# ---------------------------------------------------------------------------
# Core covariance metrics (testable, no I/O)
# ---------------------------------------------------------------------------

def compute_covariance_metrics(
    activations: np.ndarray,
    top_k: int = 50,
) -> Dict:
    """Compute covariance matrix and analyze its eigenvalue spectrum.

    Args:
        activations: (N, D) array of hidden-state activations.
        top_k: Number of top eigenvalues to use for concentration metric.

    Returns dict with eigenvalues, explained_var_ratio, effective_rank,
    top_k_concentration, spectral_entropy, max_eigenvalue, and trace.
    """
    mean = activations.mean(axis=0)
    centered = activations - mean

    num_samples = centered.shape[0]
    covariance = (centered.T @ centered) / (num_samples - 1)

    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    total_variance = eigenvalues.sum()
    explained_variance_ratio = eigenvalues / (total_variance + 1e-10)

    # Effective rank: number of eigenvalues to explain 99% variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    effective_rank = int(np.searchsorted(cumulative_variance, 0.99) + 1)

    # Top-k concentration
    top_k_concentration = float(
        explained_variance_ratio[:top_k].sum()
        if len(eigenvalues) >= top_k
        else explained_variance_ratio.sum()
    )

    # Spectral entropy (spread of the spectrum)
    probabilities = eigenvalues / (eigenvalues.sum() + 1e-10)
    probabilities = probabilities[probabilities > 1e-10]
    spectral_entropy = float(-np.sum(probabilities * np.log(probabilities)))

    return {
        "eigenvalues": eigenvalues[:min(100, len(eigenvalues))],
        "explained_var_ratio": explained_variance_ratio[:min(100, len(explained_variance_ratio))],
        "effective_rank": effective_rank,
        "top_k_concentration": top_k_concentration,
        "spectral_entropy": spectral_entropy,
        "max_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        "trace": float(total_variance),
    }


def compare_spectra(
    spectrum_a: Dict,
    spectrum_b: Dict,
) -> Dict[str, float]:
    """Compare two eigenvalue spectra.

    Returns a dict with wasserstein_distance, top_10_relative_change,
    effective_rank_change, and entropy_change.
    """
    eigenvalues_a = spectrum_a["eigenvalues"]
    eigenvalues_b = spectrum_b["eigenvalues"]

    # Pad to same length
    max_length = max(len(eigenvalues_a), len(eigenvalues_b))
    padded_a = np.pad(eigenvalues_a, (0, max_length - len(eigenvalues_a)), constant_values=0)
    padded_b = np.pad(eigenvalues_b, (0, max_length - len(eigenvalues_b)), constant_values=0)

    wasserstein_dist = float(wasserstein_distance(padded_a, padded_b))

    top_10_change = float(
        np.abs(padded_b[:10] - padded_a[:10]).mean()
        / (padded_a[:10].mean() + 1e-10)
    )

    return {
        "wasserstein_distance": wasserstein_dist,
        "top_10_relative_change": top_10_change,
        "effective_rank_change": spectrum_b["effective_rank"] - spectrum_a["effective_rank"],
        "entropy_change": spectrum_b["spectral_entropy"] - spectrum_a["spectral_entropy"],
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

_COVARIANCE_FIELDNAMES = [
    "layer",
    "forget_eff_rank_a", "forget_eff_rank_b",
    "retain_eff_rank_a", "retain_eff_rank_b",
    "forget_entropy_a", "forget_entropy_b",
    "retain_entropy_a", "retain_entropy_b",
    "forget_wasserstein", "retain_wasserstein",
    "forget_top10_change", "retain_top10_change",
]


def plot_covariance_analysis(
    results: List[Dict],
    outdir: str,
    title: Optional[str] = None,
    model_a: str = "Model A",
    model_b: str = "Model B",
) -> None:
    """Create the 3×3 panel of covariance spectrum plots."""
    layers_plot = [r["layer"] for r in results]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # (0,0) Effective rank comparison
    axis = axes[0, 0]
    axis.plot(layers_plot, [r["forget_eff_rank_a"] for r in results], "o-", label=f"Forget ({model_a.split('/')[-1]})", color="red", alpha=0.5)
    axis.plot(layers_plot, [r["forget_eff_rank_b"] for r in results], "s-", label=f"Forget ({model_b.split('/')[-1]})", color="darkred")
    axis.plot(layers_plot, [r["retain_eff_rank_a"] for r in results], "o-", label=f"Retain ({model_a.split('/')[-1]})", color="blue", alpha=0.5)
    axis.plot(layers_plot, [r["retain_eff_rank_b"] for r in results], "s-", label=f"Retain ({model_b.split('/')[-1]})", color="darkblue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Effective Rank")
    axis.set_title("Effective Rank (99% variance)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,1) Spectral entropy change
    axis = axes[0, 1]
    axis.plot(layers_plot, [r["forget_entropy_b"] - r["forget_entropy_a"] for r in results],
              "o-", label="Forget Δ", color="red")
    axis.plot(layers_plot, [r["retain_entropy_b"] - r["retain_entropy_a"] for r in results],
              "s-", label="Retain Δ", color="blue")
    axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Δ Spectral Entropy (B − A)")
    axis.set_title("Change in Spectral Entropy")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,2) Wasserstein distance
    axis = axes[0, 2]
    axis.plot(layers_plot, [r["forget_wasserstein"] for r in results], "o-", label="Forget", color="red")
    axis.plot(layers_plot, [r["retain_wasserstein"] for r in results], "s-", label="Retain", color="blue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Wasserstein Distance")
    axis.set_title("Spectrum Change (A → B)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,0) Top-10 eigenvalue changes
    axis = axes[1, 0]
    axis.plot(layers_plot, [r["forget_top10_change"] for r in results], "o-", label="Forget", color="red")
    axis.plot(layers_plot, [r["retain_top10_change"] for r in results], "s-", label="Retain", color="blue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Relative Change")
    axis.set_title("Top-10 Eigenvalue Change")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,1) through (2,1): Per-layer spectra loaded from .npz files are handled in main()
    # Fill remaining axes with placeholders
    for row, col in [(1, 1), (1, 2), (2, 0), (2, 1)]:
        axes[row, col].axis("off")

    # (2,2) Summary statistics
    axis = axes[2, 2]
    axis.axis("off")

    avg_forget_wasserstein = float(np.mean([r["forget_wasserstein"] for r in results]))
    avg_retain_wasserstein = float(np.mean([r["retain_wasserstein"] for r in results]))
    avg_forget_rank_change = float(np.mean([r["forget_eff_rank_b"] - r["forget_eff_rank_a"] for r in results]))
    avg_retain_rank_change = float(np.mean([r["retain_eff_rank_b"] - r["retain_eff_rank_a"] for r in results]))

    selectivity_ratio = avg_forget_wasserstein / (avg_retain_wasserstein + 1e-10)
    affected = "more" if avg_forget_wasserstein > avg_retain_wasserstein else "less"
    selective = "✓ Selective modification" if selectivity_ratio > 1.5 else "✗ Non-selective changes"

    summary_text = (
        f"Covariance Analysis Summary:\n\n"
        f"Spectrum Changes (Wasserstein):\n"
        f"- Forget: {avg_forget_wasserstein:.3f}\n"
        f"- Retain: {avg_retain_wasserstein:.3f}\n"
        f"- Ratio: {selectivity_ratio:.2f}x\n\n"
        f"Effective Rank Changes:\n"
        f"- Forget: {avg_forget_rank_change:+.1f}\n"
        f"- Retain: {avg_retain_rank_change:+.1f}\n\n"
        f"Interpretation:\n"
        f"Forget {affected} affected than Retain\n"
        f"{selective}"
    )

    axis.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

    plt.suptitle(title or "Activation Covariance Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "covariance_analysis.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation covariance structure changes between forget and retain datasets.",
    )
    parser.add_argument("--model-a", required=True, help="Baseline model")
    parser.add_argument("--model-b", required=True, help="Unlearned/fine-tuned model")
    parser.add_argument("--forget-text", default="data/forget.txt")
    parser.add_argument("--retain-text", default="data/retain.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max texts per split to process (default: 500)")
    parser.add_argument("--layers-to-analyze", type=str, default=None,
                        help="Comma-separated layer indices to analyze (default: every 4th)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="activation_covariance")

    init_wandb("activation_covariance", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[activation_covariance] Loading forget/retain texts...")
    with open(args.forget_text, "r") as fh:
        forget_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    with open(args.retain_text, "r") as fh:
        retain_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]

    print(
        f"[activation_covariance] Using {len(forget_texts)} forget texts, "
        f"{len(retain_texts)} retain texts (max-samples={args.max_samples})"
    )

    # Load tokenizers and models
    print(f"[activation_covariance] Loading base model: {args.model_a}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device_map_kwargs = {"device_map": "auto"} if device == "cuda" else {}

    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype, **device_map_kwargs)
    if not device_map_kwargs:
        model_a.to(device)
    model_a.eval()

    print(f"[activation_covariance] Loading unlearned model: {args.model_b}")
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=dtype, **device_map_kwargs)
    if not device_map_kwargs:
        model_b.to(device)
    model_b.eval()

    # Determine layer count
    if hasattr(model_a, "gpt_neox") and hasattr(model_a.gpt_neox, "layers"):
        num_layers = len(model_a.gpt_neox.layers)
    elif hasattr(model_a, "model") and hasattr(model_a.model, "layers"):
        num_layers = len(model_a.model.layers)
    elif hasattr(model_a, "transformer") and hasattr(model_a.transformer, "h"):
        num_layers = len(model_a.transformer.h)
    elif hasattr(model_a, "encoder") and hasattr(model_a.encoder, "layer"):
        num_layers = len(model_a.encoder.layer)
    else:
        print("Warning: Could not determine number of layers, defaulting to 32")
        num_layers = 32

    if args.layers_to_analyze:
        layers_to_analyze = [int(x) for x in args.layers_to_analyze.split(",")]
    else:
        layers_to_analyze = list(range(0, num_layers + 1, 4))

    results: List[Dict] = []
    os.makedirs(args.outdir, exist_ok=True)

    # Analyze each selected layer
    print(f"[activation_covariance] Analyzing covariance spectra across {len(layers_to_analyze)} layers...")
    for layer_index in tqdm(layers_to_analyze, desc="Analyzing covariance spectra", unit="layer"):
        print(f"\n[activation_covariance] Layer {layer_index}:")

        # Extract activations for both models + both splits
        print("  Extracting forget activations (both models)...")
        forget_activations_a = get_activations_batch(
            model_a, tokenizer, forget_texts, layer_index, device, args.max_length, args.batch_size,
        )
        forget_activations_b = get_activations_batch(
            model_b, tokenizer, forget_texts, layer_index, device, args.max_length, args.batch_size,
        )

        print("  Extracting retain activations (both models)...")
        retain_activations_a = get_activations_batch(
            model_a, tokenizer, retain_texts, layer_index, device, args.max_length, args.batch_size,
        )
        retain_activations_b = get_activations_batch(
            model_b, tokenizer, retain_texts, layer_index, device, args.max_length, args.batch_size,
        )

        # Compute covariance spectra
        print("  Computing & comparing covariance spectra...")
        forget_spectrum_a = compute_covariance_metrics(forget_activations_a)
        forget_spectrum_b = compute_covariance_metrics(forget_activations_b)
        retain_spectrum_a = compute_covariance_metrics(retain_activations_a)
        retain_spectrum_b = compute_covariance_metrics(retain_activations_b)

        forget_comparison = compare_spectra(forget_spectrum_a, forget_spectrum_b)
        retain_comparison = compare_spectra(retain_spectrum_a, retain_spectrum_b)

        result = {
            "layer": layer_index,
            "forget_eff_rank_a": forget_spectrum_a["effective_rank"],
            "forget_eff_rank_b": forget_spectrum_b["effective_rank"],
            "retain_eff_rank_a": retain_spectrum_a["effective_rank"],
            "retain_eff_rank_b": retain_spectrum_b["effective_rank"],
            "forget_entropy_a": forget_spectrum_a["spectral_entropy"],
            "forget_entropy_b": forget_spectrum_b["spectral_entropy"],
            "retain_entropy_a": retain_spectrum_a["spectral_entropy"],
            "retain_entropy_b": retain_spectrum_b["spectral_entropy"],
            "forget_wasserstein": forget_comparison["wasserstein_distance"],
            "retain_wasserstein": retain_comparison["wasserstein_distance"],
            "forget_top10_change": forget_comparison["top_10_relative_change"],
            "retain_top10_change": retain_comparison["top_10_relative_change"],
        }
        results.append(result)

        # Save detailed spectra for this layer
        np.savez_compressed(
            os.path.join(args.outdir, f"spectra_layer_{layer_index}.npz"),
            forget_eig_a=forget_spectrum_a["eigenvalues"],
            forget_eig_b=forget_spectrum_b["eigenvalues"],
            retain_eig_a=retain_spectrum_a["eigenvalues"],
            retain_eig_b=retain_spectrum_b["eigenvalues"],
            forget_var_a=forget_spectrum_a["explained_var_ratio"],
            forget_var_b=forget_spectrum_b["explained_var_ratio"],
            retain_var_a=retain_spectrum_a["explained_var_ratio"],
            retain_var_b=retain_spectrum_b["explained_var_ratio"],
        )

    # Save results CSV
    write_csv(os.path.join(args.outdir, "covariance_metrics.csv"), results, _COVARIANCE_FIELDNAMES)

    # Plots
    plot_covariance_analysis(results, args.outdir, title=args.title,
                             model_a=args.model_a, model_b=args.model_b)

    # Summary JSON
    avg_forget_wasserstein = float(np.mean([r["forget_wasserstein"] for r in results]))
    avg_retain_wasserstein = float(np.mean([r["retain_wasserstein"] for r in results]))

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_forget_wasserstein": avg_forget_wasserstein,
        "avg_retain_wasserstein": avg_retain_wasserstein,
        "forget_more_affected": bool(avg_forget_wasserstein > avg_retain_wasserstein),
        "selective_ratio": float(avg_forget_wasserstein / (avg_retain_wasserstein + 1e-10)),
    }

    with open(os.path.join(args.outdir, "covariance_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[activation_covariance] ✓ Results saved to {args.outdir}")
    print(f"[activation_covariance] Forget spectrum change (Wasserstein): {avg_forget_wasserstein:.3f}")
    print(f"[activation_covariance] Retain spectrum change (Wasserstein): {avg_retain_wasserstein:.3f}")
    print(f"[activation_covariance] Selectivity ratio: {summary['selective_ratio']:.2f}x")
    log_csv_as_table(os.path.join(args.outdir, "covariance_metrics.csv"), "covariance_metrics")
    # Log native W&B line-series charts from the computed results
    layers_list = [r["layer"] for r in results]
    ma = args.model_a.split("/")[-1]
    mb = args.model_b.split("/")[-1]
    log_line_series(
        "covariance/effective_rank",
        xs=layers_list,
        ys=[
            [r["forget_eff_rank_a"] for r in results],
            [r["forget_eff_rank_b"] for r in results],
            [r["retain_eff_rank_a"] for r in results],
            [r["retain_eff_rank_b"] for r in results],
        ],
        series_keys=[f"Forget ({ma})", f"Forget ({mb})", f"Retain ({ma})", f"Retain ({mb})"],
        title="Effective Rank (99% variance) by Layer",
        xname="Layer",
    )
    log_line_series(
        "covariance/spectral_entropy_delta",
        xs=layers_list,
        ys=[
            [r["forget_entropy_b"] - r["forget_entropy_a"] for r in results],
            [r["retain_entropy_b"] - r["retain_entropy_a"] for r in results],
        ],
        series_keys=["Forget Δ", "Retain Δ"],
        title="Change in Spectral Entropy (B − A) by Layer",
        xname="Layer",
    )
    log_line_series(
        "covariance/wasserstein_distance",
        xs=layers_list,
        ys=[
            [r["forget_wasserstein"] for r in results],
            [r["retain_wasserstein"] for r in results],
        ],
        series_keys=["Forget", "Retain"],
        title="Spectrum Change (Wasserstein Distance, A → B) by Layer",
        xname="Layer",
    )
    log_line_series(
        "covariance/top10_eigenvalue_change",
        xs=layers_list,
        ys=[
            [r["forget_top10_change"] for r in results],
            [r["retain_top10_change"] for r in results],
        ],
        series_keys=["Forget", "Retain"],
        title="Top-10 Eigenvalue Relative Change by Layer",
        xname="Layer",
    )
    finish_wandb()


if __name__ == "__main__":
    main()