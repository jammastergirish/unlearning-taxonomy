#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scipy",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Null-space and subspace-alignment analysis of weight changes.

For a sampled set of weight matrices, this script:
  1. Computes the SVD of the weight *change* (ΔW = W_b − W_a) to measure how
     low-rank the update is (effective rank, top-10 singular-value variance
     concentration, singular-value decay rate).
  2. Computes the Grassmann distance between the top-k left-singular-vector
     subspaces of the original and fine-tuned weight matrices to quantify how
     much the "principal directions" of each matrix shifted.
  3. Aggregates results by component type (qkv, proj, mlp_expand, mlp_contract)
     and produces box plots and a summary CSV.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    classify_granular,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
    SmartLoader,
)


# ---------------------------------------------------------------------------
# Core analysis functions (testable, no I/O)
# ---------------------------------------------------------------------------

def compute_null_space_projection(
    weight_change: torch.Tensor,
    rank_threshold: float = 0.99,
) -> Dict[str, object]:
    """Analyze the rank structure of a weight-change matrix via its singular values.

    Args:
        weight_change: 2-D tensor (ΔW = W_after − W_before).
        rank_threshold: fraction of total variance used to define effective rank.

    Returns a dict with:
        - ``effective_rank``: number of singular values capturing *rank_threshold*
          of the total squared-singular-value mass.
        - ``top10_variance_ratio``: fraction of variance explained by the first
          10 singular values (higher → more low-rank).
        - ``max_singular_value``: σ₁(ΔW).
        - ``singular_value_decay``: σ₁₀ / σ₁ (how fast the spectrum drops).
    """
    if weight_change.numel() == 0 or weight_change.ndim != 2:
        return {
            "effective_rank": 0,
            "top10_variance_ratio": 0.0,
            "max_singular_value": 0.0,
            "singular_value_decay": 1.0,
        }

    singular_values = torch.linalg.svdvals(weight_change.float())

    squared_singular_values = singular_values * singular_values
    total_variance = squared_singular_values.sum().item()

    if total_variance == 0:
        return {
            "effective_rank": 0,
            "top10_variance_ratio": 0.0,
            "max_singular_value": 0.0,
            "singular_value_decay": 1.0,
        }

    cumulative_variance = torch.cumsum(squared_singular_values, dim=0)
    effective_rank = int(
        torch.searchsorted(cumulative_variance, rank_threshold * total_variance).item()
    ) + 1

    # Fraction of variance captured by the first 10 singular values
    top10_index = min(9, len(singular_values) - 1)
    top10_variance_ratio = float(cumulative_variance[top10_index].item() / total_variance)

    # Decay rate: how much smaller is σ₁₀ compared to σ₁
    if len(singular_values) > 10:
        singular_value_decay = float(
            (singular_values[10] / (singular_values[0] + 1e-10)).item()
        )
    else:
        singular_value_decay = 1.0

    return {
        "effective_rank": effective_rank,
        "top10_variance_ratio": top10_variance_ratio,
        "max_singular_value": float(singular_values[0].item()),
        "singular_value_decay": singular_value_decay,
    }


def analyze_subspace_alignment(
    weight_before: torch.Tensor,
    weight_after: torch.Tensor,
    num_top_vectors: int = 20,
) -> Dict[str, float]:
    """Measure how much the principal subspaces of two weight matrices overlap.

    Computes the Grassmann distance between the top-k left-singular-vector
    subspaces of *weight_before* and *weight_after*.

    Args:
        weight_before: 2-D tensor (original weights).
        weight_after: 2-D tensor (fine-tuned weights).
        num_top_vectors: number of leading singular vectors to compare.

    Returns a dict with:
        - ``subspace_alignment``: mean singular value of Uₐᵀ Uᵦ (1.0 = identical).
        - ``grassmann_distance``: distance between the two subspaces (0.0 = identical).
        - ``singular_value_ratio``: σ₁(W_after) / σ₁(W_before).
    """
    if weight_before.numel() == 0 or weight_before.ndim != 2:
        return {}

    left_vectors_before, svals_before, _ = torch.linalg.svd(
        weight_before.float(), full_matrices=False,
    )
    left_vectors_after, svals_after, _ = torch.linalg.svd(
        weight_after.float(), full_matrices=False,
    )

    k = min(num_top_vectors, left_vectors_before.shape[1], left_vectors_after.shape[1])

    top_before = left_vectors_before[:, :k]
    top_after = left_vectors_after[:, :k]

    # Alignment matrix: singular values of Uₐᵀ Uᵦ measure per-direction overlap
    alignment_matrix = top_before.T @ top_after
    alignment_singular_values = torch.linalg.svdvals(alignment_matrix)

    mean_alignment = float(alignment_singular_values.mean().item())

    # Grassmann distance: sqrt(k − Σ σᵢ²)
    grassmann_distance = float(
        torch.sqrt(
            torch.clamp(k - (alignment_singular_values ** 2).sum(), min=0)
        ).item()
    )

    # Ratio of leading singular values — did the overall "scale" of the matrix change?
    if len(svals_before) > 0 and len(svals_after) > 0:
        singular_value_ratio = float(
            (svals_after[0] / (svals_before[0] + 1e-10)).item()
        )
    else:
        singular_value_ratio = 1.0

    return {
        "subspace_alignment": mean_alignment,
        "grassmann_distance": grassmann_distance,
        "singular_value_ratio": singular_value_ratio,
    }


# ---------------------------------------------------------------------------
# Aggregation helper (testable, no I/O)
# ---------------------------------------------------------------------------

COMPONENT_LABELS = ("qkv", "proj", "mlp_expand", "mlp_contract")


def aggregate_by_component(
    results: List[Dict],
) -> Dict[str, Dict[str, List[float]]]:
    """Group null-space and alignment metrics by component type.

    Returns a dict mapping each component label to sub-dicts with lists of
    ``"null_space"`` (top-10 variance ratios) and ``"alignment"`` values.
    """
    aggregated = {
        label: {"null_space": [], "alignment": []}
        for label in COMPONENT_LABELS
    }

    for row in results:
        component = row.get("component", "")
        if component in aggregated:
            aggregated[component]["null_space"].append(
                row.get("top10_variance_ratio", 0.0)
            )
            aggregated[component]["alignment"].append(
                row.get("subspace_alignment", 0.0)
            )

    return aggregated


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_null_space_results(
    component_results: Dict[str, Dict[str, List[float]]],
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """Create box-plot visualisations for null-space concentration and subspace alignment."""
    has_data = any(
        component_results[label]["null_space"] for label in COMPONENT_LABELS
    )
    if not has_data:
        return

    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: top-10 SV variance concentration
    null_space_data = []
    null_space_labels = []
    for label in COMPONENT_LABELS:
        if component_results[label]["null_space"]:
            null_space_data.append(component_results[label]["null_space"])
            null_space_labels.append(label)

    left_axis.boxplot(null_space_data, tick_labels=null_space_labels)
    left_axis.set_ylabel("Top-10 SV Variance Ratio")
    left_axis.set_title("Weight Change Concentration (Higher = More Low-Rank)")
    left_axis.grid(alpha=0.3)

    # Right panel: subspace alignment
    alignment_data = []
    alignment_labels = []
    for label in COMPONENT_LABELS:
        if component_results[label]["alignment"]:
            alignment_data.append(component_results[label]["alignment"])
            alignment_labels.append(label)

    right_axis.boxplot(alignment_data, tick_labels=alignment_labels)
    right_axis.set_ylabel("Subspace Alignment")
    right_axis.set_title("Original vs Fine-tuned Subspace Alignment")
    right_axis.grid(alpha=0.3)

    plt.suptitle(title or "Null Space and Subspace Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "null_space_visualization.png"))
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Null-space and subspace-alignment analysis of weight changes."
    )
    parser.add_argument("--model-a", required=True, help="Baseline model path")
    parser.add_argument("--model-b", required=True, help="Fine-tuned model path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of weight matrices to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="null_space_analysis")

    init_wandb("null_space_analysis", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"[null_space_analysis] Loading model weights...")
    print(f"  Model A (baseline): {args.model_a}")
    print(f"  Model B (target)  : {args.model_b}")
    loader_before = SmartLoader(args.model_a)
    loader_after = SmartLoader(args.model_b)

    # Find shared weight names
    names_before = loader_before.get_all_param_names()
    names_after = loader_after.get_all_param_names()
    weight_names = sorted(
        name for name in names_before.intersection(names_after) if name.endswith(".weight")
    )

    # Sample a subset for analysis
    if len(weight_names) > args.num_samples:
        weight_names = np.random.choice(weight_names, args.num_samples, replace=False).tolist()

    results: List[Dict] = []

    print(f"[null_space_analysis] Running SVD on {len(weight_names)} weight matrices...")
    for name in tqdm(weight_names, desc="SVD on weight matrices", unit="matrix"):
        weight_before_tensor = loader_before.get_param(name, device, dtype)
        if weight_before_tensor is None or weight_before_tensor.ndim != 2:
            continue

        weight_after_tensor = loader_after.get_param(name, device, dtype)
        if weight_after_tensor is None or weight_after_tensor.shape != weight_before_tensor.shape:
            continue

        weight_change = weight_after_tensor - weight_before_tensor

        null_space_metrics = compute_null_space_projection(weight_change)
        alignment_metrics = analyze_subspace_alignment(weight_before_tensor, weight_after_tensor)

        component_type = classify_granular(name)

        result_row = {
            "name": name,
            "component": component_type,
            **null_space_metrics,
            **alignment_metrics,
        }
        results.append(result_row)

        del weight_before_tensor, weight_after_tensor, weight_change

    # Aggregate by component type
    component_results = aggregate_by_component(results)

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    if results:
        fieldnames = list(results[0].keys())
        write_csv(
            os.path.join(args.outdir, "null_space_results.csv"),
            results,
            fieldnames,
        )

    # Create visualisations
    plot_null_space_results(component_results, args.outdir, title=args.title)

    # Print summary
    print("\n[null_space_analysis] === Null Space Analysis Summary ===")
    for label in COMPONENT_LABELS:
        if component_results[label]["null_space"]:
            mean_variance = np.mean(component_results[label]["null_space"])
            mean_alignment = np.mean(component_results[label]["alignment"])
            print(f"  {label} — Avg variance in top-10 SVs: {mean_variance:.3f}")
            print(f"  {label} — Avg subspace alignment: {mean_alignment:.3f}")

    print(f"\n[null_space_analysis] ✓ Results saved to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "null_space_results.csv"), "null_space_results")
    log_plots(args.outdir, "null_space")
    finish_wandb()


if __name__ == "__main__":
    main()