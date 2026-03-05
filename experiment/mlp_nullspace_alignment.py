#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
# ]
# ///

"""
MLP null-space alignment analysis.

For every MLP weight matrix in the model, decomposes the fine-tuning
update ΔW into its projection onto the column space versus the
approximate null space of the original weight W.  High null-space
projection ratio means the update acts in directions the original
weight did not use — i.e. it is *off-manifold*.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional

from utils import (
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    extract_layer,
    classify_granular,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
    SmartLoader,
)


# ---------------------------------------------------------------------------
# Core analysis (testable, no I/O)
# ---------------------------------------------------------------------------

def compute_nullspace_alignment(
    W_orig: torch.Tensor,
    weight_update: torch.Tensor,
    rank_threshold: float = 0.99,
) -> Optional[Dict]:
    """Analyze how *weight_update* aligns with the null space of *W_orig*.

    Returns a dict of alignment metrics, or None if the inputs are
    incompatible (non-2D, zero variance, etc.).
    """
    if W_orig.ndim != 2 or weight_update.ndim != 2:
        return None

    W = W_orig.float()
    update_float = weight_update.float()

    # SVD of original weights (full U for null-space projection)
    U, singular_values, _Vt = torch.linalg.svd(W, full_matrices=True)

    singular_values_sq = singular_values * singular_values
    total_variance = singular_values_sq.sum().item()
    if total_variance == 0:
        return None

    cumulative_sum = torch.cumsum(singular_values_sq, dim=0)
    effective_rank = int(torch.searchsorted(cumulative_sum, rank_threshold * total_variance).item()) + 1
    effective_rank = min(effective_rank, len(singular_values))

    n_rows, _n_cols = W.shape
    if effective_rank >= min(n_rows, len(singular_values)):
        # Full rank — no null space
        nullspace_dim = 0
        colspace_proj_norm = float(update_float.norm().item())
        nullspace_proj_norm = 0.0
    else:
        U_colspace = U[:, :effective_rank]
        U_nullspace = U[:, effective_rank:]

        update_colspace = U_colspace @ (U_colspace.T @ update_float)
        colspace_proj_norm = float(update_colspace.norm().item())

        update_nullspace = U_nullspace @ (U_nullspace.T @ update_float)
        nullspace_proj_norm = float(update_nullspace.norm().item())

        nullspace_dim = U_nullspace.shape[1]

    total_norm = float(update_float.norm().item())

    if total_norm > 0:
        colspace_ratio = colspace_proj_norm / total_norm
        nullspace_ratio = nullspace_proj_norm / total_norm
    else:
        colspace_ratio = 0.0
        nullspace_ratio = 0.0

    # Row-space effective rank
    singular_values_row = torch.linalg.svdvals(W.T)
    sv_row_sq = singular_values_row * singular_values_row
    total_var_row = sv_row_sq.sum().item()
    if total_var_row > 0:
        cumsum_row = torch.cumsum(sv_row_sq, dim=0)
        _effective_rank_row = int(torch.searchsorted(cumsum_row, rank_threshold * total_var_row).item()) + 1
    # (not currently exposed in the return dict, but computed for completeness)

    # Rank change after adding the update
    W_new = W + update_float
    singular_values_new = torch.linalg.svdvals(W_new)
    sv_new_sq = singular_values_new * singular_values_new
    total_var_new = sv_new_sq.sum().item()
    if total_var_new > 0:
        cumsum_new = torch.cumsum(sv_new_sq, dim=0)
        effective_rank_new = int(torch.searchsorted(cumsum_new, rank_threshold * total_var_new).item()) + 1
    else:
        effective_rank_new = 0

    return {
        "original_eff_rank": int(effective_rank),
        "updated_eff_rank": int(effective_rank_new),
        "rank_increase": int(effective_rank_new - effective_rank),
        "colspace_projection_ratio": float(colspace_ratio),
        "nullspace_projection_ratio": float(nullspace_ratio),
        "nullspace_dimension": int(nullspace_dim),
        "update_norm": float(total_norm),
        "original_norm": float(W.norm().item()),
        "relative_update_size": float(total_norm / (W.norm().item() + 1e-10)),
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_nullspace_alignment(
    layer_results: List[Dict],
    per_matrix_results: List[Dict],
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """Create a 2×3 panel of null-space alignment plots."""
    layers = [r["layer"] for r in layer_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (0,0) Null-space vs column-space projection
    axis = axes[0, 0]
    axis.plot(layers, [r["avg_nullspace_ratio"] for r in layer_results], "o-",
              label="Nullspace", color="red")
    axis.plot(layers, [r["avg_colspace_ratio"] for r in layer_results], "s-",
              label="Column Space", color="blue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Projection Ratio")
    axis.set_title("Update Alignment (Higher Nullspace = Off-manifold)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,1) Encoder vs decoder null-space alignment
    axis = axes[0, 1]
    axis.plot(layers, [r["encoder_nullspace_ratio"] for r in layer_results], "o-",
              label="Encoder (Input)", color="green")
    axis.plot(layers, [r["decoder_nullspace_ratio"] for r in layer_results], "s-",
              label="Decoder (Output)", color="purple")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Nullspace Projection Ratio")
    axis.set_title("Encoder vs Decoder Nullspace Alignment")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,2) Rank increase
    axis = axes[0, 2]
    axis.plot(layers, [r["avg_rank_increase"] for r in layer_results], "o-", color="orange")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Average Rank Increase")
    axis.set_title("Effective Rank Change (W → W + ΔW)")
    axis.grid(alpha=0.3)

    # (1,0) Scatter: null-space ratio vs rank increase
    axis = axes[1, 0]
    nullspace_all = [r["nullspace_projection_ratio"] for r in per_matrix_results]
    rank_inc_all = [r["rank_increase"] for r in per_matrix_results]
    axis.scatter(nullspace_all, rank_inc_all, alpha=0.5, s=20)
    axis.set_xlabel("Nullspace Projection Ratio")
    axis.set_ylabel("Rank Increase")
    axis.set_title("Nullspace Alignment vs Rank Increase")
    axis.grid(alpha=0.3)

    # (1,1) Distribution of projection ratios
    axis = axes[1, 1]
    colspace_all = [r["colspace_projection_ratio"] for r in per_matrix_results]
    axis.hist(nullspace_all, bins=30, alpha=0.5, label="Nullspace", color="red")
    axis.hist(colspace_all, bins=30, alpha=0.5, label="Column Space", color="blue")
    axis.set_xlabel("Projection Ratio")
    axis.set_ylabel("Count")
    axis.set_title("Distribution of Projection Ratios")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,2) Summary text
    axis = axes[1, 2]
    axis.axis("off")

    avg_nullspace = float(np.mean(nullspace_all))
    avg_colspace = float(np.mean(colspace_all))
    avg_rank_inc = float(np.mean(rank_inc_all))

    top_layers = sorted(layer_results, key=lambda r: r["avg_nullspace_ratio"], reverse=True)[:3]
    summary_text = (
        f"MLP Nullspace Alignment Summary:\n\n"
        f"Overall Statistics:\n"
        f"- Avg Nullspace Projection: {avg_nullspace:.3f}\n"
        f"- Avg Colspace Projection: {avg_colspace:.3f}\n"
        f"- Nullspace/Colspace Ratio: {avg_nullspace / (avg_colspace + 1e-10):.2f}x\n"
        f"- Avg Rank Increase: {avg_rank_inc:.1f}\n\n"
        f"Top Nullspace-Aligned Layers:\n"
        + "\n".join(f"  Layer {l['layer']}: {l['avg_nullspace_ratio']:.3f}" for l in top_layers)
        + f"\n\nInterpretation:\n"
        f"{'✓ Updates primarily off-manifold' if avg_nullspace > 0.6 else '✗ Updates mostly on-manifold'}\n"
        f"{'✓ Rank expansion observed' if avg_rank_inc > 5 else '✗ Minimal rank change'}"
    )
    axis.text(0.05, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

    plt.suptitle(title or "MLP Nullspace Alignment Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mlp_nullspace_alignment.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze null-space alignment of MLP weight updates.",
    )
    parser.add_argument("--model-a", required=True, help="Baseline model")
    parser.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--rank-threshold", type=float, default=0.99)
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="mlp_nullspace_alignment")

    init_wandb("mlp_nullspace_alignment", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("[mlp_nullspace_alignment] Loading model weights...")
    print(f"  Model A (baseline): {args.model_a}")
    print(f"  Model B (target)  : {args.model_b}")

    loader_a = SmartLoader(args.model_a)
    loader_b = SmartLoader(args.model_b)

    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()

    # Filter for MLP weights only
    mlp_names = [
        name for name in sorted(names_a.intersection(names_b))
        if name.endswith(".weight") and classify_granular(name) in ("mlp_expand", "mlp_contract")
    ]

    print(f"[mlp_nullspace_alignment] Found {len(mlp_names)} MLP weight matrices")

    per_matrix_results: List[Dict] = []
    layer_aggregated: Dict[int, Dict[str, list]] = {}

    print("[mlp_nullspace_alignment] Computing SVD + nullspace projections...")
    for name in tqdm(mlp_names, desc="Analyzing MLP nullspace alignment", unit="matrix"):
        weight_a = loader_a.get_param(name, device, dtype)
        if weight_a is None or weight_a.ndim != 2:
            continue

        weight_b = loader_b.get_param(name, device, dtype)
        if weight_b is None or weight_b.shape != weight_a.shape:
            continue

        weight_update = weight_b - weight_a
        metrics = compute_nullspace_alignment(weight_a, weight_update, args.rank_threshold)
        if metrics is None:
            continue

        layer = extract_layer(name)
        metrics["name"] = name
        metrics["layer"] = layer if layer is not None else -1
        metrics["shape"] = f"{weight_a.shape[0]}x{weight_a.shape[1]}"
        metrics["type"] = (
            "decoder"
            if any(k in name for k in ("up_proj", "out_proj", "o_proj", "fc2"))
            else "encoder"
        )
        per_matrix_results.append(metrics)

        if layer is not None:
            if layer not in layer_aggregated:
                layer_aggregated[layer] = {
                    "colspace_ratios": [],
                    "nullspace_ratios": [],
                    "rank_increases": [],
                    "encoder_nullspace": [],
                    "decoder_nullspace": [],
                }
            agg = layer_aggregated[layer]
            agg["colspace_ratios"].append(metrics["colspace_projection_ratio"])
            agg["nullspace_ratios"].append(metrics["nullspace_projection_ratio"])
            agg["rank_increases"].append(metrics["rank_increase"])
            if metrics["type"] == "encoder":
                agg["encoder_nullspace"].append(metrics["nullspace_projection_ratio"])
            else:
                agg["decoder_nullspace"].append(metrics["nullspace_projection_ratio"])

        del weight_a, weight_b, weight_update

    # Aggregate layer-wise statistics
    layer_results: List[Dict] = []
    for layer in sorted(layer_aggregated.keys()):
        stats = layer_aggregated[layer]
        layer_results.append({
            "layer": layer,
            "avg_colspace_ratio": float(np.mean(stats["colspace_ratios"])),
            "avg_nullspace_ratio": float(np.mean(stats["nullspace_ratios"])),
            "avg_rank_increase": float(np.mean(stats["rank_increases"])),
            "encoder_nullspace_ratio": float(np.mean(stats["encoder_nullspace"])) if stats["encoder_nullspace"] else 0.0,
            "decoder_nullspace_ratio": float(np.mean(stats["decoder_nullspace"])) if stats["decoder_nullspace"] else 0.0,
            "num_matrices": len(stats["colspace_ratios"]),
        })

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    write_csv(
        os.path.join(args.outdir, "mlp_nullspace_metrics.csv"),
        per_matrix_results,
        ["name", "layer", "type", "shape", "original_eff_rank", "updated_eff_rank",
         "rank_increase", "colspace_projection_ratio", "nullspace_projection_ratio",
         "nullspace_dimension", "update_norm", "original_norm", "relative_update_size"],
    )
    write_csv(
        os.path.join(args.outdir, "layer_nullspace_summary.csv"),
        layer_results,
        ["layer", "avg_colspace_ratio", "avg_nullspace_ratio", "avg_rank_increase",
         "encoder_nullspace_ratio", "decoder_nullspace_ratio", "num_matrices"],
    )

    # Plots
    if layer_results:
        plot_nullspace_alignment(
            layer_results, per_matrix_results, args.outdir, title=args.title,
        )

    # Summary JSON
    nullspace_ratios = [r["nullspace_projection_ratio"] for r in per_matrix_results]
    colspace_ratios = [r["colspace_projection_ratio"] for r in per_matrix_results]

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_nullspace_ratio": float(np.mean(nullspace_ratios)) if per_matrix_results else 0,
        "avg_colspace_ratio": float(np.mean(colspace_ratios)) if per_matrix_results else 0,
        "avg_rank_increase": float(np.mean([r["rank_increase"] for r in per_matrix_results])) if per_matrix_results else 0,
        "primarily_off_manifold": bool(np.mean(nullspace_ratios) > 0.6) if per_matrix_results else False,
        "total_matrices_analyzed": len(per_matrix_results),
    }

    with open(os.path.join(args.outdir, "nullspace_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[mlp_nullspace_alignment] ✓ Results saved to {args.outdir}")
    print(f"[mlp_nullspace_alignment] Average nullspace projection: {summary['avg_nullspace_ratio']:.3f}")
    print(f"[mlp_nullspace_alignment] Average colspace projection: {summary['avg_colspace_ratio']:.3f}")
    print(f"[mlp_nullspace_alignment] Updates are {'primarily off-manifold' if summary['primarily_off_manifold'] else 'mostly on-manifold'}")
    log_csv_as_table(os.path.join(args.outdir, "mlp_nullspace_metrics.csv"), "mlp_nullspace_metrics")
    log_csv_as_table(os.path.join(args.outdir, "layer_nullspace_summary.csv"), "layer_nullspace_summary")
    log_plots(args.outdir, "mlp_nullspace")
    finish_wandb()


if __name__ == "__main__":
    main()