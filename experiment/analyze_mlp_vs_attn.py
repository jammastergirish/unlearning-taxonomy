#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "pandas",
#   "matplotlib",
#   "numpy",
#   "wandb",
# ]
# ///

"""
Compare weight-change magnitudes between MLP and Attention layers.

Reads the per-layer and per-matrix CSVs produced by ``collect_weight_comparison.py`` and
generates:
  1. A side-by-side plot of Frobenius-norm changes (MLP vs Attention) plus
     the per-layer ratio.
  2. A detailed 2×2 panel with stable-rank and (optionally) empirical-rank
     comparisons, rank-efficiency, and a total-change bar chart.
  3. A summary CSV with per-layer MLP/Attention Frobenius norms, their ratio,
     and stable-rank values.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import init_wandb, log_csv_as_table, log_plots, finish_wandb


# ---------------------------------------------------------------------------
# Core logic (testable, no I/O or plotting)
# ---------------------------------------------------------------------------

def build_mlp_attn_summary(per_layer_df: pd.DataFrame) -> List[Dict]:
    """Build a per-layer summary comparing MLP and Attention weight changes.

    For each layer that has *both* an ``mlp`` and ``attn`` group row, produces
    a dict with:
      - ``layer``: the layer index
      - ``mlp_frobenius``: Frobenius norm of MLP weight change
      - ``attn_frobenius``: Frobenius norm of Attention weight change
      - ``ratio_mlp_attn``: MLP / Attention Frobenius ratio
      - ``mlp_stable_rank``: mean stable rank of MLP weight change (or None)
      - ``attn_stable_rank``: mean stable rank of Attention weight change (or None)

    Returns an empty list if there are no layers with both groups.
    """
    summary_rows: List[Dict] = []

    for layer in sorted(per_layer_df["layer"].unique()):
        mlp_row = per_layer_df[
            (per_layer_df["layer"] == layer) & (per_layer_df["group"] == "mlp")
        ]
        attn_row = per_layer_df[
            (per_layer_df["layer"] == layer) & (per_layer_df["group"] == "attn")
        ]

        if mlp_row.empty or attn_row.empty:
            continue

        mlp_frobenius = mlp_row["dW_fro_layer"].values[0]
        attn_frobenius = attn_row["dW_fro_layer"].values[0]

        row: Dict = {
            "layer": layer,
            "mlp_frobenius": mlp_frobenius,
            "attn_frobenius": attn_frobenius,
            "ratio_mlp_attn": mlp_frobenius / (attn_frobenius + 1e-10),
        }

        if "mean_dW_stable_rank" in per_layer_df.columns:
            row["mlp_stable_rank"] = mlp_row["mean_dW_stable_rank"].values[0]
            row["attn_stable_rank"] = attn_row["mean_dW_stable_rank"].values[0]
        else:
            row["mlp_stable_rank"] = None
            row["attn_stable_rank"] = None

        summary_rows.append(row)

    return summary_rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_magnitude_comparison(
    per_layer_df: pd.DataFrame,
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """Plot MLP vs Attention Frobenius-norm changes and their per-layer ratio."""
    mlp_data = per_layer_df[per_layer_df["group"] == "mlp"]
    attn_data = per_layer_df[per_layer_df["group"] == "attn"]

    if mlp_data.empty or attn_data.empty:
        return

    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: absolute Frobenius-norm comparison
    left_axis.plot(
        mlp_data["layer"], mlp_data["dW_fro_layer"],
        "o-", label="MLP", color="blue",
    )
    left_axis.plot(
        attn_data["layer"], attn_data["dW_fro_layer"],
        "s-", label="Attention", color="orange",
    )
    left_axis.set_xlabel("Layer")
    left_axis.set_ylabel(r"$\|\Delta W\|_F$")
    left_axis.set_title("Weight Change Magnitude: MLP vs Attention")
    left_axis.legend()
    left_axis.grid(alpha=0.3)

    # Right panel: MLP / Attention ratio
    merged = pd.merge(
        mlp_data[["layer", "dW_fro_layer"]],
        attn_data[["layer", "dW_fro_layer"]],
        on="layer",
        suffixes=("_mlp", "_attn"),
    )
    merged["ratio"] = merged["dW_fro_layer_mlp"] / (merged["dW_fro_layer_attn"] + 1e-10)

    right_axis.plot(merged["layer"], merged["ratio"], "o-", color="green")
    right_axis.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    right_axis.set_xlabel("Layer")
    right_axis.set_ylabel(r"$\|\Delta W_{\mathrm{MLP}}\|_F / \|\Delta W_{\mathrm{Attn}}\|_F$")
    right_axis.set_title("MLP / Attention Change Ratio")
    right_axis.grid(alpha=0.3)

    plt.suptitle(title or "MLP vs Attention Weight Changes")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mlp_vs_attn_magnitude.png"))
    plt.close()


def plot_detailed_analysis(
    per_layer_df: pd.DataFrame,
    outdir: str,
    title: Optional[str] = None,
) -> None:
    """Plot a 2×2 panel with rank structure and total-change bar chart."""
    if "mean_dW_stable_rank" not in per_layer_df.columns:
        return

    mlp_data = per_layer_df[per_layer_df["group"] == "mlp"]
    attn_data = per_layer_df[per_layer_df["group"] == "attn"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: stable-rank comparison
    if not mlp_data.empty and not attn_data.empty:
        axis = axes[0, 0]
        axis.plot(
            mlp_data["layer"], mlp_data["mean_dW_stable_rank"],
            "o-", label="MLP", color="blue",
        )
        axis.plot(
            attn_data["layer"], attn_data["mean_dW_stable_rank"],
            "s-", label="Attention", color="orange",
        )
        axis.set_xlabel("Layer")
        axis.set_ylabel("Mean Stable Rank")
        axis.set_title("Stable Rank of Weight Changes")
        axis.legend()
        axis.grid(alpha=0.3)

        # Top-right: empirical-rank comparison (if available)
        if "mean_dW_empirical_rank" in per_layer_df.columns:
            axis = axes[0, 1]
            axis.plot(
                mlp_data["layer"], mlp_data["mean_dW_empirical_rank"],
                "o-", label="MLP", color="blue",
            )
            axis.plot(
                attn_data["layer"], attn_data["mean_dW_empirical_rank"],
                "s-", label="Attention", color="orange",
            )
            axis.set_xlabel("Layer")
            axis.set_ylabel("Mean Empirical Rank")
            axis.set_title("Empirical Rank of Weight Changes")
            axis.legend()
            axis.grid(alpha=0.3)

            # Bottom-left: rank efficiency (empirical / stable ratio)
            axis = axes[1, 0]
            mlp_efficiency = mlp_data["mean_dW_empirical_rank"] / (
                mlp_data["mean_dW_stable_rank"] + 1e-10
            )
            attn_efficiency = attn_data["mean_dW_empirical_rank"] / (
                attn_data["mean_dW_stable_rank"] + 1e-10
            )
            axis.plot(mlp_data["layer"], mlp_efficiency, "o-", label="MLP", color="blue")
            axis.plot(attn_data["layer"], attn_efficiency, "s-", label="Attention", color="orange")
            axis.set_xlabel("Layer")
            axis.set_ylabel("Empirical / Stable Rank Ratio")
            axis.set_title("Rank Efficiency (Higher = More Concentrated)")
            axis.legend()
            axis.grid(alpha=0.3)

    # Bottom-right: total change bar chart
    axis = axes[1, 1]
    mlp_total = mlp_data["dW_fro_layer"].sum() if not mlp_data.empty else 0
    attn_total = attn_data["dW_fro_layer"].sum() if not attn_data.empty else 0

    if mlp_total > 0 or attn_total > 0:
        axis.bar(["MLP", "Attention"], [mlp_total, attn_total], color=["blue", "orange"])
        axis.set_ylabel(r"Total $\|\Delta W\|_F$ across layers")
        axis.set_title("Total Weight Change by Component")
        axis.grid(alpha=0.3, axis="y")

    plt.suptitle(title or "Detailed MLP vs Attention Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mlp_vs_attn_detailed.png"))
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare weight-change magnitudes between MLP and Attention layers."
    )
    parser.add_argument("--per-layer-csv", required=True, help="Path to per_layer.csv from collect_weight_comparison.py")
    parser.add_argument("--per-matrix-csv", required=True, help="Path to per_matrix.csv from collect_weight_comparison.py")
    parser.add_argument("--outdir", default="outputs/mlp_attn_analysis")
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()
    init_wandb("analyze_mlp_vs_attn", args)

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print(f"[analyze_mlp_vs_attn] Loading per-layer and per-matrix CSVs...")
    per_layer_df = pd.read_csv(args.per_layer_csv)
    per_matrix_df = pd.read_csv(args.per_matrix_csv)
    print(f"[analyze_mlp_vs_attn] Loaded {len(per_layer_df)} layer rows, {len(per_matrix_df)} matrix rows")

    # Generate plots
    plot_magnitude_comparison(per_layer_df, args.outdir, title=args.title)
    plot_detailed_analysis(per_layer_df, args.outdir, title=args.title)

    # Build and write summary CSV
    summary_rows = build_mlp_attn_summary(per_layer_df)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.outdir, "mlp_attn_summary.csv"), index=False)

        print(f"\n[analyze_mlp_vs_attn] MLP vs Attention Summary:")
        print(f"  Average MLP/Attention change ratio: {summary_df['ratio_mlp_attn'].mean():.3f}")
        print(f"  Layers where MLP changes more: {sum(summary_df['ratio_mlp_attn'] > 1)}/{len(summary_df)}")

        max_mlp_idx = summary_df["ratio_mlp_attn"].idxmax()
        min_mlp_idx = summary_df["ratio_mlp_attn"].idxmin()
        print(f"  Max MLP dominance (layer {summary_df.loc[max_mlp_idx, 'layer']}): {summary_df['ratio_mlp_attn'].max():.3f}x")
        print(f"  Max Attn dominance (layer {summary_df.loc[min_mlp_idx, 'layer']}): {1 / summary_df['ratio_mlp_attn'].min():.3f}x")

    print(f"\n[analyze_mlp_vs_attn] ✓ Plots and summary saved to {args.outdir}")
    log_csv_as_table(os.path.join(args.outdir, "mlp_attn_summary.csv"), "mlp_attn_summary")
    log_plots(args.outdir, "mlp_attn")
    finish_wandb()


if __name__ == "__main__":
    main()