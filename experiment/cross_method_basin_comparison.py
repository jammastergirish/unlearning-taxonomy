#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "pandas",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "wandb",
# ]
# ///

"""
Cross-method basin comparison: aggregate basin analysis results across all
unlearning methods and produce comparative visualisations.

Reads basin_summary.csv from each model pair's basin_analysis directory,
tags each with its unlearning method, and produces:
  1. cross_method_basin.csv       — all methods' per-layer data combined
  2. cross_method_summary.csv     — per-method aggregate statistics
  3. cross_method_scatter.png     — weight distance vs accuracy drop, by method
  4. cross_method_selectivity.png — selectivity profiles across methods
  5. cross_method_heatmap.png     — layer × method heatmap of accuracy drop
  6. summary.json                 — per-method correlation statistics
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    infer_method_from_model_name,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_basin_results(output_root: str, model_a_dir: str) -> pd.DataFrame:
    """Find and load basin_summary.csv from all model comparisons.

    Searches for: {output_root}/{model_a_dir}__to__*/basin_analysis/basin_summary.csv
    Tags each with the inferred unlearning method.
    """
    pattern = os.path.join(
        output_root, f"{model_a_dir}__to__*", "basin_analysis", "basin_summary.csv"
    )
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        print(f"[cross_method] No basin_summary.csv files found matching: {pattern}")
        return pd.DataFrame()

    frames = []
    for csv_path in csv_paths:
        # Extract model_b from the directory structure
        comparison_dir = csv_path.split("/basin_analysis/")[0]
        model_b_part = os.path.basename(comparison_dir).split("__to__")[1]

        method = infer_method_from_model_name(model_b_part)
        if not method:
            print(f"  Skipping {csv_path} — could not infer method")
            continue

        df = pd.read_csv(csv_path)
        df["method"] = method
        df["model_b"] = model_b_part
        frames.append(df)
        print(f"  Loaded {method}: {len(df)} layers from {os.path.basename(comparison_dir)}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_per_method_stats(combined_df: pd.DataFrame) -> List[Dict]:
    """Compute summary statistics for each method."""
    rows = []
    for method, group in combined_df.groupby("method"):
        row: Dict = {"method": method, "num_layers": len(group)}

        # Mean metrics
        row["mean_weight_distance"] = round(float(group["weight_distance_total"].mean()), 4)
        row["mean_accuracy_drop"] = round(float(group["accuracy_drop"].mean()), 4)
        row["total_accuracy_drop"] = round(float(group["accuracy_drop"].sum()), 4)
        row["mean_selectivity"] = round(float(group["selectivity"].mean()), 4)
        row["max_selectivity"] = round(float(group["selectivity"].max()), 4)
        row["mean_activation_change_forget"] = round(float(group["activation_change_forget"].mean()), 4)
        row["mean_activation_change_retain"] = round(float(group["activation_change_retain"].mean()), 4)

        # Goldilocks zone
        threshold = group["accuracy_drop"].quantile(0.75)
        effective = group[group["accuracy_drop"] >= threshold]
        row["goldilocks_layers"] = sorted(effective["layer"].tolist())
        row["goldilocks_mean_distance"] = round(float(effective["weight_distance_total"].mean()), 4)

        # Correlations
        if len(group) >= 3:
            r_dist, p_dist = stats.pearsonr(
                group["weight_distance_total"], group["accuracy_drop"]
            )
            r_sel, p_sel = stats.pearsonr(
                group["selectivity"], group["accuracy_drop"]
            )
            r_mlp, p_mlp = stats.pearsonr(
                group["weight_distance_mlp"], group["accuracy_drop"]
            )
            row["r_distance_vs_drop"] = round(float(r_dist), 4)
            row["p_distance_vs_drop"] = round(float(p_dist), 4)
            row["r_selectivity_vs_drop"] = round(float(r_sel), 4)
            row["p_selectivity_vs_drop"] = round(float(p_sel), 4)
            row["r_mlp_distance_vs_drop"] = round(float(r_mlp), 4)
            row["p_mlp_distance_vs_drop"] = round(float(p_mlp), 4)

        rows.append(row)

    return sorted(rows, key=lambda r: r.get("total_accuracy_drop", 0), reverse=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Consistent color palette across all plots
METHOD_COLORS = {
    "ga_simple": "#d62728",   # red
    "ga": "#e41a1c",          # bright red
    "grad_diff": "#ff7f0e",   # orange
    "dpo": "#2ca02c",         # green
    "npo": "#1f77b4",         # blue
    "simnpo": "#9467bd",      # purple
    "rmu": "#8c564b",         # brown
    "cb": "#17becf",          # cyan
    "cb_lat": "#bcbd22",      # olive
    "lat": "#e377c2",         # pink
    "wt_dist": "#7f7f7f",     # grey
    "wt_dist_reg": "#aec7e8", # light blue
    "tar": "#ffbb78",         # light orange
}


def plot_cross_method_scatter(combined_df: pd.DataFrame, outdir: str) -> None:
    """Weight distance vs accuracy drop scatter, colored by method."""
    methods = sorted(combined_df["method"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (distance_col, label) in enumerate([
        ("weight_distance_total", "Total"),
        ("weight_distance_mlp", "MLP"),
        ("weight_distance_attn", "Attention"),
    ]):
        axis = axes[ax_idx]
        for method in methods:
            subset = combined_df[combined_df["method"] == method]
            color = METHOD_COLORS.get(method, "#333333")
            axis.scatter(
                subset[distance_col], subset["accuracy_drop"],
                c=color, label=method, s=30, alpha=0.6, edgecolors="none",
            )
        axis.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        axis.set_xlabel(f"{label} Weight Distance (Frobenius)")
        axis.set_ylabel("WMDP Accuracy Drop")
        axis.set_title(f"{label} Distance vs Accuracy Drop")
        axis.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    plt.suptitle("Cross-Method Basin Analysis: Weight Distance vs Unlearning", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_method_scatter.png"), dpi=300)
    plt.close()


def plot_selectivity_profiles(combined_df: pd.DataFrame, outdir: str) -> None:
    """Per-layer selectivity curves for each method."""
    methods = sorted(combined_df["method"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: selectivity per layer
    axis = axes[0]
    for method in methods:
        subset = combined_df[combined_df["method"] == method].sort_values("layer")
        color = METHOD_COLORS.get(method, "#333333")
        axis.plot(subset["layer"], subset["selectivity"], "-o",
                  label=method, color=color, markersize=3, alpha=0.8)
    axis.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Selectivity (Forget / Retain)")
    axis.set_title("Selectivity Profile by Method")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.3)

    # Right: accuracy drop per layer
    axis = axes[1]
    for method in methods:
        subset = combined_df[combined_df["method"] == method].sort_values("layer")
        color = METHOD_COLORS.get(method, "#333333")
        axis.plot(subset["layer"], subset["accuracy_drop"], "-o",
                  label=method, color=color, markersize=3, alpha=0.8)
    axis.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
    axis.set_xlabel("Layer")
    axis.set_ylabel("WMDP Accuracy Drop")
    axis.set_title("Per-Layer Accuracy Drop by Method")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.3)

    plt.suptitle("Cross-Method: Selectivity and Accuracy Drop Profiles", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_method_selectivity.png"), dpi=300)
    plt.close()


def plot_method_heatmap(combined_df: pd.DataFrame, outdir: str) -> None:
    """Layer × method heatmap of accuracy drop."""
    pivot = combined_df.pivot_table(
        index="method", columns="layer", values="accuracy_drop", aggfunc="mean"
    )
    if pivot.empty:
        return

    # Sort methods by total accuracy drop (most aggressive at top)
    method_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[method_order]

    fig, axis = plt.subplots(figsize=(18, max(4, len(pivot) * 0.8)))
    im = axis.imshow(pivot.values, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axis.set_yticks(range(len(pivot)))
    axis.set_yticklabels(pivot.index, fontsize=9)

    # Show every 4th layer label to avoid crowding
    layer_labels = pivot.columns.tolist()
    tick_positions = range(0, len(layer_labels), 4)
    axis.set_xticks(list(tick_positions))
    axis.set_xticklabels([layer_labels[i] for i in tick_positions])
    axis.set_xlabel("Layer")
    axis.set_title("Accuracy Drop by Method and Layer (red = more drop)")
    plt.colorbar(im, ax=axis, label="Accuracy Drop", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_method_heatmap.png"), dpi=300)
    plt.close()


def plot_method_summary_bars(summary_rows: List[Dict], outdir: str) -> None:
    """Bar chart comparing key metrics across methods."""
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    methods = df["method"].tolist()
    colors = [METHOD_COLORS.get(m, "#333333") for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: total accuracy drop
    axis = axes[0, 0]
    axis.barh(methods, df["total_accuracy_drop"], color=colors, alpha=0.8)
    axis.set_xlabel("Total Accuracy Drop (sum across layers)")
    axis.set_title("Unlearning Strength")
    axis.grid(alpha=0.3, axis="x")

    # Top-right: mean selectivity
    axis = axes[0, 1]
    axis.barh(methods, df["mean_selectivity"], color=colors, alpha=0.8)
    axis.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Mean Selectivity (Forget / Retain)")
    axis.set_title("Targeting Precision")
    axis.grid(alpha=0.3, axis="x")

    # Bottom-left: mean weight distance
    axis = axes[1, 0]
    axis.barh(methods, df["mean_weight_distance"], color=colors, alpha=0.8)
    axis.set_xlabel("Mean Weight Distance (Frobenius)")
    axis.set_title("Intervention Magnitude")
    axis.grid(alpha=0.3, axis="x")

    # Bottom-right: selectivity ↔ accuracy drop correlation
    if "r_selectivity_vs_drop" in df.columns:
        axis = axes[1, 1]
        r_vals = df["r_selectivity_vs_drop"].fillna(0)
        axis.barh(methods, r_vals, color=colors, alpha=0.8)
        axis.set_xlabel("Pearson r (selectivity vs accuracy drop)")
        axis.set_title("How Well Selectivity Predicts Forgetting")
        axis.set_xlim(-1, 1)
        axis.grid(alpha=0.3, axis="x")

    plt.suptitle("Cross-Method Comparison: Basin Analysis Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_method_summary.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare basin analysis results across unlearning methods."
    )
    parser.add_argument("--output-root", default="outputs",
                        help="Root directory containing all model comparison outputs")
    parser.add_argument("--model-a", default="EleutherAI/deep-ignorance-unfiltered",
                        help="Base model (used to find comparison directories)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: {output-root}/cross_method_basin)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    model_a_dir = args.model_a.replace("/", "_")
    outdir = args.outdir or os.path.join(args.output_root, "cross_method_basin")
    os.makedirs(outdir, exist_ok=True)

    init_wandb("cross_method_basin", args)

    # Load all basin results
    print("[cross_method] Searching for basin results...")
    combined_df = load_all_basin_results(args.output_root, model_a_dir)

    if combined_df.empty:
        print("[cross_method] No basin results found. Run the pipeline first.")
        finish_wandb()
        return

    methods = sorted(combined_df["method"].unique())
    print(f"[cross_method] Found {len(methods)} methods: {', '.join(methods)}")
    print(f"[cross_method] Total rows: {len(combined_df)}")

    # Save combined data
    combined_csv = os.path.join(outdir, "cross_method_basin.csv")
    combined_df.to_csv(combined_csv, index=False)

    # Compute per-method statistics
    summary_rows = compute_per_method_stats(combined_df)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(outdir, "cross_method_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Save JSON summary
    summary_json = os.path.join(outdir, "summary.json")
    # Convert goldilocks_layers lists for JSON serialization
    json_rows = []
    for row in summary_rows:
        json_row = {k: v for k, v in row.items()}
        json_row["goldilocks_layers"] = str(json_row.get("goldilocks_layers", []))
        json_rows.append(json_row)
    with open(summary_json, "w") as fh:
        json.dump({"methods": json_rows, "num_methods": len(methods)}, fh, indent=2)

    # Generate plots
    print("[cross_method] Generating comparison plots...")
    plot_cross_method_scatter(combined_df, outdir)
    plot_selectivity_profiles(combined_df, outdir)
    plot_method_heatmap(combined_df, outdir)
    plot_method_summary_bars(summary_rows, outdir)

    # Print summary table
    print(f"\n[cross_method] Summary (sorted by total accuracy drop):")
    print(f"  {'Method':<15} {'Total Drop':>10} {'Mean Sel':>10} {'Mean Dist':>10} {'r(sel,drop)':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for row in summary_rows:
        r_sel = row.get("r_selectivity_vs_drop", float("nan"))
        print(f"  {row['method']:<15} {row['total_accuracy_drop']:>10.4f} "
              f"{row['mean_selectivity']:>10.2f} {row['mean_weight_distance']:>10.4f} "
              f"{r_sel:>12.4f}")

    print(f"\n[cross_method] Results saved to {outdir}")
    log_csv_as_table(combined_csv, "cross_method_basin")
    log_csv_as_table(summary_csv, "cross_method_summary")
    log_plots(outdir, "cross_method")
    finish_wandb()


if __name__ == "__main__":
    main()
