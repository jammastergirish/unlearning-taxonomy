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
Basin analysis: correlate per-layer weight distance from the pretrained model
with unlearning effectiveness (WMDP accuracy drop) and activation changes.

Inspired by the "Goldilocks basin" finding in subliminal learning research:
effective interventions move the model a specific distance from initialization
--- too little changes nothing, too much destroys general capability.

Reads CSVs produced by earlier pipeline steps:
  - Step 1:  per_coarse_layer.csv   (per-layer weight distances)
  - Step 3:  activation_comparison.csv  (per-layer activation norm changes)
  - Step 5:  wmdp_logit_lens results for both models (per-layer accuracy)

Produces:
  1. basin_summary.csv          — per-layer metrics joining all three sources
  2. basin_goldilocks.png       — scatter of weight distance vs accuracy drop
  3. basin_profile.png          — per-layer distance colored by effectiveness
  4. summary.json               — correlation statistics
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    init_wandb,
    infer_method_from_model_name,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_basin_summary(
    weight_df: pd.DataFrame,
    activation_df: pd.DataFrame,
    wmdp_a_df: pd.DataFrame,
    wmdp_b_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join per-layer weight distances, activation changes, and WMDP accuracy.

    Returns a DataFrame with one row per layer containing:
      - weight_distance_mlp, weight_distance_attn, weight_distance_total
      - rel_weight_distance_mlp, rel_weight_distance_attn
      - activation_change_forget, activation_change_retain, selectivity
      - wmdp_accuracy_a, wmdp_accuracy_b, accuracy_drop
    """
    rows: List[Dict] = []

    # Index weight data by layer
    mlp_weights = weight_df[weight_df["group"] == "mlp"].set_index("layer")
    attn_weights = weight_df[weight_df["group"] == "attn"].set_index("layer")

    # Index activation data by layer (exclude summary rows)
    forget_activations = activation_df[
        (activation_df["split"] == "forget") & (activation_df["layer"] != "ALL_MEAN")
    ].copy()
    forget_activations["layer"] = forget_activations["layer"].astype(int)
    forget_activations = forget_activations.set_index("layer")

    retain_activations = activation_df[
        (activation_df["split"] == "retain") & (activation_df["layer"] != "ALL_MEAN")
    ].copy()
    retain_activations["layer"] = retain_activations["layer"].astype(int)
    retain_activations = retain_activations.set_index("layer")

    # Index WMDP accuracy by layer
    wmdp_a = wmdp_a_df.set_index("layer")
    wmdp_b = wmdp_b_df.set_index("layer")

    # Determine common layers across all sources
    common_layers = sorted(
        set(mlp_weights.index)
        & set(attn_weights.index)
        & set(wmdp_a.index)
        & set(wmdp_b.index)
    )

    for layer in common_layers:
        mlp_dist = float(mlp_weights.loc[layer, "dW_fro_layer"])
        attn_dist = float(attn_weights.loc[layer, "dW_fro_layer"])
        total_dist = np.sqrt(mlp_dist**2 + attn_dist**2)

        mlp_rel = float(mlp_weights.loc[layer, "dW_fro_layer_rel"])
        attn_rel = float(attn_weights.loc[layer, "dW_fro_layer_rel"])

        # Activation changes (L2 norm of difference)
        forget_change = 0.0
        retain_change = 0.0
        if layer in forget_activations.index:
            forget_change = float(forget_activations.loc[layer, "mean_diff_l2"])
        if layer in retain_activations.index:
            retain_change = float(retain_activations.loc[layer, "mean_diff_l2"])

        selectivity = forget_change / (retain_change + 1e-10)

        # WMDP accuracy
        acc_a = float(wmdp_a.loc[layer, "accuracy"])
        acc_b = float(wmdp_b.loc[layer, "accuracy"])
        accuracy_drop = acc_a - acc_b  # positive = unlearning reduced accuracy

        rows.append({
            "layer": layer,
            "weight_distance_mlp": mlp_dist,
            "weight_distance_attn": attn_dist,
            "weight_distance_total": total_dist,
            "rel_weight_distance_mlp": mlp_rel,
            "rel_weight_distance_attn": attn_rel,
            "activation_change_forget": forget_change,
            "activation_change_retain": retain_change,
            "selectivity": selectivity,
            "wmdp_accuracy_a": acc_a,
            "wmdp_accuracy_b": acc_b,
            "accuracy_drop": accuracy_drop,
        })

    return pd.DataFrame(rows)


def compute_basin_statistics(summary_df: pd.DataFrame) -> Dict:
    """Compute correlation statistics for the basin analysis."""
    result: Dict = {}

    if len(summary_df) < 3:
        return {"error": "Too few layers for correlation analysis"}

    # Weight distance vs accuracy drop
    for distance_col in ["weight_distance_total", "weight_distance_mlp", "weight_distance_attn"]:
        pearson_r, pearson_p = stats.pearsonr(
            summary_df[distance_col], summary_df["accuracy_drop"]
        )
        spearman_rho, spearman_p = stats.spearmanr(
            summary_df[distance_col], summary_df["accuracy_drop"]
        )
        col_label = distance_col.replace("weight_distance_", "")
        result[f"pearson_r_{col_label}_vs_accuracy_drop"] = round(float(pearson_r), 4)
        result[f"pearson_p_{col_label}_vs_accuracy_drop"] = round(float(pearson_p), 4)
        result[f"spearman_rho_{col_label}_vs_accuracy_drop"] = round(float(spearman_rho), 4)
        result[f"spearman_p_{col_label}_vs_accuracy_drop"] = round(float(spearman_p), 4)

    # Selectivity vs accuracy drop
    pearson_r, pearson_p = stats.pearsonr(
        summary_df["selectivity"], summary_df["accuracy_drop"]
    )
    result["pearson_r_selectivity_vs_accuracy_drop"] = round(float(pearson_r), 4)
    result["pearson_p_selectivity_vs_accuracy_drop"] = round(float(pearson_p), 4)

    # Activation change (forget) vs accuracy drop
    pearson_r, pearson_p = stats.pearsonr(
        summary_df["activation_change_forget"], summary_df["accuracy_drop"]
    )
    result["pearson_r_forget_activation_vs_accuracy_drop"] = round(float(pearson_r), 4)
    result["pearson_p_forget_activation_vs_accuracy_drop"] = round(float(pearson_p), 4)

    # Summary statistics
    result["mean_weight_distance_total"] = round(float(summary_df["weight_distance_total"].mean()), 4)
    result["std_weight_distance_total"] = round(float(summary_df["weight_distance_total"].std()), 4)
    result["mean_accuracy_drop"] = round(float(summary_df["accuracy_drop"].mean()), 4)
    result["mean_selectivity"] = round(float(summary_df["selectivity"].mean()), 4)

    # Identify layers in the "Goldilocks zone" (top quartile of accuracy drop)
    threshold = summary_df["accuracy_drop"].quantile(0.75)
    effective_layers = summary_df[summary_df["accuracy_drop"] >= threshold]
    result["goldilocks_threshold"] = round(float(threshold), 4)
    result["goldilocks_layers"] = sorted(effective_layers["layer"].tolist())
    result["goldilocks_mean_distance"] = round(
        float(effective_layers["weight_distance_total"].mean()), 4
    )
    result["non_goldilocks_mean_distance"] = round(
        float(summary_df[summary_df["accuracy_drop"] < threshold]["weight_distance_total"].mean()), 4
    )

    result["num_layers"] = len(summary_df)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_goldilocks_scatter(summary_df: pd.DataFrame, outdir: str, title: Optional[str] = None) -> None:
    """Scatter of per-layer weight distance vs WMDP accuracy drop."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax_idx, (distance_col, label) in enumerate([
        ("weight_distance_total", "Total"),
        ("weight_distance_mlp", "MLP"),
        ("weight_distance_attn", "Attention"),
    ]):
        axis = axes[ax_idx]
        scatter = axis.scatter(
            summary_df[distance_col],
            summary_df["accuracy_drop"],
            c=summary_df["layer"],
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        plt.colorbar(scatter, ax=axis, label="Layer")

        # Add layer labels
        for _, row in summary_df.iterrows():
            axis.annotate(
                str(int(row["layer"])),
                (row[distance_col], row["accuracy_drop"]),
                fontsize=6,
                ha="center",
                va="bottom",
                alpha=0.7,
            )

        # Trend line
        if len(summary_df) >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                summary_df[distance_col], summary_df["accuracy_drop"]
            )
            x_range = np.linspace(
                summary_df[distance_col].min(),
                summary_df[distance_col].max(),
                100,
            )
            axis.plot(x_range, slope * x_range + intercept, "r--", alpha=0.5,
                      label=f"r={r_value:.3f}, p={p_value:.3f}")
            axis.legend(fontsize=8)

        axis.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        axis.set_xlabel(f"{label} Weight Distance (Frobenius)")
        axis.set_ylabel("WMDP Accuracy Drop (A - B)")
        axis.set_title(f"{label} Distance vs Accuracy Drop")
        axis.grid(alpha=0.3)

    plt.suptitle(title or "Basin Analysis: Weight Distance vs Unlearning Effectiveness", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "basin_goldilocks.png"), dpi=300)
    plt.close()


def plot_basin_profile(summary_df: pd.DataFrame, outdir: str, title: Optional[str] = None) -> None:
    """Per-layer distance profile colored by unlearning effectiveness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: weight distance per layer (MLP vs Attn)
    axis = axes[0, 0]
    axis.bar(
        summary_df["layer"] - 0.2, summary_df["weight_distance_mlp"],
        width=0.4, label="MLP", color="steelblue", alpha=0.8,
    )
    axis.bar(
        summary_df["layer"] + 0.2, summary_df["weight_distance_attn"],
        width=0.4, label="Attention", color="coral", alpha=0.8,
    )
    axis.set_xlabel("Layer")
    axis.set_ylabel("Weight Distance (Frobenius)")
    axis.set_title("Per-Layer Weight Distance from Base Model")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")

    # Top-right: WMDP accuracy per layer (before vs after)
    axis = axes[0, 1]
    axis.plot(summary_df["layer"], summary_df["wmdp_accuracy_a"], "o-",
              label="Base (A)", color="green", markersize=4)
    axis.plot(summary_df["layer"], summary_df["wmdp_accuracy_b"], "s-",
              label="Unlearned (B)", color="red", markersize=4)
    axis.fill_between(
        summary_df["layer"],
        summary_df["wmdp_accuracy_b"],
        summary_df["wmdp_accuracy_a"],
        alpha=0.2, color="red", label="Accuracy drop",
    )
    axis.set_xlabel("Layer")
    axis.set_ylabel("WMDP Accuracy")
    axis.set_title("Layer-wise WMDP Accuracy (Logit Lens)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # Bottom-left: activation changes (forget vs retain)
    axis = axes[1, 0]
    axis.plot(summary_df["layer"], summary_df["activation_change_forget"], "o-",
              label="Forget", color="red", markersize=4)
    axis.plot(summary_df["layer"], summary_df["activation_change_retain"], "s-",
              label="Retain", color="blue", markersize=4)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Activation Change (L2)")
    axis.set_title("Activation Norm Change: Forget vs Retain")
    axis.legend()
    axis.grid(alpha=0.3)

    # Bottom-right: selectivity (forget / retain change ratio)
    axis = axes[1, 1]
    colors = ["red" if s > 1.5 else "orange" if s > 1.0 else "gray"
              for s in summary_df["selectivity"]]
    axis.bar(summary_df["layer"], summary_df["selectivity"], color=colors, alpha=0.8)
    axis.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Selectivity = 1.0")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Selectivity (Forget / Retain)")
    axis.set_title("Unlearning Selectivity per Layer")
    axis.legend()
    axis.grid(alpha=0.3, axis="y")

    plt.suptitle(title or "Basin Profile: Per-Layer Unlearning Characterisation", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "basin_profile.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Basin analysis: correlate weight distance with unlearning effectiveness."
    )
    parser.add_argument(
        "--weight-csv", required=True,
        help="Path to per_coarse_layer.csv from collect_weight_comparison.py",
    )
    parser.add_argument(
        "--activation-csv", required=True,
        help="Path to activation_comparison.csv from collect_activation_comparison.py",
    )
    parser.add_argument(
        "--wmdp-a-csv", required=True,
        help="Path to wmdp_lens_results.csv for model A (base model)",
    )
    parser.add_argument(
        "--wmdp-b-csv", required=True,
        help="Path to wmdp_lens_results.csv for model B (unlearned model)",
    )
    parser.add_argument("--outdir", default="outputs/basin_analysis")
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    method = infer_method_from_model_name(args.outdir)
    init_wandb("basin_analysis", args, method=method)

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("[basin_analysis] Loading CSVs from earlier pipeline steps...")
    weight_df = pd.read_csv(args.weight_csv)
    activation_df = pd.read_csv(args.activation_csv)
    wmdp_a_df = pd.read_csv(args.wmdp_a_csv)
    wmdp_b_df = pd.read_csv(args.wmdp_b_csv)
    print(f"  Weight rows: {len(weight_df)}, Activation rows: {len(activation_df)}")
    print(f"  WMDP-A layers: {len(wmdp_a_df)}, WMDP-B layers: {len(wmdp_b_df)}")

    # Build summary
    summary_df = build_basin_summary(weight_df, activation_df, wmdp_a_df, wmdp_b_df)
    if summary_df.empty:
        print("[basin_analysis] No overlapping layers found — cannot produce basin analysis.")
        finish_wandb()
        return

    print(f"[basin_analysis] Joined {len(summary_df)} layers across all sources.")

    # Save summary CSV
    summary_csv_path = os.path.join(args.outdir, "basin_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Compute statistics
    basin_stats = compute_basin_statistics(summary_df)

    # Save summary JSON
    summary_json_path = os.path.join(args.outdir, "summary.json")
    with open(summary_json_path, "w") as fh:
        json.dump(basin_stats, fh, indent=2)

    # Generate plots
    plot_goldilocks_scatter(summary_df, args.outdir, title=args.title)
    plot_basin_profile(summary_df, args.outdir, title=args.title)

    # Print key findings
    print(f"\n[basin_analysis] Key findings:")
    print(f"  Mean weight distance (total): {basin_stats.get('mean_weight_distance_total', 'N/A')}")
    print(f"  Mean accuracy drop:           {basin_stats.get('mean_accuracy_drop', 'N/A')}")
    print(f"  Mean selectivity:             {basin_stats.get('mean_selectivity', 'N/A')}")

    total_key = "pearson_r_total_vs_accuracy_drop"
    if total_key in basin_stats:
        print(f"\n  Weight distance ↔ accuracy drop:")
        print(f"    Total:  r={basin_stats['pearson_r_total_vs_accuracy_drop']}, p={basin_stats['pearson_p_total_vs_accuracy_drop']}")
        print(f"    MLP:    r={basin_stats['pearson_r_mlp_vs_accuracy_drop']}, p={basin_stats['pearson_p_mlp_vs_accuracy_drop']}")
        print(f"    Attn:   r={basin_stats['pearson_r_attn_vs_accuracy_drop']}, p={basin_stats['pearson_p_attn_vs_accuracy_drop']}")

    selectivity_key = "pearson_r_selectivity_vs_accuracy_drop"
    if selectivity_key in basin_stats:
        print(f"\n  Selectivity ↔ accuracy drop:")
        print(f"    r={basin_stats[selectivity_key]}, p={basin_stats['pearson_p_selectivity_vs_accuracy_drop']}")

    if "goldilocks_layers" in basin_stats:
        print(f"\n  Goldilocks zone (top-quartile accuracy drop):")
        print(f"    Layers:               {basin_stats['goldilocks_layers']}")
        print(f"    Mean distance:        {basin_stats['goldilocks_mean_distance']}")
        print(f"    Non-Goldilocks mean:  {basin_stats['non_goldilocks_mean_distance']}")

    print(f"\n[basin_analysis] Results saved to {args.outdir}")
    log_csv_as_table(summary_csv_path, "basin_summary")
    log_plots(args.outdir, "basin")
    finish_wandb()


if __name__ == "__main__":
    main()
