#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "wandb",
# ]
# ///
"""
Aggregate results from multiple seed runs to compute error bars.

This script combines results from different seeds and computes mean ± std
for statistical robustness. It handles different file types:
- CSV files: Compute mean/std across seeds for numeric columns
- JSON files: Aggregate summary statistics with error bars
- PNG files: Use first seed's plots as representative

Usage:
    python aggregate_multiseed_results.py \
        --seed-dirs outputs/comp1/analysis/seed_42 outputs/comp1/analysis/seed_123 \
        --output-dir outputs/comp1/analysis \
        --sentinel-file summary.json
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
import warnings

# Allow importing from project root and experiment/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def aggregate_csv_files(csv_paths: List[str], output_path: str) -> None:
    """Aggregate CSV files across seeds by computing mean and std for numeric columns."""
    if not csv_paths:
        return

    # Load all CSV files
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        print(f"Warning: No CSV files found for aggregation at {output_path}")
        return

    # Find common columns across all dataframes
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns &= set(df.columns)

    if not common_columns:
        print(f"Warning: No common columns found across CSV files")
        return

    # Combine dataframes, keeping only common columns
    combined_df = dfs[0][list(common_columns)].copy()

    # Identify numeric columns for aggregation
    numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = [col for col in common_columns if col not in numeric_columns]

    # For numeric columns, compute mean and std across seeds
    result_df = combined_df[non_numeric_columns].copy() if non_numeric_columns else pd.DataFrame()

    for col in numeric_columns:
        values = np.array([df[col].values for df in dfs])  # Shape: (n_seeds, n_rows)

        # Compute mean and std across seeds (axis=0)
        mean_vals = np.mean(values, axis=0)
        std_vals = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean_vals)

        result_df[col] = mean_vals
        result_df[f"{col}_std"] = std_vals

    # Save aggregated results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Aggregated CSV saved to {output_path}")

def aggregate_json_files(json_paths: List[str], output_path: str) -> None:
    """Aggregate JSON summary files by computing mean and std for numeric values."""
    if not json_paths:
        return

    # Load all JSON files
    data_list = []
    for path in json_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                data_list.append(data)

    if not data_list:
        print(f"Warning: No JSON files found for aggregation at {output_path}")
        return

    # Find common keys across all JSON files
    common_keys = set(data_list[0].keys())
    for data in data_list[1:]:
        common_keys &= set(data.keys())

    # Aggregate numeric values
    aggregated = {}
    for key in common_keys:
        values = [data[key] for data in data_list]

        # Check if all values are numeric
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            # For non-numeric values, just take the first one
            aggregated[key] = values[0]

    # Add metadata about aggregation
    aggregated["_aggregation_info"] = {
        "num_seeds": len(data_list),
        "seeds_aggregated": True
    }

    # Save aggregated results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated JSON saved to {output_path}")

def copy_representative_plots(seed_dirs: List[str], output_dir: str) -> None:
    """Copy plots from the first seed directory as representative plots.

    Note: for activation_comparison CSVs this is superseded by
    plot_consolidated_activation_comparison(), which re-generates the plots
    with proper error bands from the aggregated CSV.
    """
    if not seed_dirs:
        return

    first_seed_dir = seed_dirs[0]
    if not os.path.exists(first_seed_dir):
        return

    # Find all PNG files in the first seed directory
    png_files = list(Path(first_seed_dir).glob("*.png"))

    for png_file in png_files:
        dest_path = Path(output_dir) / png_file.name
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(png_file, dest_path)
        print(f"Copied representative plot: {dest_path}")


def plot_consolidated_activation_comparison(
    aggregated_csv: str,
    output_dir: str,
    seed_dirs: List[str],
) -> None:
    """Re-generate activation comparison plots from the aggregated (mean±std) CSV.

    Unlike copy_representative_plots(), this produces shaded error-band charts
    that visualise variance across seeds, then logs them to W&B under the tag
    ``consolidated_activation_comparison``.
    """
    if not os.path.exists(aggregated_csv):
        print(f"[aggregate] Skipping consolidated plots — {aggregated_csv} not found")
        return

    # Derive a plot output directory alongside the CSV
    plot_outdir = os.path.join(output_dir, "consolidated_plots")
    os.makedirs(plot_outdir, exist_ok=True)

    # Pull model names from a seed's args (stored in the wandb config or as
    # a best-effort from the directory name).  We fall back to generic labels
    # when they cannot be determined.
    model_a_label = "Model A (before)"
    model_b_label = "Model B (after)"
    # Try to infer from the output_dir path: .../MODEL_A__to__MODEL_B/...
    for part in Path(output_dir).parts:
        if "__to__" in part:
            halves = part.split("__to__", 1)
            model_a_label = halves[0].replace("_", "/", 1)  # un-sanitize first /
            model_b_label = halves[1].replace("_", "/", 1)
            break

    num_seeds = len(seed_dirs)
    title = f"{model_b_label}: Activation Norms ({num_seeds} seeds)"

    try:
        from collect_activation_comparison import plot_activation_comparison
        plot_activation_comparison(
            aggregated_csv,
            plot_outdir,
            title=title,
            model_a=model_a_label,
            model_b=model_b_label,
        )
        print(f"[aggregate] ✓ Consolidated error-band plots written to {plot_outdir}")
    except Exception as exc:
        print(f"[aggregate] Warning: could not generate consolidated plots: {exc}")
        return

    # Log the consolidated plots to W&B under a dedicated run
    try:
        import wandb
        from utils import load_dotenv, log_line_series, finish_wandb
        load_dotenv()
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(
                project="cambridge_era",
                name=f"consolidated_activation/{model_b_label.split('/')[-1]}",
                tags=["consolidated_activation_comparison"],
                config={
                    "num_seeds": num_seeds,
                    "seed_dirs": seed_dirs,
                    "aggregated_csv": aggregated_csv,
                    "model_a": model_a_label,
                    "model_b": model_b_label,
                },
                reinit=True,
            )
            # Log native W&B line-series from the aggregated CSV
            try:
                import pandas as _pd
                agg_df = _pd.read_csv(aggregated_csv)
                agg_df = agg_df[agg_df["layer"] != "ALL_MEAN"]
                agg_df["layer"] = agg_df["layer"].astype(int)
                ma = model_a_label.split("/")[-1]
                mb = model_b_label.split("/")[-1]
                for split_name in ["forget", "retain"]:
                    sub = agg_df[agg_df["split"] == split_name].sort_values("layer")
                    if sub.empty:
                        continue
                    layers_list = sub["layer"].tolist()
                    log_line_series(
                        f"activation/{split_name}/absolute_norms",
                        xs=layers_list,
                        ys=[
                            sub["model_a_l1_norm"].tolist(),
                            sub["model_b_l1_norm"].tolist(),
                            sub["model_a_l2_norm"].tolist(),
                            sub["model_b_l2_norm"].tolist(),
                        ],
                        series_keys=[f"{ma} (L1)", f"{mb} (L1)", f"{ma} (L2)", f"{mb} (L2)"],
                        title=f"Activation Norm by Layer ({split_name}, {num_seeds} seeds)",
                        xname="Layer",
                    )
                    log_line_series(
                        f"activation/{split_name}/diff_norms",
                        xs=layers_list,
                        ys=[
                            sub["mean_diff_l1"].tolist(),
                            sub["mean_diff_l2"].tolist(),
                        ],
                        series_keys=["Diff L1 norm", "Diff L2 norm"],
                        title=f"Activation Difference Norm by Layer ({split_name}, {num_seeds} seeds)",
                        xname="Layer",
                    )
            except Exception as _exc:
                print(f"[aggregate] Warning: could not build W&B line charts: {_exc}")
            finish_wandb()
            print(f"[aggregate] ✓ Logged consolidated charts to W&B (tag: consolidated_activation_comparison)")
    except Exception as exc:
        print(f"[aggregate] W&B logging skipped: {exc}")

def find_file_patterns(seed_dirs: List[str]) -> Dict[str, List[str]]:
    """Find common file patterns across seed directories."""
    file_patterns = {}

    for seed_dir in seed_dirs:
        if not os.path.exists(seed_dir):
            continue

        for file_path in Path(seed_dir).iterdir():
            if file_path.is_file():
                filename = file_path.name
                if filename not in file_patterns:
                    file_patterns[filename] = []
                file_patterns[filename].append(str(file_path))

    return file_patterns

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results")
    parser.add_argument("--seed-dirs", nargs="+", required=True,
                       help="Directories containing results from individual seeds")
    parser.add_argument("--output-dir", required=True,
                       help="Directory to save aggregated results")
    parser.add_argument("--sentinel-file", required=True,
                       help="Name of the sentinel file to create after aggregation")

    args = parser.parse_args()

    # Find all file patterns across seed directories
    file_patterns = find_file_patterns(args.seed_dirs)

    print(f"Aggregating results from {len(args.seed_dirs)} seed directories:")
    for seed_dir in args.seed_dirs:
        print(f"  - {seed_dir}")

    # Process each file pattern
    aggregated_activation_csv = None
    for filename, file_paths in file_patterns.items():
        output_path = os.path.join(args.output_dir, filename)

        if filename.endswith('.csv'):
            aggregate_csv_files(file_paths, output_path)
            # Track activation_comparison.csv for consolidated plotting below
            if filename == "activation_comparison.csv":
                aggregated_activation_csv = output_path
        elif filename.endswith('.json'):
            aggregate_json_files(file_paths, output_path)
        elif filename.endswith('.png'):
            # Plots are handled separately
            continue
        else:
            # For other files, just copy from the first seed
            if file_paths and os.path.exists(file_paths[0]):
                os.makedirs(args.output_dir, exist_ok=True)
                shutil.copy2(file_paths[0], output_path)
                print(f"Copied file: {output_path}")

    # Generate consolidated error-band plots for activation comparison.
    # This supersedes copy_representative_plots() for activation data.
    if aggregated_activation_csv:
        plot_consolidated_activation_comparison(
            aggregated_activation_csv,
            args.output_dir,
            args.seed_dirs,
        )
    else:
        # For other analyses, fall back to copying seed-0 plots
        copy_representative_plots(args.seed_dirs, args.output_dir)

    # Create sentinel file to mark completion
    sentinel_path = os.path.join(args.output_dir, args.sentinel_file)

    if args.sentinel_file.endswith('.json'):
        # Create a JSON sentinel with aggregation metadata
        sentinel_data = {
            "aggregation_complete": True,
            "num_seeds": len(args.seed_dirs),
            "seed_directories": args.seed_dirs,
            "aggregated_files": list(file_patterns.keys())
        }
        with open(sentinel_path, 'w') as f:
            json.dump(sentinel_data, f, indent=2)
    else:
        # Create a simple sentinel file
        with open(sentinel_path, 'w') as f:
            f.write("Aggregation complete\n")

    print(f"\n✓ Multi-seed aggregation complete! Results saved to {args.output_dir}")
    print(f"✓ Sentinel file created: {sentinel_path}")

if __name__ == "__main__":
    main()