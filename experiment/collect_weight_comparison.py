#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
#   "safetensors",
#   "huggingface_hub",
#   "wandb",
#   "pandas",
#   "matplotlib",
# ]
# ///

"""
Per-component, per-layer weight comparison between model checkpoints.

Computes a comprehensive set of metrics for each weight matrix:
  - Cosine similarity, element-wise diff stats
  - Frobenius norm (absolute, relative, normalized)
  - Spectral norm (dW and W, relative)
  - Stable rank (dW and W)
  - Optional: Empirical rank via full SVD

Outputs four CSVs:
  - per_matrix.csv: one row per weight matrix (all metrics)
  - per_component.csv: aggregate stats per granular component (qkv/proj/mlp_expand/mlp_contract) across layers
  - per_layer.csv: aggregate stats per layer per granular component (qkv/proj/mlp_expand/mlp_contract)
  - per_coarse_layer.csv: aggregate stats per layer per coarse group (attn/mlp)
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    SmartLoader,
    resolve_device,
    resolve_dtype,
    extract_layer,
    classify_granular,
    stable_rank_and_spectral,
    empirical_rank,
    write_csv,
    comparison_outdir,
    spectral_norm_power,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)

# Coarse group mapping: granular component -> coarse group (for per_coarse_layer.csv)
_COARSE_MAP = {
    "qkv": "attn",
    "proj": "attn",
    "mlp_expand": "mlp",
    "mlp_contract": "mlp",
}

_GRANULAR_COMPONENTS = ["qkv", "proj", "mlp_expand", "mlp_contract"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(
    Wa: torch.Tensor,
    Wb: torch.Tensor,
    sr_iters: int = 5,
    do_empirical_rank: bool = False,
    empirical_threshold: float = 0.99,
) -> dict:
    """Compute all per-matrix metrics for a weight pair (Wa=baseline, Wb=target)."""
    Wa_f = Wa.float()
    Wb_f = Wb.float()

    # Element-wise difference (change from A to B)
    dW = Wb_f - Wa_f

    # Cosine similarity between full weight vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        Wa_f.flatten().unsqueeze(0), Wb_f.flatten().unsqueeze(0)
    ).item()

    # Element-wise diff stats
    dW_flat = dW.flatten()
    n_elem = dW_flat.numel()
    diff_mean = dW_flat.mean().item()
    diff_std = dW_flat.std().item()
    diff_abs_mean = dW_flat.abs().mean().item()

    # Frobenius norms
    dW_fro = float(dW.norm().item())
    W_fro = float(Wa_f.norm().item())
    Wb_fro = float(Wb_f.norm().item())
    rel_fro = dW_fro / W_fro if W_fro > 0 else float("inf")
    fro_norm_normalized = dW_fro / (n_elem ** 0.5)

    # Stable rank and spectral norm (for both dW and W)
    dW_sr, dW_spec = stable_rank_and_spectral(dW, iters=sr_iters)
    W_sr, W_spec = stable_rank_and_spectral(Wa_f, iters=sr_iters)
    Wb_sr, Wb_spec = stable_rank_and_spectral(Wb_f, iters=sr_iters)
    dW_spec_rel = dW_spec / W_spec if W_spec > 0 else 0.0

    row = {
        "elements": n_elem,
        "cosine_sim": cos_sim,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_abs_mean": diff_abs_mean,
        "frobenius_norm": dW_fro,
        "W_fro": W_fro,
        "Wb_fro": Wb_fro,
        "rel_frobenius": rel_fro,
        "fro_norm_normalized": fro_norm_normalized,
        "diff_spectral_norm": dW_spec,
        "W_spectral": W_spec,
        "Wb_spectral": Wb_spec,
        "dW_spectral_rel": dW_spec_rel,
        "dW_stable_rank": dW_sr,
        "W_stable_rank": W_sr,
        "Wb_stable_rank": Wb_sr,
    }

    if do_empirical_rank:
        row["dW_empirical_rank"] = empirical_rank(dW, threshold=empirical_threshold)
        row["W_empirical_rank"] = empirical_rank(Wa_f, threshold=empirical_threshold)

    return row


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _pick_sanity_params(all_linear_names: List[str], n: int = 3) -> List[str]:
    """Pick up to n weight names that belong to known components for sanity checks."""
    picked = []
    for name in all_linear_names:
        if classify_granular(name) != "other" and extract_layer(name) is not None:
            picked.append(name)
            if len(picked) >= n:
                break
    return picked


def run_sanity_checks(
    loader_a: SmartLoader,
    loader_b: SmartLoader,
    linear_names: List[str],
    device: str,
    dtype: torch.dtype,
) -> bool:
    """Run sanity checks. Returns True if all pass, False otherwise."""
    sample = _pick_sanity_params(linear_names)
    if not sample:
        print("FAIL: No suitable weight matrices found for sanity checks")
        return False

    all_passed = True

    # --- Check 1: Self-comparison (A vs A) ---
    print("\n--- Sanity Check 1: Self-comparison (A vs A) ---")
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        m = _compute_metrics(Wa, Wa)

        cos_ok = m["cosine_sim"] == 1.0
        fro_ok = m["rel_frobenius"] == 0.0
        passed = cos_ok and fro_ok

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not cos_ok:
            print(f"         cosine_sim = {m['cosine_sim']} (expected 1.0)")
        if not fro_ok:
            print(f"         rel_frobenius = {m['rel_frobenius']} (expected 0.0)")

        if not passed:
            all_passed = False
        del Wa

    # --- Check 2: Symmetry cos(A,B) == cos(B,A) ---
    print("\n--- Sanity Check 2: Cosine similarity symmetry ---")
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wa.shape != Wb.shape:
            print(f"  SKIP: {name} (shape mismatch or missing)")
            continue

        m_ab = _compute_metrics(Wa, Wb)
        m_ba = _compute_metrics(Wb, Wa)

        passed = abs(m_ab["cosine_sim"] - m_ba["cosine_sim"]) < 1e-6
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        print(f"         cos(A,B) = {m_ab['cosine_sim']:.10f}")
        print(f"         cos(B,A) = {m_ba['cosine_sim']:.10f}")
        if not passed:
            print(f"         delta = {abs(m_ab['cosine_sim'] - m_ba['cosine_sim']):.2e}")
            all_passed = False

        del Wa, Wb

    # --- Check 3: Valid ranges ---
    print("\n--- Sanity Check 3: Value ranges ---")
    for name in sample:
        Wa = loader_a.get_param(name, device, dtype)
        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None or Wa.shape != Wb.shape:
            continue

        m = _compute_metrics(Wa, Wb)

        cos_ok = -1.0 <= m["cosine_sim"] <= 1.0
        fro_ok = m["rel_frobenius"] >= 0.0
        norm_ok = m["frobenius_norm"] >= 0.0

        passed = cos_ok and fro_ok and norm_ok
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not cos_ok:
            print(f"         cosine_sim = {m['cosine_sim']} (out of [-1, 1])")
        if not fro_ok:
            print(f"         rel_frobenius = {m['rel_frobenius']} (negative)")
        if not norm_ok:
            print(f"         frobenius_norm = {m['frobenius_norm']} (negative)")

        if not passed:
            all_passed = False
        del Wa, Wb

    # --- Summary ---
    print()
    if all_passed:
        print("All sanity checks PASSED")
    else:
        print("Some sanity checks FAILED — aborting")
    print()
    return all_passed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_weight_comparison(per_matrix_csv: str, outdir: str, title: str = None,
                           model_a: str = "", model_b: str = ""):
    """Generate per-component plots from per_matrix.csv."""
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(per_matrix_csv)
    print(f"Generating plots from {per_matrix_csv} ({len(df)} rows)")

    components = [c for c in _GRANULAR_COMPONENTS if c in df["component"].values]
    if not components:
        print("No recognized components found — skipping plots")
        return

    # ---- Plot A: Relative Frobenius norm (layer locality) ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["rel_frobenius"], marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\|\Delta W\|_F / \|W\|_F$")
        ax.set_title(f"{comp}")
        ax.grid(alpha=0.3)
    _subtitle = f"\n{model_a.split('/')[-1]} → {model_b.split('/')[-1]}" if model_a else ""
    fig.suptitle((title or "Layer locality — relative Frobenius norm") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "layer_locality.png"))
    plt.close(fig)

    # ---- Plot A2: Absolute Frobenius norm (baseline vs unlearned) ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    label_a = model_a.split("/")[-1] if model_a else "baseline"
    label_b = model_b.split("/")[-1] if model_b else "unlearned"
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["W_fro"], marker="o", color="tab:blue", label=label_a)
        if "Wb_fro" in sub.columns:
            ax.plot(sub["layer"], sub["Wb_fro"], marker="o", color="tab:orange", label=label_b)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\|W\|_F$")
        ax.set_title(f"{comp}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle((title or "Absolute Frobenius norm — baseline vs unlearned") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "absolute_frobenius.png"))
    plt.close(fig)

    # ---- Plot B: Stable rank of dW ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["dW_stable_rank"], marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Stable rank of $\Delta W$")
        ax.set_title(f"{comp}")
        ax.grid(alpha=0.3)
    fig.suptitle((title or "Edit dimensionality — stable rank") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stable_rank.png"))
    plt.close(fig)

    # ---- Plot B2: Absolute stable rank (baseline vs unlearned) ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["W_stable_rank"], marker="o", color="tab:blue", label=label_a)
        if "Wb_stable_rank" in sub.columns:
            ax.plot(sub["layer"], sub["Wb_stable_rank"], marker="o", color="tab:orange", label=label_b)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Stable rank of $W$")
        ax.set_title(f"{comp}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle((title or "Absolute stable rank — baseline vs unlearned") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "absolute_stable_rank.png"))
    plt.close(fig)

    # ---- Plot C: Relative spectral norm ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["dW_spectral_rel"], marker="o", color="tab:red")
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\sigma_1(\Delta W) / \sigma_1(W)$")
        ax.set_title(f"{comp}")
        ax.grid(alpha=0.3)
    fig.suptitle((title or "Spectral norm — worst-case amplification") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spectral_norm.png"))
    plt.close(fig)

    # ---- Plot C2: Absolute spectral norm (baseline vs unlearned) ----
    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
    for i, comp in enumerate(components):
        ax = axes[0, i]
        sub = df[df["component"] == comp].sort_values("layer")
        ax.plot(sub["layer"], sub["W_spectral"], marker="o", color="tab:blue", label=label_a)
        if "Wb_spectral" in sub.columns:
            ax.plot(sub["layer"], sub["Wb_spectral"], marker="o", color="tab:orange", label=label_b)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\sigma_1(W)$")
        ax.set_title(f"{comp}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle((title or "Absolute spectral norm — baseline vs unlearned") + _subtitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "absolute_spectral_norm.png"))
    plt.close(fig)

    # ---- Plot D: Empirical rank (if present) ----
    if "dW_empirical_rank" in df.columns:
        fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 5), squeeze=False)
        for i, comp in enumerate(components):
            ax = axes[0, i]
            sub = df[df["component"] == comp].sort_values("layer")
            ax.plot(sub["layer"], sub["dW_empirical_rank"], marker="o", color="darkorange")
            ax.set_xlabel("Layer")
            ax.set_ylabel(r"Empirical rank of $\Delta W$")
            ax.set_title(f"{comp}")
            ax.grid(alpha=0.3)
        fig.suptitle((title or "Edit dimensionality — empirical rank") + _subtitle, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "empirical_rank.png"))
        plt.close(fig)

    print(f"All plots written to {outdir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-component, per-layer weight comparison metrics "
                    "between two model checkpoints."
    )
    ap.add_argument("--model-a", required=True, help="Baseline / before model path")
    ap.add_argument("--model-b", required=True, help="Target / after model path")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--outdir", default=None,
                    help="Output dir (default: auto-derived from model names)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-check", action="store_true",
                    help="Run sanity checks before main comparison and exit early on failure")
    ap.add_argument("--trust-remote-code", action="store_true")
    # Flags from collect_weight_comparison
    ap.add_argument("--sr-iters", type=int, default=5,
                    help="Power iteration count for spectral norm (default: 5)")
    ap.add_argument("--empirical-rank", action="store_true", default=False,
                    help="Compute empirical rank via full SVD (slow, off by default)")
    ap.add_argument("--empirical-threshold", type=float, default=0.99,
                    help="Fraction of variance to capture for empirical rank (default: 0.99)")
    ap.add_argument("--plot-outdir", default=None,
                    help="Generate plots in this directory (default: no plots)")
    ap.add_argument("--title", default=None,
                    help="Title for generated plots")
    args = ap.parse_args()

    # Derive output directory
    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="weight_comparison")

    init_wandb("weight_comparison", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")

    try:
        loader_a = SmartLoader(args.model_a)
        loader_b = SmartLoader(args.model_b)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    names_a = loader_a.get_all_param_names()
    names_b = loader_b.get_all_param_names()
    all_names = sorted(names_a.intersection(names_b))

    # Filter to weight tensors only
    linear_names = [n for n in all_names if n.endswith(".weight")]

    if args.sanity_check:
        if not run_sanity_checks(loader_a, loader_b, linear_names, device, dtype):
            return

    # ---- Main comparison loop ----
    rows = []
    per_layer_accum = {}  # (layer, coarse_group) -> running stats

    print(f"Scanning {len(linear_names)} weight matrices...")

    for name in tqdm(linear_names, desc="Comparing weight matrices", unit="matrix"):
        Wa = loader_a.get_param(name, device, dtype)
        if Wa is None or Wa.ndim != 2:
            continue

        Wb = loader_b.get_param(name, device, dtype)
        if Wb is None:
            continue

        if Wa.shape != Wb.shape:
            print(f"Skipping {name}: shape mismatch {Wa.shape} vs {Wb.shape}")
            continue

        layer = extract_layer(name)
        component = classify_granular(name)

        # Skip non-layer or non-component weights (embeddings, final LN, etc.)
        if layer is None or component == "other":
            del Wa, Wb
            continue

        metrics = _compute_metrics(
            Wa, Wb,
            sr_iters=args.sr_iters,
            do_empirical_rank=args.empirical_rank,
            empirical_threshold=args.empirical_threshold,
        )

        row = {
            "name": name,
            "layer": layer,
            "component": component,
            "shape0": Wa.shape[0],
            "shape1": Wa.shape[1],
        }
        row.update(metrics)
        rows.append(row)

        # Accumulate per-layer stats (coarse groups for backward compat)
        coarse = _COARSE_MAP.get(component)
        if coarse and layer is not None:
            key = (layer, coarse)
            defaults = {
                "sum_dW_fro_sq": 0.0, "sum_W_fro_sq": 0.0,
                "sum_dW_sr": 0.0,
                "max_dW_spec": 0.0, "max_W_spec": 0.0,
                "count": 0,
            }
            if args.empirical_rank:
                defaults["sum_dW_er"] = 0.0
            stats = per_layer_accum.setdefault(key, defaults)
            stats["sum_dW_fro_sq"] += metrics["frobenius_norm"] ** 2
            stats["sum_W_fro_sq"] += metrics["W_fro"] ** 2
            stats["sum_dW_sr"] += metrics["dW_stable_rank"]
            stats["max_dW_spec"] = max(stats["max_dW_spec"], metrics["diff_spectral_norm"])
            stats["max_W_spec"] = max(stats["max_W_spec"], metrics["W_spectral"])
            if args.empirical_rank:
                stats["sum_dW_er"] += metrics["dW_empirical_rank"]
            stats["count"] += 1

        del Wa, Wb

    # ---- Output ----
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 1) per_matrix.csv — one row per weight matrix
    per_matrix_fields = [
        "name", "layer", "component", "shape0", "shape1", "elements",
        "cosine_sim", "rel_frobenius", "frobenius_norm", "fro_norm_normalized",
        "W_fro", "Wb_fro", "diff_mean", "diff_std", "diff_abs_mean",
        "diff_spectral_norm", "W_spectral", "Wb_spectral", "dW_spectral_rel",
        "dW_stable_rank", "W_stable_rank", "Wb_stable_rank",
    ]
    if args.empirical_rank:
        per_matrix_fields += ["dW_empirical_rank", "W_empirical_rank"]
    rows.sort(key=lambda r: (r["layer"], r["component"]))
    per_matrix_csv = os.path.join(outdir, "per_matrix.csv")
    write_csv(per_matrix_csv, rows, per_matrix_fields)

    # 2) per_component.csv — aggregate stats per granular component across layers
    metric_keys = [
        "cosine_sim", "rel_frobenius", "frobenius_norm", "fro_norm_normalized",
        "diff_mean", "diff_std", "diff_abs_mean",
        "diff_spectral_norm", "dW_spectral_rel",
        "dW_stable_rank", "W_stable_rank",
    ]
    comp_stats = defaultdict(lambda: {k: [] for k in metric_keys})
    for r in rows:
        c = r["component"]
        for k in metric_keys:
            comp_stats[c][k].append(r[k])

    summary_rows = []
    for comp in _GRANULAR_COMPONENTS:
        if comp not in comp_stats:
            continue
        s = comp_stats[comp]
        comp_rows = [r for r in rows if r["component"] == comp]
        sr = {
            "component": comp,
            "n_layers": len(s["cosine_sim"]),
            "elements": comp_rows[0]["elements"] if comp_rows else 0,
        }
        for k in metric_keys:
            vals = s[k]
            sr[f"{k}_mean"] = float(np.mean(vals))
            sr[f"{k}_min"] = float(np.min(vals))
            sr[f"{k}_max"] = float(np.max(vals))
            sr[f"{k}_std"] = float(np.std(vals))
        summary_rows.append(sr)

    summary_fields = ["component", "n_layers", "elements"]
    for k in metric_keys:
        summary_fields.extend([f"{k}_mean", f"{k}_min", f"{k}_max", f"{k}_std"])
    write_csv(os.path.join(outdir, "per_component.csv"), summary_rows, summary_fields)

    # 3) per_layer.csv — aggregate stats per layer per granular component
    per_layer_granular = defaultdict(lambda: {
        "sum_dW_fro_sq": 0.0, "sum_W_fro_sq": 0.0,
        "sum_dW_sr": 0.0,
        "max_dW_spec": 0.0, "max_W_spec": 0.0,
        "count": 0,
    })
    if args.empirical_rank:
        for k in per_layer_granular.values():
            k["sum_dW_er"] = 0.0
    for r in rows:
        layer = r["layer"]
        comp = r["component"]
        key = (layer, comp)
        stats = per_layer_granular[key]
        if args.empirical_rank and "sum_dW_er" not in stats:
            stats["sum_dW_er"] = 0.0
        stats["sum_dW_fro_sq"] += r["frobenius_norm"] ** 2
        stats["sum_W_fro_sq"] += r["W_fro"] ** 2
        stats["sum_dW_sr"] += r["dW_stable_rank"]
        stats["max_dW_spec"] = max(stats["max_dW_spec"], r["diff_spectral_norm"])
        stats["max_W_spec"] = max(stats["max_W_spec"], r["W_spectral"])
        if args.empirical_rank:
            stats["sum_dW_er"] += r.get("dW_empirical_rank", 0.0)
        stats["count"] += 1

    granular_layer_rows = []
    for (layer, comp), stats in sorted(per_layer_granular.items(), key=lambda x: (x[0][0], x[0][1])):
        dW_fro = float(np.sqrt(stats["sum_dW_fro_sq"]))
        W_fro = float(np.sqrt(stats["sum_W_fro_sq"]))
        lr = {
            "layer": layer,
            "component": comp,
            "dW_fro_layer": dW_fro,
            "W_fro_layer": W_fro,
            "dW_fro_layer_rel": dW_fro / W_fro if W_fro > 0 else 0.0,
            "max_dW_spectral": stats["max_dW_spec"],
            "max_W_spectral": stats["max_W_spec"],
            "max_dW_spectral_rel": stats["max_dW_spec"] / stats["max_W_spec"] if stats["max_W_spec"] > 0 else 0.0,
            "mean_dW_stable_rank": stats["sum_dW_sr"] / max(stats["count"], 1),
            "count_mats": stats["count"],
        }
        if args.empirical_rank:
            lr["mean_dW_empirical_rank"] = stats["sum_dW_er"] / max(stats["count"], 1)
        granular_layer_rows.append(lr)

    per_layer_granular_fields = [
        "layer", "component",
        "dW_fro_layer", "W_fro_layer", "dW_fro_layer_rel",
        "max_dW_spectral", "max_W_spectral", "max_dW_spectral_rel",
        "mean_dW_stable_rank", "count_mats",
    ]
    if args.empirical_rank:
        per_layer_granular_fields.insert(-1, "mean_dW_empirical_rank")
    per_layer_csv = os.path.join(outdir, "per_layer.csv")
    write_csv(per_layer_csv, granular_layer_rows, per_layer_granular_fields)

    # 4) per_coarse_layer.csv — aggregate stats per layer per coarse group (attn/mlp)
    layer_rows = []
    for (layer, group), stats in sorted(per_layer_accum.items(), key=lambda x: (x[0][0], x[0][1])):
        dW_fro_layer = float(np.sqrt(stats["sum_dW_fro_sq"]))
        W_fro_layer = float(np.sqrt(stats["sum_W_fro_sq"]))
        max_dW_spec = stats["max_dW_spec"]
        max_W_spec = stats["max_W_spec"]
        lr = {
            "layer": layer,
            "group": group,
            "dW_fro_layer": dW_fro_layer,
            "W_fro_layer": W_fro_layer,
            "dW_fro_layer_rel": dW_fro_layer / W_fro_layer if W_fro_layer > 0 else 0.0,
            "max_dW_spectral": max_dW_spec,
            "max_W_spectral": max_W_spec,
            "max_dW_spectral_rel": max_dW_spec / max_W_spec if max_W_spec > 0 else 0.0,
            "mean_dW_stable_rank": stats["sum_dW_sr"] / max(stats["count"], 1),
            "count_mats": stats["count"],
        }
        if args.empirical_rank:
            lr["mean_dW_empirical_rank"] = stats["sum_dW_er"] / max(stats["count"], 1)
        layer_rows.append(lr)

    per_layer_fields = [
        "layer", "group",
        "dW_fro_layer", "W_fro_layer", "dW_fro_layer_rel",
        "max_dW_spectral", "max_W_spectral", "max_dW_spectral_rel",
        "mean_dW_stable_rank", "count_mats",
    ]
    if args.empirical_rank:
        per_layer_fields.insert(-1, "mean_dW_empirical_rank")
    per_coarse_layer_csv = os.path.join(outdir, "per_coarse_layer.csv")
    write_csv(per_coarse_layer_csv, layer_rows, per_layer_fields)

    # ---- Log to wandb ----
    log_csv_as_table(per_matrix_csv, key="per_matrix")
    log_csv_as_table(os.path.join(outdir, "per_component.csv"), key="per_component")
    log_csv_as_table(per_layer_csv, key="per_layer")
    log_csv_as_table(per_coarse_layer_csv, key="per_coarse_layer")

    # ---- Stdout summary ----
    print(f"\n{'='*70}")
    print(f"Results: {outdir}")
    print(f"{'='*70}")
    print(f"{'Component':<15} {'Cos Sim':>10} {'Rel Fro':>10} {'Stable Rk':>10} {'Layers':>7}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
    for sr in summary_rows:
        print(
            f"{sr['component']:<15} "
            f"{sr['cosine_sim_mean']:>10.6f} "
            f"{sr['rel_frobenius_mean']:>10.6f} "
            f"{sr['dW_stable_rank_mean']:>10.2f} "
            f"{sr['n_layers']:>7}"
        )
    print()

    # ---- Plots ----
    if args.plot_outdir:
        plot_weight_comparison(
            per_matrix_csv,
            args.plot_outdir,
            args.title,
            model_a=args.model_a,
            model_b=args.model_b,
        )
        log_plots(args.plot_outdir, "weight_plots")

    finish_wandb()


if __name__ == "__main__":
    main()
