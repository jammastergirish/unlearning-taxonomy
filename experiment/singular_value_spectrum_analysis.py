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
Singular Value Spectrum Analysis.

For a set of weight matrices (mlp_expand, mlp_contract, proj), plots the full
sorted singular value spectrum of:
  - W_a  (baseline model)
  - W_b  (target / unlearned model)
  - dW   (W_b - W_a, i.e. the update matrix)

All spectra are normalised by their own leading singular value so curves from
matrices of very different scales land on the same [0, 1] axis.

Representative layers (early/mid/late) are chosen automatically from however
many layers the model has.

Outputs:
  sv_spectrum/<component>_layer<N>.png   — per-matrix overlay plot
  sv_spectrum/dW_spectrum_<component>.png — dW spectra stacked across layers
  sv_spectrum/elbow_summary.csv          — elbow index per matrix
  sv_spectrum/sv_spectrum.png            — sentinel / overview figure
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    SmartLoader,
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    extract_layer,
    classify_granular,
    write_csv,
    init_wandb,
    infer_method_from_model_name,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_COMPONENTS = ["mlp_expand", "mlp_contract", "proj"]

_COMPONENT_COLORS = {
    "mlp_expand":   ("tab:blue",   "tab:cyan"),
    "mlp_contract": ("tab:orange", "tab:red"),
    "proj":         ("tab:green",  "tab:olive"),
}

_COMP_DISPLAY = {
    "mlp_expand":   "MLP Expand (up/gate)",
    "mlp_contract": "MLP Contract (down)",
    "proj":         "Attention Out-proj",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _pick_representative_layers(all_layers: List[int], n: int = 3) -> List[int]:
    """Return n evenly-spaced layers spanning early / mid / late."""
    layers = sorted(set(all_layers))
    if not layers:
        return []
    if len(layers) <= n:
        return layers
    # Always include first and last; fill the middle
    indices = [int(round(i * (len(layers) - 1) / (n - 1))) for i in range(n)]
    return [layers[i] for i in indices]


def _svdvals_normalized(W: torch.Tensor) -> np.ndarray:
    """Return normalised singular values (descending, divided by σ₁).

    Uses torch.linalg.svdvals which computes σ values only — much cheaper
    than the full SVD.
    """
    Wf = W.detach().float()
    try:
        s = torch.linalg.svdvals(Wf).cpu().numpy()
    except RuntimeError:
        # Very small matrices or degenerate case
        s = np.array([0.0])
    s = np.sort(s)[::-1]  # descending
    if s[0] > 0:
        s = s / s[0]
    return s


def _elbow_index(s: np.ndarray) -> int:
    """Locate the elbow using the maximum-distance (knee) method.

    Draws a line from the first to the last point and finds the index
    of the point furthest from that line.
    """
    n = len(s)
    if n < 3:
        return 0
    x = np.arange(n, dtype=float)
    # Start and end of the line
    p1 = np.array([0.0, s[0]])
    p2 = np.array([float(n - 1), s[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return 0
    line_unit = line_vec / line_len
    # Vector from p1 to each point
    pts = np.column_stack([x, s])
    vecs = pts - p1
    # Perpendicular distance = |cross product| / |line_unit| (2-D)
    cross = vecs[:, 0] * line_unit[1] - vecs[:, 1] * line_unit[0]
    return int(np.argmax(np.abs(cross)))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_overlay(
    name: str,
    s_a: np.ndarray,
    s_b: np.ndarray,
    s_dw: np.ndarray,
    elbow_a: int,
    elbow_b: int,
    elbow_dw: int,
    label_a: str,
    label_b: str,
    title: str,
    outpath: str,
) -> None:
    """Three-panel plot: W_a vs W_b overlaid, dW spectrum, and elbow zoom."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel 1: W_a vs W_b overlay ----
    ax = axes[0]
    ax.plot(s_a, color="tab:blue",   lw=1.5, label=label_a)
    ax.plot(s_b, color="tab:orange", lw=1.5, label=label_b)
    ax.axvline(elbow_a, color="tab:blue",   ls="--", lw=0.8, alpha=0.7, label=f"elbow A (k={elbow_a})")
    ax.axvline(elbow_b, color="tab:orange", ls="--", lw=0.8, alpha=0.7, label=f"elbow B (k={elbow_b})")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalised σ")
    ax.set_title("Spectrum: baseline vs unlearned")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel 2: dW spectrum ----
    ax = axes[1]
    ax.plot(s_dw, color="tab:red", lw=1.5, label="ΔW")
    ax.axvline(elbow_dw, color="tab:red", ls="--", lw=0.8, alpha=0.7, label=f"elbow ΔW (k={elbow_dw})")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalised σ")
    ax.set_title("Spectrum: update matrix ΔW")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel 3: log-scale view ----
    ax = axes[2]
    ax.semilogy(s_a + 1e-12, color="tab:blue",   lw=1.5, label=label_a)
    ax.semilogy(s_b + 1e-12, color="tab:orange", lw=1.5, label=label_b)
    ax.semilogy(s_dw + 1e-12, color="tab:red",   lw=1.5, ls=":", label="ΔW")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalised σ (log)")
    ax.set_title("Log-scale spectrum")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"{title}\n{name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def _plot_dw_all_layers(
    component: str,
    layer_spectra: Dict[int, np.ndarray],  # layer -> normalised dW spectrum
    elbows: Dict[int, int],
    title: str,
    outpath: str,
) -> None:
    """Overlay dW spectra for all representative layers of one component."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("plasma")
    layers = sorted(layer_spectra)
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for i, layer in enumerate(layers):
        s = layer_spectra[layer]
        elbow = elbows.get(layer, 0)
        label = f"layer {layer} (k={elbow})"
        ax.plot(s, color=colors[i], lw=1.5, label=label)
        ax.axvline(elbow, color=colors[i], ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalised σ of ΔW")
    ax.set_title(f"ΔW spectra by layer — {_COMP_DISPLAY.get(component, component)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def _plot_elbow_summary(
    elbow_rows: List[Dict],
    components: List[str],
    label_a: str,
    label_b: str,
    title: str,
    outpath: str,
) -> None:
    """Bar chart of elbow indices per layer per component (A, B, dW side-by-side)."""
    if not elbow_rows:
        return

    import pandas as pd
    df = pd.DataFrame(elbow_rows)

    n_comp = len(components)
    fig, axes = plt.subplots(1, n_comp, figsize=(6 * n_comp, 5), squeeze=False)

    for ci, comp in enumerate(components):
        ax = axes[0, ci]
        sub = df[df["component"] == comp].sort_values("layer")
        if sub.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub["elbow_a"], width=w, label=label_a,   color="tab:blue",   alpha=0.8)
        ax.bar(x,     sub["elbow_b"], width=w, label=label_b,   color="tab:orange", alpha=0.8)
        ax.bar(x + w, sub["elbow_dw"], width=w, label="ΔW",     color="tab:red",    alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{r}" for r in sub["layer"]], fontsize=9)
        ax.set_ylabel("Elbow index (effective rank)")
        ax.set_title(_COMP_DISPLAY.get(comp, comp))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{title}\nElbow indices (effective rank at drop-off)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# W&B native data logging
# ---------------------------------------------------------------------------

def _log_wandb_spectrum(
    key: str,
    s_a: np.ndarray,
    s_b: np.ndarray,
    s_dw: np.ndarray,
    label_a: str,
    label_b: str,
) -> None:
    """Upload one spectrum triple as a W&B line_series chart (interactive).

    W&B's line_series takes parallel lists of xs and ys per series. We
    truncate all three spectra to a common length so the x-axis is shared.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        n = min(len(s_a), len(s_b), len(s_dw))
        xs = list(range(n))
        wandb.log({
            key: wandb.plot.line_series(
                xs=xs,
                ys=[s_a[:n].tolist(), s_b[:n].tolist(), s_dw[:n].tolist()],
                keys=[label_a, label_b, "ΔW"],
                title=key,
                xname="Singular value index",
            )
        })
    except Exception:
        pass


def _log_wandb_elbow_table(elbow_rows: list[dict], label_a: str, label_b: str) -> None:
    """Upload elbow summary as a wandb.Table so W&B can render bar charts.

    Columns: component, layer, elbow_a, elbow_b, elbow_dw, elbow_shift.
    Each row is one (component, layer) pair.
    """
    try:
        import wandb
        if wandb.run is None or not elbow_rows:
            return
        cols = ["component", "layer", f"elbow_{label_a}", f"elbow_{label_b}", "elbow_dW", "elbow_shift"]
        table = wandb.Table(columns=cols)
        for r in elbow_rows:
            table.add_data(
                r["component"], r["layer"],
                r["elbow_a"], r["elbow_b"], r["elbow_dw"], r["elbow_shift"],
            )
        wandb.log({"sv_spectrum/elbow_summary": table})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    loader_a: SmartLoader,
    loader_b: SmartLoader,
    linear_names: List[str],
    device: str,
    dtype: torch.dtype,
    target_components: List[str],
    n_layers: int,
    outdir: str,
    label_a: str,
    label_b: str,
    title: str,
) -> List[Dict]:
    """Run the full spectrum analysis. Returns elbow_rows for CSV export."""
    os.makedirs(outdir, exist_ok=True)

    # Group names by component
    by_comp: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for name in linear_names:
        if not name.endswith(".weight"):
            continue
        comp = classify_granular(name)
        if comp not in target_components:
            continue
        layer = extract_layer(name)
        if layer is None:
            continue
        by_comp[comp].append((layer, name))

    elbow_rows: List[Dict] = []
    written_pngs: List[str] = []

    for comp in target_components:
        entries = sorted(by_comp.get(comp, []), key=lambda x: x[0])
        if not entries:
            print(f"  [sv_spectrum] No matrices found for component '{comp}' — skipping")
            continue

        all_layers = [layer for layer, _ in entries]
        rep_layers = _pick_representative_layers(all_layers, n=n_layers)
        print(f"  [sv_spectrum] {comp}: {len(entries)} matrices, "
              f"representative layers: {rep_layers}")

        # For the dW-all-layers overlay plot
        dw_layer_spectra: Dict[int, np.ndarray] = {}
        dw_elbows: Dict[int, int] = {}

        for layer, name in tqdm(entries, desc=f"  {comp}", unit="matrix", leave=False):
            if layer not in rep_layers:
                continue

            Wa = loader_a.get_param(name, device, dtype)
            if Wa is None or Wa.ndim != 2:
                continue
            Wb = loader_b.get_param(name, device, dtype)
            if Wb is None or Wa.shape != Wb.shape:
                del Wa
                continue

            dW = Wb.float() - Wa.float()

            s_a  = _svdvals_normalized(Wa)
            s_b  = _svdvals_normalized(Wb)
            s_dw = _svdvals_normalized(dW)

            elbow_a  = _elbow_index(s_a)
            elbow_b  = _elbow_index(s_b)
            elbow_dw = _elbow_index(s_dw)

            dw_layer_spectra[layer] = s_dw
            dw_elbows[layer] = elbow_dw

            # Log spectra as native W&B line charts
            wandb_key = f"sv_spectrum/{comp}_layer{layer}"
            _log_wandb_spectrum(wandb_key, s_a, s_b, s_dw, label_a, label_b)

            # Per-matrix overlay plot
            plot_path = os.path.join(outdir, f"{comp}_layer{layer}.png")
            _plot_overlay(
                name=name,
                s_a=s_a, s_b=s_b, s_dw=s_dw,
                elbow_a=elbow_a, elbow_b=elbow_b, elbow_dw=elbow_dw,
                label_a=label_a, label_b=label_b,
                title=title,
                outpath=plot_path,
            )
            written_pngs.append(plot_path)

            elbow_rows.append({
                "component": comp,
                "layer": layer,
                "name": name,
                "shape": f"{Wa.shape[0]}×{Wa.shape[1]}",
                "elbow_a": elbow_a,
                "elbow_b": elbow_b,
                "elbow_dw": elbow_dw,
                "elbow_shift": elbow_b - elbow_a,
            })

            del Wa, Wb, dW

        # dW overlay across representative layers
        if dw_layer_spectra:
            dw_plot_path = os.path.join(outdir, f"dW_spectrum_{comp}.png")
            _plot_dw_all_layers(
                component=comp,
                layer_spectra=dw_layer_spectra,
                elbows=dw_elbows,
                title=title,
                outpath=dw_plot_path,
            )
            written_pngs.append(dw_plot_path)

    # Summary elbow bar chart (also acts as overview / sentinel figure)
    elbow_summary_path = os.path.join(outdir, "sv_spectrum.png")
    _plot_elbow_summary(
        elbow_rows=elbow_rows,
        components=[c for c in target_components if any(r["component"] == c for r in elbow_rows)],
        label_a=label_a,
        label_b=label_b,
        title=title,
        outpath=elbow_summary_path,
    )
    written_pngs.append(elbow_summary_path)

    log_plots(outdir, "sv_spectrum", files=written_pngs)
    _log_wandb_elbow_table(elbow_rows, label_a, label_b)
    return elbow_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot the full singular value spectrum for selected weight matrices "
            "across early/mid/late transformer layers, overlaying baseline vs. "
            "unlearned models, plus the dW update spectrum."
        )
    )
    ap.add_argument("--model-a", required=True, help="Baseline model path or HF repo ID")
    ap.add_argument("--model-b", required=True, help="Target / unlearned model path or HF repo ID")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype",  default="fp32",
                    help="Weight dtype for SVD (default fp32; fp16/bf16 work but fp32 is more stable)")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: auto-derived from model names)")
    ap.add_argument("--n-layers", type=int, default=3,
                    help="Number of representative layers to sample (default: 3 = early/mid/late)")
    ap.add_argument("--components", nargs="+", default=_TARGET_COMPONENTS,
                    choices=_TARGET_COMPONENTS,
                    help="Weight components to analyse (default: all three)")
    ap.add_argument("--title", default=None, help="Plot title prefix")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="sv_spectrum")

    method = infer_method_from_model_name(args.model_b)
    init_wandb("sv_spectrum", args, method=method)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    # SVD is numerically sensitive — prefer float32 regardless of device
    dtype = torch.float32 if args.dtype in ("auto", "fp32") else resolve_dtype(args.dtype, device)

    label_a = args.model_a.split("/")[-1]
    label_b = args.model_b.split("/")[-1]
    title = args.title or f"{label_a} → {label_b}"

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
    linear_names = sorted(names_a.intersection(names_b))

    elbow_rows = run_analysis(
        loader_a=loader_a,
        loader_b=loader_b,
        linear_names=linear_names,
        device=device,
        dtype=dtype,
        target_components=args.components,
        n_layers=args.n_layers,
        outdir=args.outdir,
        label_a=label_a,
        label_b=label_b,
        title=title,
    )

    # Write elbow summary CSV
    if elbow_rows:
        csv_path = os.path.join(args.outdir, "elbow_summary.csv")
        write_csv(csv_path, elbow_rows, [
            "component", "layer", "name", "shape",
            "elbow_a", "elbow_b", "elbow_dw", "elbow_shift",
        ])
        print(f"\n[sv_spectrum] Elbow summary saved to {csv_path}")

        print(f"\n{'Component':<15} {'Layer':>6} {'Elbow A':>9} {'Elbow B':>9} "
              f"{'Elbow dW':>10} {'Shift (B-A)':>12}")
        print("-" * 65)
        for r in sorted(elbow_rows, key=lambda x: (x["component"], x["layer"])):
            print(f"{r['component']:<15} {r['layer']:>6} {r['elbow_a']:>9} "
                  f"{r['elbow_b']:>9} {r['elbow_dw']:>10} {r['elbow_shift']:>+12}")

    # Write a JSON summary too (useful for downstream aggregation)
    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "components": args.components,
        "n_representative_layers": args.n_layers,
        "n_matrices_analysed": len(elbow_rows),
    }
    with open(os.path.join(args.outdir, "sv_spectrum_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[sv_spectrum] ✓ All outputs saved to {args.outdir}/")
    finish_wandb()


if __name__ == "__main__":
    main()
