#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
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
Row-space projection analysis.

Measures how pre-MLP activations from forget vs retain datasets project
onto the row space of each MLP weight update ΔW.  If forget activations
project *more* onto the row space, the update is selectively reshaping
the forward-pass directions that *carry* forget knowledge.
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    extract_layer,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
    SmartLoader,
)


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Hook to capture pre-MLP activations."""

    def __init__(self):
        self.activations: List[torch.Tensor] = []
        self.hooks: list = []

    def capture_hook(self, module, input, output):
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        self.activations.append(activation.detach().cpu())

    def register_hooks(self, model, layer_indices: List[int]):
        """Register hooks on specified MLP layers."""
        self.clear()
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for idx in layer_indices:
                if idx < len(model.model.layers):
                    mlp = model.model.layers[idx].mlp
                    hook = mlp.register_forward_hook(self.capture_hook)
                    self.hooks.append(hook)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for idx in layer_indices:
                if idx < len(model.transformer.h):
                    mlp = model.transformer.h[idx].mlp
                    hook = mlp.register_forward_hook(self.capture_hook)
                    self.hooks.append(hook)
        elif hasattr(model, "layers"):
            for idx in layer_indices:
                if idx < len(model.layers) and hasattr(model.layers[idx], "mlp"):
                    mlp = model.layers[idx].mlp
                    hook = mlp.register_forward_hook(self.capture_hook)
                    self.hooks.append(hook)

    def clear(self):
        """Remove hooks and clear activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = []


# ---------------------------------------------------------------------------
# Weight update extraction
# ---------------------------------------------------------------------------

def get_mlp_weight_updates(
    loader_a: SmartLoader,
    loader_b: SmartLoader,
    layer_idx: int,
    device: str,
    dtype,
) -> Dict[str, np.ndarray]:
    """Get MLP weight updates for a specific layer.

    Returns a dict with keys ``"encoder"`` and/or ``"decoder"`` mapping
    to numpy weight-update arrays.
    """
    updates: Dict[str, np.ndarray] = {}

    possible_names = [
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight",
        f"transformer.h.{layer_idx}.mlp.c_fc.weight",
        f"transformer.h.{layer_idx}.mlp.c_proj.weight",
        f"layers.{layer_idx}.mlp.fc1.weight",
        f"layers.{layer_idx}.mlp.fc2.weight",
    ]

    for name in possible_names:
        weight_a = loader_a.get_param(name, device, dtype)
        if weight_a is None or weight_a.ndim != 2:
            continue

        weight_b = loader_b.get_param(name, device, dtype)
        if weight_b is None or weight_b.shape != weight_a.shape:
            continue

        weight_update = (weight_b - weight_a).cpu().float().numpy()

        if any(k in name for k in ("gate", "up", "fc1", "c_fc")):
            updates["encoder"] = weight_update
        else:
            updates["decoder"] = weight_update

    return updates


# ---------------------------------------------------------------------------
# Core analysis (testable, no I/O)
# ---------------------------------------------------------------------------

def compute_row_space_projection(
    activations: List[np.ndarray],
    weight_update: np.ndarray,
    top_k: int = 20,
) -> Optional[Dict]:
    """Compute how activations project onto the row space of *weight_update*.

    Args:
        activations: List of activation arrays with shape (..., hidden_dim).
        weight_update: Weight update matrix (out_dim, in_dim).
        top_k: Number of top singular vectors to use.

    Returns a dict with projection_norm, original_norm, projection_ratio,
    variance_ratio, top_alignments, and num_samples; or None on empty input.
    """
    if len(activations) == 0 or weight_update is None:
        return None

    flattened_activations = np.vstack([a.reshape(-1, a.shape[-1]) for a in activations])

    # SVD of weight update transposed → row-space basis in columns of U
    U, _singular_values, _Vt = np.linalg.svd(weight_update.T, full_matrices=False)
    k = min(top_k, U.shape[1])
    U_top = U[:, :k]

    projected = flattened_activations @ U_top
    projection_norm = float(np.linalg.norm(projected, axis=1).mean())
    original_norm = float(np.linalg.norm(flattened_activations, axis=1).mean())
    projection_ratio = projection_norm / (original_norm + 1e-10)

    projection_variance = float(np.var(projected))
    original_variance = float(np.var(flattened_activations))
    variance_ratio = projection_variance / (original_variance + 1e-10)

    # Alignment with individual top singular vectors
    top_alignments = [
        float(np.abs(flattened_activations @ U[:, i]).mean())
        for i in range(min(5, k))
    ]

    return {
        "projection_norm": projection_norm,
        "original_norm": original_norm,
        "projection_ratio": projection_ratio,
        "variance_ratio": variance_ratio,
        "top_alignments": top_alignments,
        "num_samples": len(flattened_activations),
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_row_space_projections(
    layer_results: List[Dict],
    per_weight_results: List[Dict],
    outdir: str,
    title: Optional[str] = None,
    model_a: str = "Model A",
    model_b: str = "Model B",
) -> None:
    """Create the 2×2 panel of row-space projection plots."""
    layers_plot = [r["layer"] for r in layer_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0) Projection ratios by layer
    axis = axes[0, 0]
    axis.plot(layers_plot, [r["avg_forget_proj"] for r in layer_results], "o-",
              label="Forget", color="red")
    axis.plot(layers_plot, [r["avg_retain_proj"] for r in layer_results], "s-",
              label="Retain", color="blue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Projection Ratio")
    axis.set_title("Activation Projection onto Update Row Space")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,1) Projection difference (forget − retain)
    axis = axes[0, 1]
    diffs = [r["avg_diff"] for r in layer_results]
    bar_colors = ["red" if d > 0 else "blue" for d in diffs]
    axis.bar(layers_plot, diffs, color=bar_colors, alpha=0.6)
    axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Forget − Retain Projection")
    axis.set_title("Differential Projection (Positive = Forget More Affected)")
    axis.grid(alpha=0.3)

    # (1,0) Scatter: forget vs retain projection
    axis = axes[1, 0]
    forget_all = [r["forget_proj_ratio"] for r in per_weight_results]
    retain_all = [r["retain_proj_ratio"] for r in per_weight_results]
    axis.scatter(retain_all, forget_all, alpha=0.6, s=50)
    max_val = max(max(retain_all + forget_all), 0.01)
    axis.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal projection")
    axis.set_xlabel("Retain Projection Ratio")
    axis.set_ylabel("Forget Projection Ratio")
    axis.set_title("Forget vs Retain Projections")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,1) Summary text
    axis = axes[1, 1]
    axis.axis("off")

    avg_forget = float(np.mean(forget_all))
    avg_retain = float(np.mean(retain_all))
    forget_stronger_count = sum(1 for r in per_weight_results if r.get("forget_stronger", False))

    summary_text = (
        f"Row Space Projection Summary ({model_a.split('/')[-1]} → {model_b.split('/')[-1]}):\n\n"
        f"Average Projections:\n"
        f"- Forget: {avg_forget:.3f}\n"
        f"- Retain: {avg_retain:.3f}\n"
        f"- Ratio: {avg_forget / (avg_retain + 1e-10):.2f}x\n\n"
        f"Layer Analysis:\n"
        f"- Forget stronger in {forget_stronger_count}/{len(per_weight_results)} cases\n"
        f"- Max difference at layer {layers_plot[int(np.argmax(diffs))]}\n"
        f"- Avg difference: {float(np.mean(diffs)):.3f}\n\n"
        f"Interpretation:\n"
        f"{'✓ Forget more affected by updates' if avg_forget > avg_retain * 1.2 else '✗ Similar affect on both'}\n"
        f"{'✓ Selective modification' if float(np.mean(diffs)) > 0.05 else '✗ Non-selective updates'}"
    )
    axis.text(0.1, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

    plt.suptitle(title or f"Row Space Projection Analysis\n{model_a.split('/')[-1]} → {model_b.split('/')[-1]}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "row_space_projections.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze how activations project onto the row space of MLP weight updates.",
    )
    parser.add_argument("--model-a", required=True, help="Baseline model")
    parser.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    parser.add_argument("--forget-text", default="data/forget.txt")
    parser.add_argument("--retain-text", default="data/retain.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max texts per split to process (default: 200)")
    parser.add_argument("--layers-to-analyze", type=str, default=None,
                        help="Comma-separated layer indices (default: every 4th layer)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="row_space_projection")

    init_wandb("row_space_projection", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[row_space_projection] Loading forget/retain texts...")
    with open(args.forget_text, "r") as fh:
        forget_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    with open(args.retain_text, "r") as fh:
        retain_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]

    print(
        f"[row_space_projection] Loaded {len(forget_texts)} forget texts, "
        f"{len(retain_texts)} retain texts (max-samples={args.max_samples})"
    )

    # Load tokenizers and models
    print(f"Loading Base Model: {args.model_a}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map_kwargs = {"device_map": "auto"} if device == "cuda" else {}

    model_a = AutoModelForCausalLM.from_pretrained(args.model_a, torch_dtype=dtype, **device_map_kwargs)
    if not device_map_kwargs:
        model_a.to(device)
    model_a.eval()

    loader_a = SmartLoader(args.model_a)
    loader_b = SmartLoader(args.model_b)

    # Determine layers
    if hasattr(model_a, "gpt_neox") and hasattr(model_a.gpt_neox, "layers"):
        num_layers = len(model_a.gpt_neox.layers)
    elif hasattr(model_a, "model") and hasattr(model_a.model, "layers"):
        num_layers = len(model_a.model.layers)
    elif hasattr(model_a, "transformer") and hasattr(model_a.transformer, "h"):
        num_layers = len(model_a.transformer.h)
    else:
        print("Warning: Could not determine number of layers")
        num_layers = 32

    if args.layers_to_analyze:
        layers_to_analyze = [int(x) for x in args.layers_to_analyze.split(",")]
    else:
        layers_to_analyze = list(range(0, num_layers, 4))

    print(f"[row_space_projection] Will analyze layers: {layers_to_analyze}")

    per_weight_results: List[Dict] = []
    os.makedirs(args.outdir, exist_ok=True)

    capturer = ActivationCapture()

    for layer_idx in tqdm(layers_to_analyze, desc="Projecting onto row space", unit="layer"):
        print(f"\nLayer {layer_idx}:")

        updates = get_mlp_weight_updates(loader_a, loader_b, layer_idx, device, dtype)
        if not updates:
            print(f"  No MLP weights found for layer {layer_idx}")
            continue

        # Collect forget activations
        capturer.register_hooks(model_a, [layer_idx])
        print("  Collecting forget activations...")
        capturer.clear()
        capturer.activations = []

        for batch_start in range(0, len(forget_texts), args.batch_size):
            batch_texts = forget_texts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                _ = model_a(**inputs)

        forget_activations = capturer.activations.copy()

        # Collect retain activations
        capturer.clear()
        capturer.activations = []
        print("  Collecting retain activations...")

        for batch_start in range(0, len(retain_texts), args.batch_size):
            batch_texts = retain_texts[batch_start : batch_start + args.batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                _ = model_a(**inputs)

        retain_activations = capturer.activations.copy()

        # Compute projections for encoder weights
        if "encoder" in updates:
            print("  Computing encoder projections...")
            forget_projection = compute_row_space_projection(forget_activations, updates["encoder"])
            retain_projection = compute_row_space_projection(retain_activations, updates["encoder"])

            if forget_projection and retain_projection:
                result = {
                    "layer": layer_idx,
                    "weight_type": "encoder",
                    "forget_proj_ratio": forget_projection["projection_ratio"],
                    "retain_proj_ratio": retain_projection["projection_ratio"],
                    "forget_proj_norm": forget_projection["projection_norm"],
                    "retain_proj_norm": retain_projection["projection_norm"],
                    "projection_diff": forget_projection["projection_ratio"] - retain_projection["projection_ratio"],
                    "forget_stronger": forget_projection["projection_ratio"] > retain_projection["projection_ratio"],
                }
                per_weight_results.append(result)

                np.savez_compressed(
                    os.path.join(args.outdir, f"projections_layer_{layer_idx}_encoder.npz"),
                    forget_alignments=forget_projection["top_alignments"],
                    retain_alignments=retain_projection["top_alignments"],
                    dW_shape=updates["encoder"].shape,
                )

        capturer.clear()

    # Save results
    if per_weight_results:
        write_csv(
            os.path.join(args.outdir, "row_space_projections.csv"),
            per_weight_results,
            ["layer", "weight_type", "forget_proj_ratio", "retain_proj_ratio",
             "forget_proj_norm", "retain_proj_norm", "projection_diff", "forget_stronger"],
        )

        # Layer-wise summary
        layer_summary_map: Dict[int, Dict[str, list]] = {}
        for result in per_weight_results:
            layer = result["layer"]
            if layer not in layer_summary_map:
                layer_summary_map[layer] = {"forget_ratios": [], "retain_ratios": [], "diffs": []}
            layer_summary_map[layer]["forget_ratios"].append(result["forget_proj_ratio"])
            layer_summary_map[layer]["retain_ratios"].append(result["retain_proj_ratio"])
            layer_summary_map[layer]["diffs"].append(result["projection_diff"])

        layer_results: List[Dict] = []
        for layer in sorted(layer_summary_map.keys()):
            stats = layer_summary_map[layer]
            layer_results.append({
                "layer": layer,
                "avg_forget_proj": float(np.mean(stats["forget_ratios"])),
                "avg_retain_proj": float(np.mean(stats["retain_ratios"])),
                "avg_diff": float(np.mean(stats["diffs"])),
            })

        write_csv(
            os.path.join(args.outdir, "layer_projection_summary.csv"),
            layer_results,
            ["layer", "avg_forget_proj", "avg_retain_proj", "avg_diff"],
        )

        # Plots
        plot_row_space_projections(
            layer_results, per_weight_results, args.outdir, title=args.title,
            model_a=args.model_a, model_b=args.model_b,
        )

        # Summary JSON
        avg_forget_proj = float(np.mean([r["forget_proj_ratio"] for r in per_weight_results]))
        avg_retain_proj = float(np.mean([r["retain_proj_ratio"] for r in per_weight_results]))

        summary = {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "avg_forget_projection": avg_forget_proj,
            "avg_retain_projection": avg_retain_proj,
            "projection_ratio": float(avg_forget_proj / (avg_retain_proj + 1e-10)),
            "forget_more_affected": bool(avg_forget_proj > avg_retain_proj * 1.2),
            "layers_analyzed": len(layers_to_analyze),
        }

        with open(os.path.join(args.outdir, "row_space_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        print(f"\n[row_space_projection] ✓ Results saved to {args.outdir}")
        print(f"[row_space_projection] Forget projection: {avg_forget_proj:.3f}")
        print(f"[row_space_projection] Retain projection: {avg_retain_proj:.3f}")
        print(f"[row_space_projection] Selectivity: {summary['projection_ratio']:.2f}x more aligned with forget data")
        log_csv_as_table(os.path.join(args.outdir, "row_space_projections.csv"), "row_space_projections")
        log_csv_as_table(os.path.join(args.outdir, "layer_projection_summary.csv"), "layer_projection_summary")
        log_plots(args.outdir, "row_space")
        finish_wandb()


if __name__ == "__main__":
    main()