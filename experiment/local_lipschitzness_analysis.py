#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Local Lipschitzness analysis of models on forget/retain data.

Estimates how "smooth" each model's mapping is locally by:
  1. Finite-difference Lipschitz estimation:
       L ≈ max ‖f(x+δ) − f(x)‖ / ‖δ‖
  2. Input gradient norms (direct smoothness).
  3. Output variance under random embedding-space perturbations.

Lower Lipschitz constant → smoother function in that region.
If unlearning *selectively* makes the model smoother on forget data
while preserving retain smoothness, the method is acting surgically.
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
    init_wandb,
    infer_method_from_model_name,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Core analysis helpers (testable where practical)
# ---------------------------------------------------------------------------

def _get_embeddings(model, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """Return token embeddings from the model's embedding layer."""
    if hasattr(model, "model"):
        return model.model.embed_tokens(input_ids)
    elif hasattr(model, "transformer"):
        return model.transformer.wte(input_ids)
    return None


def estimate_local_lipschitz(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    dtype,
    epsilon: float = 0.01,
    num_perturbations: int = 5,
    max_length: int = 512,
) -> List[float]:
    """Estimate local Lipschitz constant via finite differences.

    For each text, perturbs the embedding and measures
    max ‖f(x+δ) − f(x)‖ / ‖δ‖ over *num_perturbations* random directions.
    """
    model.eval()
    lipschitz_estimates: List[float] = []

    for text in tqdm(texts[:100], desc="Estimating Lipschitz constants", unit="text", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        with torch.no_grad():
            embeddings = _get_embeddings(model, inputs.input_ids)
            if embeddings is None:
                continue

            base_outputs = model(inputs_embeds=embeddings, output_hidden_states=True)
            hidden_base = base_outputs.hidden_states[-1]
            logits_base = base_outputs.logits

        max_ratio = 0.0
        for _ in range(num_perturbations):
            perturbation = torch.randn_like(embeddings) * epsilon
            perturbation_norm = perturbation.norm().item()

            with torch.no_grad():
                perturbed_outputs = model(
                    inputs_embeds=embeddings + perturbation, output_hidden_states=True,
                )
                hidden_diff = (perturbed_outputs.hidden_states[-1] - hidden_base).norm().item()
                logits_diff = (perturbed_outputs.logits - logits_base).norm().item()

            if perturbation_norm > 0:
                ratio_hidden = hidden_diff / perturbation_norm
                ratio_logits = logits_diff / perturbation_norm
                max_ratio = max(max_ratio, ratio_hidden, ratio_logits)

        lipschitz_estimates.append(max_ratio)

    return lipschitz_estimates


def compute_gradient_norms(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    dtype,
    max_length: int = 512,
) -> List[float]:
    """Compute gradient norms ∂loss/∂embeddings as a direct smoothness measure."""
    model.eval()
    gradient_norms: List[float] = []

    for text in tqdm(texts[:50], desc="Computing gradient norms", unit="text", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        embeddings = _get_embeddings(model, inputs.input_ids)
        if embeddings is None:
            continue

        embeddings = embeddings.detach().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings)
        loss = outputs.logits.mean()
        loss.backward()

        if embeddings.grad is not None:
            gradient_norms.append(float(embeddings.grad.norm().item()))

        model.zero_grad()

    return gradient_norms


def analyze_output_variance(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    dtype,
    num_perturbations: int = 10,
    epsilon: float = 0.01,
    max_length: int = 512,
) -> List[float]:
    """Measure output-logit variance under random embedding perturbations.

    High variance ⇒ less smooth / less stable.
    """
    model.eval()
    variances: List[float] = []

    for text in tqdm(texts[:50], desc="Measuring output variance", unit="text", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        with torch.no_grad():
            embeddings = _get_embeddings(model, inputs.input_ids)
            if embeddings is None:
                continue

            logits_list = []
            for _ in range(num_perturbations):
                perturbation = torch.randn_like(embeddings) * epsilon
                outputs = model(inputs_embeds=embeddings + perturbation)
                logits_list.append(outputs.logits.cpu().float().numpy())

        logits_array = np.stack(logits_list)
        variance = float(np.var(logits_array, axis=0).mean())
        variances.append(variance)

    return variances


# ---------------------------------------------------------------------------
# Summary row builder (testable)
# ---------------------------------------------------------------------------

def build_summary_rows(
    forget_lipschitz_a: List[float],
    retain_lipschitz_a: List[float],
    forget_gradient_a: List[float],
    retain_gradient_a: List[float],
    forget_variance_a: List[float],
    retain_variance_a: List[float],
    forget_lipschitz_b: List[float],
    retain_lipschitz_b: List[float],
    forget_gradient_b: List[float],
    retain_gradient_b: List[float],
    forget_variance_b: List[float],
    retain_variance_b: List[float],
) -> List[Dict]:
    """Build the four summary rows (A-forget, A-retain, B-forget, B-retain)."""
    def _row(model_label, data_label, lip, grad, var):
        return {
            "model": model_label,
            "data": data_label,
            "avg_lipschitz": float(np.mean(lip)) if lip else 0.0,
            "std_lipschitz": float(np.std(lip)) if lip else 0.0,
            "avg_gradient_norm": float(np.mean(grad)) if grad else 0.0,
            "std_gradient_norm": float(np.std(grad)) if grad else 0.0,
            "avg_output_variance": float(np.mean(var)) if var else 0.0,
            "std_output_variance": float(np.std(var)) if var else 0.0,
        }

    return [
        _row("A", "forget", forget_lipschitz_a, forget_gradient_a, forget_variance_a),
        _row("A", "retain", retain_lipschitz_a, retain_gradient_a, retain_variance_a),
        _row("B", "forget", forget_lipschitz_b, forget_gradient_b, forget_variance_b),
        _row("B", "retain", retain_lipschitz_b, retain_gradient_b, retain_variance_b),
    ]


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

_LIPSCHITZ_FIELDNAMES = [
    "model", "data",
    "avg_lipschitz", "std_lipschitz",
    "avg_gradient_norm", "std_gradient_norm",
    "avg_output_variance", "std_output_variance",
]


def plot_lipschitzness_analysis(
    forget_lipschitz_a: List[float],
    retain_lipschitz_a: List[float],
    forget_lipschitz_b: List[float],
    retain_lipschitz_b: List[float],
    forget_gradient_a: List[float],
    retain_gradient_a: List[float],
    forget_gradient_b: List[float],
    retain_gradient_b: List[float],
    forget_variance_a: List[float],
    retain_variance_a: List[float],
    forget_variance_b: List[float],
    retain_variance_b: List[float],
    outdir: str,
    title: Optional[str] = None,
    model_a: str = "Model A (Baseline)",
    model_b: str = "Model B (Unlearned)",
) -> None:
    """Create the 2×3 panel of Lipschitzness analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    x_positions = np.array([0, 1, 3, 4])
    bar_colors = ["red", "blue", "darkred", "darkblue"]
    bar_labels = [f"Forget ({model_a.split('/')[-1]})", f"Retain ({model_a.split('/')[-1]})",
                  f"Forget ({model_b.split('/')[-1]})", f"Retain ({model_b.split('/')[-1]})"]

    # (0,0) Lipschitz constants
    axis = axes[0, 0]
    means = [np.mean(forget_lipschitz_a), np.mean(retain_lipschitz_a),
             np.mean(forget_lipschitz_b), np.mean(retain_lipschitz_b)]
    stds = [np.std(forget_lipschitz_a), np.std(retain_lipschitz_a),
            np.std(forget_lipschitz_b), np.std(retain_lipschitz_b)]
    axis.bar(x_positions, means, yerr=stds, color=bar_colors, alpha=0.6, capsize=5)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(bar_labels, rotation=45, ha="right")
    axis.set_ylabel("Lipschitz Constant")
    axis.set_title("Local Lipschitzness (Lower = Smoother)")
    axis.grid(alpha=0.3, axis="y")

    # (0,1) Gradient norms
    axis = axes[0, 1]
    means_grad = [np.mean(forget_gradient_a), np.mean(retain_gradient_a),
                  np.mean(forget_gradient_b), np.mean(retain_gradient_b)]
    stds_grad = [np.std(forget_gradient_a), np.std(retain_gradient_a),
                 np.std(forget_gradient_b), np.std(retain_gradient_b)]
    axis.bar(x_positions, means_grad, yerr=stds_grad, color=bar_colors, alpha=0.6, capsize=5)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(bar_labels, rotation=45, ha="right")
    axis.set_ylabel("Gradient Norm")
    axis.set_title("Input Gradient Norms")
    axis.grid(alpha=0.3, axis="y")

    # (0,2) Output variance
    axis = axes[0, 2]
    means_var = [np.mean(forget_variance_a), np.mean(retain_variance_a),
                 np.mean(forget_variance_b), np.mean(retain_variance_b)]
    stds_var = [np.std(forget_variance_a), np.std(retain_variance_a),
                np.std(forget_variance_b), np.std(retain_variance_b)]
    axis.bar(x_positions, means_var, yerr=stds_var, color=bar_colors, alpha=0.6, capsize=5)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(bar_labels, rotation=45, ha="right")
    axis.set_ylabel("Output Variance")
    axis.set_title("Variance Under Perturbation")
    axis.grid(alpha=0.3, axis="y")

    # (1,0) Change in Lipschitzness (B − A)
    axis = axes[1, 0]
    forget_lip_change = float(np.mean(forget_lipschitz_b) - np.mean(forget_lipschitz_a))
    retain_lip_change = float(np.mean(retain_lipschitz_b) - np.mean(retain_lipschitz_a))
    axis.bar([0, 1], [forget_lip_change, retain_lip_change], color=["red", "blue"], alpha=0.6)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Forget", "Retain"])
    axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axis.set_ylabel("Δ Lipschitz (B − A)")
    axis.set_title("Change in Smoothness")
    axis.grid(alpha=0.3, axis="y")

    # (1,1) Distribution comparison
    axis = axes[1, 1]
    axis.hist(forget_lipschitz_a, bins=20, alpha=0.3, label=f"Forget ({model_a.split('/')[-1]})", color="red")
    axis.hist(forget_lipschitz_b, bins=20, alpha=0.3, label=f"Forget ({model_b.split('/')[-1]})", color="darkred")
    axis.hist(retain_lipschitz_a, bins=20, alpha=0.3, label=f"Retain ({model_a.split('/')[-1]})", color="blue")
    axis.hist(retain_lipschitz_b, bins=20, alpha=0.3, label=f"Retain ({model_b.split('/')[-1]})", color="darkblue")
    axis.set_xlabel("Lipschitz Constant")
    axis.set_ylabel("Frequency")
    axis.set_title("Distribution of Local Lipschitz Constants")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,2) Summary text
    axis = axes[1, 2]
    axis.axis("off")

    forget_lip_ratio = float(np.mean(forget_lipschitz_b)) / (float(np.mean(forget_lipschitz_a)) + 1e-10)
    retain_lip_ratio = float(np.mean(retain_lipschitz_b)) / (float(np.mean(retain_lipschitz_a)) + 1e-10)

    forget_trend = "smoother" if forget_lip_ratio < 0.9 else ("rougher" if forget_lip_ratio > 1.1 else "similar")
    retain_trend = "smoother" if retain_lip_ratio < 0.9 else ("rougher" if retain_lip_ratio > 1.1 else "similar")
    selective = (
        "✓ Selective smoothing on forget"
        if forget_lip_ratio < 0.9 and retain_lip_ratio > 0.95
        else "✗ Non-selective changes"
    )

    summary_text = (
        f"Lipschitzness Analysis Summary:\n\n"
        f"{model_a}:\n"
        f"- Forget Lipschitz: {np.mean(forget_lipschitz_a):.3f} ± {np.std(forget_lipschitz_a):.3f}\n"
        f"- Retain Lipschitz: {np.mean(retain_lipschitz_a):.3f} ± {np.std(retain_lipschitz_a):.3f}\n\n"
        f"{model_b}:\n"
        f"- Forget Lipschitz: {np.mean(forget_lipschitz_b):.3f} ± {np.std(forget_lipschitz_b):.3f}\n"
        f"- Retain Lipschitz: {np.mean(retain_lipschitz_b):.3f} ± {np.std(retain_lipschitz_b):.3f}\n\n"
        f"Changes (B/A ratio):\n"
        f"- Forget: {forget_lip_ratio:.2f}x\n"
        f"- Retain: {retain_lip_ratio:.2f}x\n\n"
        f"Interpretation:\n"
        f"Forget became {forget_trend}\n"
        f"Retain became {retain_trend}\n"
        f"{selective}"
    )
    axis.text(0.05, 0.5, summary_text, fontsize=9, family="monospace", verticalalignment="center")

    plt.suptitle(title or "Local Lipschitzness Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lipschitzness_analysis.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze local Lipschitzness of models on forget/retain data.",
    )
    parser.add_argument("--model-a", required=True, help="Baseline model")
    parser.add_argument("--model-b", required=True, help="Unlearned/finetuned model")
    parser.add_argument("--forget-text", default="data/forget.txt")
    parser.add_argument("--retain-text", default="data/retain.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Perturbation magnitude for Lipschitz estimation")
    parser.add_argument("--num-perturbations", type=int, default=10,
                        help="Number of perturbations for variance estimation")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max texts per split to process (default: 500)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="lipschitzness_analysis")

    method = infer_method_from_model_name(args.model_b)
    init_wandb("local_lipschitzness", args, method=method)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[local_lipschitz] Loading forget/retain texts...")
    with open(args.forget_text, "r") as fh:
        forget_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    with open(args.retain_text, "r") as fh:
        retain_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]

    print(
        f"[local_lipschitz] Loaded {len(forget_texts)} forget texts, "
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

    print(f"Loading Unlearned Model: {args.model_b}...")
    model_b = AutoModelForCausalLM.from_pretrained(args.model_b, torch_dtype=dtype, **device_map_kwargs)
    if not device_map_kwargs:
        model_b.to(device)
    model_b.eval()
    # --- Model A ---
    print("\n[local_lipschitz] === Analyzing Model A (baseline) ===")
    print("  Forget data:")
    forget_lipschitz_a = estimate_local_lipschitz(
        model_a, tokenizer, forget_texts, device, dtype, args.epsilon,
    )
    forget_gradient_a = compute_gradient_norms(
        model_a, tokenizer, forget_texts, device, dtype, args.max_length,
    )
    forget_variance_a = analyze_output_variance(
        model_a, tokenizer, forget_texts, device, dtype,
        args.num_perturbations, args.epsilon, args.max_length,
    )

    print("  Retain data:")
    retain_lipschitz_a = estimate_local_lipschitz(
        model_a, tokenizer, retain_texts, device, dtype, args.epsilon,
    )
    retain_gradient_a = compute_gradient_norms(
        model_a, tokenizer, retain_texts, device, dtype, args.max_length,
    )
    retain_variance_a = analyze_output_variance(
        model_a, tokenizer, retain_texts, device, dtype,
        args.num_perturbations, args.epsilon, args.max_length,
    )

    # --- Model B ---
    print("\n[local_lipschitz] === Analyzing Model B (target) ===")
    print("  Forget data:")
    forget_lipschitz_b = estimate_local_lipschitz(
        model_b, tokenizer, forget_texts, device, dtype, args.epsilon,
    )
    forget_gradient_b = compute_gradient_norms(
        model_b, tokenizer, forget_texts, device, dtype, args.max_length,
    )
    forget_variance_b = analyze_output_variance(
        model_b, tokenizer, forget_texts, device, dtype,
        args.num_perturbations, args.epsilon, args.max_length,
    )

    print("  Retain data:")
    retain_lipschitz_b = estimate_local_lipschitz(
        model_b, tokenizer, retain_texts, device, dtype, args.epsilon,
    )
    retain_gradient_b = compute_gradient_norms(
        model_b, tokenizer, retain_texts, device, dtype, args.max_length,
    )
    retain_variance_b = analyze_output_variance(
        model_b, tokenizer, retain_texts, device, dtype,
        args.num_perturbations, args.epsilon, args.max_length,
    )

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    results_summary = build_summary_rows(
        forget_lipschitz_a, retain_lipschitz_a,
        forget_gradient_a, retain_gradient_a,
        forget_variance_a, retain_variance_a,
        forget_lipschitz_b, retain_lipschitz_b,
        forget_gradient_b, retain_gradient_b,
        forget_variance_b, retain_variance_b,
    )

    write_csv(
        os.path.join(args.outdir, "lipschitzness_summary.csv"),
        results_summary,
        _LIPSCHITZ_FIELDNAMES,
    )

    # Plots
    plot_lipschitzness_analysis(
        forget_lipschitz_a, retain_lipschitz_a,
        forget_lipschitz_b, retain_lipschitz_b,
        forget_gradient_a, retain_gradient_a,
        forget_gradient_b, retain_gradient_b,
        forget_variance_a, retain_variance_a,
        forget_variance_b, retain_variance_b,
        args.outdir,
        title=args.title,
        model_a=args.model_a,
        model_b=args.model_b,
    )

    # Detailed arrays
    np.savez_compressed(
        os.path.join(args.outdir, "lipschitz_details.npz"),
        forget_lip_a=forget_lipschitz_a,
        forget_lip_b=forget_lipschitz_b,
        retain_lip_a=retain_lipschitz_a,
        retain_lip_b=retain_lipschitz_b,
        forget_grad_a=forget_gradient_a,
        forget_grad_b=forget_gradient_b,
        retain_grad_a=retain_gradient_a,
        retain_grad_b=retain_gradient_b,
        forget_var_a=forget_variance_a,
        forget_var_b=forget_variance_b,
        retain_var_a=retain_variance_a,
        retain_var_b=retain_variance_b,
    )

    # Summary JSON
    forget_lip_change = float(np.mean(forget_lipschitz_b) - np.mean(forget_lipschitz_a))
    retain_lip_change = float(np.mean(retain_lipschitz_b) - np.mean(retain_lipschitz_a))
    forget_lip_ratio = float(np.mean(forget_lipschitz_b)) / (float(np.mean(forget_lipschitz_a)) + 1e-10)
    retain_lip_ratio = float(np.mean(retain_lipschitz_b)) / (float(np.mean(retain_lipschitz_a)) + 1e-10)

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "forget_lipschitz_change": forget_lip_change,
        "retain_lipschitz_change": retain_lip_change,
        "forget_smoother": bool(forget_lip_ratio < 0.9),
        "selective_smoothing": bool(forget_lip_ratio < 0.9 and retain_lip_ratio > 0.95),
        "epsilon": args.epsilon,
        "num_perturbations": args.num_perturbations,
    }

    with open(os.path.join(args.outdir, "lipschitz_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    forget_trend = "smoother" if forget_lip_ratio < 0.9 else ("rougher" if forget_lip_ratio > 1.1 else "similar")
    print(f"\n[local_lipschitz] ✓ Results saved to {args.outdir}")
    print(f"[local_lipschitz] Forget Lipschitz change: {forget_lip_change:.3f}")
    print(f"[local_lipschitz] Retain Lipschitz change: {retain_lip_change:.3f}")
    print(f"[local_lipschitz] Forget became {forget_trend}")
    log_csv_as_table(os.path.join(args.outdir, "lipschitzness_summary.csv"), "lipschitzness_summary")
    log_plots(args.outdir, "lipschitzness")
    finish_wandb()


if __name__ == "__main__":
    main()