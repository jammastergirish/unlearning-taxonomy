#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "scikit-learn",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Activation-separation analysis between forget and retain datasets.

For each transformer layer (including the embedding layer), this script:
  1. Extracts mean-pooled hidden-state activations for both the forget and
     retain text splits, under both model A (baseline) and model B (target).
  2. Measures separation between the two splits using:
       - Cosine distance between centroids
       - Euclidean distance between centroids
       - Linear discriminability (LDA → AUC)
       - Between-cluster / within-cluster variance ratio
  3. Produces per-layer comparison plots and a JSON summary highlighting
     which layers exhibit the largest separation change.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    comparison_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_activations(
    model,
    tokenizer,
    texts: List[str],
    layer_index: int,
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract mean-pooled activations at *layer_index* for a list of texts.

    Returns an (N, hidden_dim) float32 numpy array.
    """
    all_activations: List[np.ndarray] = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index]
            # Average-pool over the sequence dimension → one vector per sample
            pooled = hidden_states.mean(dim=1)
            all_activations.append(pooled.cpu().float().numpy())

    return np.vstack(all_activations)


# ---------------------------------------------------------------------------
# Core metric computation (testable, no I/O)
# ---------------------------------------------------------------------------

def compute_separation_metrics(
    forget_activations: np.ndarray,
    retain_activations: np.ndarray,
) -> Dict[str, float]:
    """Compute separation metrics between forget and retain activation sets.

    Args:
        forget_activations: (N_forget, hidden_dim) array.
        retain_activations: (N_retain, hidden_dim) array.

    Returns a dict with:
        - ``cosine_distance``: 1 − cos(centroid_forget, centroid_retain).
        - ``euclidean_distance``: ‖centroid_forget − centroid_retain‖₂.
        - ``linear_discriminability_auc``: AUC of an LDA classifier (0.5 = random).
        - ``variance_ratio``: between-cluster / within-cluster variance.
        - ``forget_centroid_norm``: ‖centroid_forget‖₂.
        - ``retain_centroid_norm``: ‖centroid_retain‖₂.
    """
    forget_centroid = forget_activations.mean(axis=0)
    retain_centroid = retain_activations.mean(axis=0)

    # Cosine distance between centroids
    forget_unit = forget_centroid / (np.linalg.norm(forget_centroid) + 1e-8)
    retain_unit = retain_centroid / (np.linalg.norm(retain_centroid) + 1e-8)
    cosine_distance = 1.0 - float(np.dot(forget_unit, retain_unit))

    # Euclidean distance between centroids
    euclidean_distance = float(np.linalg.norm(forget_centroid - retain_centroid))

    # Linear discriminability via LDA
    combined_features = np.vstack([forget_activations, retain_activations])
    labels = np.array([0] * len(forget_activations) + [1] * len(retain_activations))

    shuffled_indices = np.random.permutation(len(combined_features))
    split_point = int(0.8 * len(combined_features))
    train_features = combined_features[shuffled_indices[:split_point]]
    test_features = combined_features[shuffled_indices[split_point:]]
    train_labels = labels[shuffled_indices[:split_point]]
    test_labels = labels[shuffled_indices[split_point:]]

    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(train_features, train_labels)
        predicted_probabilities = lda.predict_proba(test_features)[:, 1]
        auc_score = roc_auc_score(test_labels, predicted_probabilities)
    except Exception:
        auc_score = 0.5  # fallback if LDA fails (e.g. singular covariance)

    # Within-cluster vs between-cluster variance ratio
    forget_variance = np.var(forget_activations, axis=0).mean()
    retain_variance = np.var(retain_activations, axis=0).mean()
    within_cluster_variance = (forget_variance + retain_variance) / 2.0

    between_cluster_variance = np.var(
        np.vstack([forget_centroid, retain_centroid]), axis=0
    ).mean()
    variance_ratio = float(between_cluster_variance / (within_cluster_variance + 1e-8))

    return {
        "cosine_distance": float(cosine_distance),
        "euclidean_distance": float(euclidean_distance),
        "linear_discriminability_auc": float(auc_score),
        "variance_ratio": variance_ratio,
        "forget_centroid_norm": float(np.linalg.norm(forget_centroid)),
        "retain_centroid_norm": float(np.linalg.norm(retain_centroid)),
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

_METRIC_FIELDNAMES = [
    "layer",
    "cosine_distance",
    "euclidean_distance",
    "linear_discriminability_auc",
    "variance_ratio",
    "forget_centroid_norm",
    "retain_centroid_norm",
]


def plot_separation_analysis(
    results_model_a: List[Dict],
    results_model_b: List[Dict],
    outdir: str,
    title: Optional[str] = None,
    model_a: str = "Model A (baseline)",
    model_b: str = "Model B (unlearned)",
) -> None:
    """Create the 2×3 panel of separation-metric plots."""
    layers = [row["layer"] for row in results_model_a]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (0,0) Cosine distance
    axis = axes[0, 0]
    axis.plot(layers, [r["cosine_distance"] for r in results_model_a], "o-", label=model_a)
    axis.plot(layers, [r["cosine_distance"] for r in results_model_b], "s-", label=model_b)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Cosine Distance")
    axis.set_title("Forget/Retain Centroid Separation (Cosine)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,1) Linear discriminability AUC
    axis = axes[0, 1]
    axis.plot(layers, [r["linear_discriminability_auc"] for r in results_model_a], "o-", label=model_a)
    axis.plot(layers, [r["linear_discriminability_auc"] for r in results_model_b], "s-", label=model_b)
    axis.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("AUC Score")
    axis.set_title("Linear Discriminability (Forget vs Retain)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (0,2) Variance ratio
    axis = axes[0, 2]
    axis.plot(layers, [r["variance_ratio"] for r in results_model_a], "o-", label=model_a)
    axis.plot(layers, [r["variance_ratio"] for r in results_model_b], "s-", label=model_b)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Between / Within Variance Ratio")
    axis.set_title("Cluster Separation (Variance Ratio)")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,0) Δ cosine distance
    delta_cosine = [
        results_model_b[i]["cosine_distance"] - results_model_a[i]["cosine_distance"]
        for i in range(len(results_model_a))
    ]
    axis = axes[1, 0]
    axis.plot(layers, delta_cosine, "o-", color="green")
    axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Δ Cosine Distance (B − A)")
    axis.set_title("Change in Separation After Unlearning")
    axis.grid(alpha=0.3)

    # (1,1) Centroid norms
    axis = axes[1, 1]
    axis.plot(layers, [r["forget_centroid_norm"] for r in results_model_a], "o-",
              label=f"Forget ({model_a.split('/')[-1]})", color="red", alpha=0.5)
    axis.plot(layers, [r["retain_centroid_norm"] for r in results_model_a], "o-",
              label=f"Retain ({model_a.split('/')[-1]})", color="blue", alpha=0.5)
    axis.plot(layers, [r["forget_centroid_norm"] for r in results_model_b], "s-",
              label=f"Forget ({model_b.split('/')[-1]})", color="darkred")
    axis.plot(layers, [r["retain_centroid_norm"] for r in results_model_b], "s-",
              label=f"Retain ({model_b.split('/')[-1]})", color="darkblue")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Centroid L2 Norm")
    axis.set_title("Activation Magnitudes")
    axis.legend()
    axis.grid(alpha=0.3)

    # (1,2) Text summary
    axis = axes[1, 2]
    axis.axis("off")

    avg_cosine_change = float(np.mean(delta_cosine))
    avg_auc_a = float(np.mean([r["linear_discriminability_auc"] for r in results_model_a]))
    avg_auc_b = float(np.mean([r["linear_discriminability_auc"] for r in results_model_b]))

    separation_verdict = "✓ Increased separation" if avg_cosine_change > 0.05 else "✗ Minimal separation change"
    discriminability_verdict = "✓ More discriminable" if avg_auc_b > avg_auc_a + 0.05 else "✗ Similar discriminability"

    summary_text = (
        f"Summary Statistics:\n\n"
        f"{model_a}:\n"
        f"- Avg Linear Discriminability: {avg_auc_a:.3f}\n"
        f"- Avg Cosine Distance: {np.mean([r['cosine_distance'] for r in results_model_a]):.3f}\n\n"
        f"{model_b}:\n"
        f"- Avg Linear Discriminability: {avg_auc_b:.3f}\n"
        f"- Avg Cosine Distance: {np.mean([r['cosine_distance'] for r in results_model_b]):.3f}\n\n"
        f"Change (B − A):\n"
        f"- Avg Δ Cosine Distance: {avg_cosine_change:.3f}\n"
        f"- Avg Δ AUC: {avg_auc_b - avg_auc_a:.3f}\n\n"
        f"Interpretation:\n"
        f"{separation_verdict}\n"
        f"{discriminability_verdict}"
    )

    axis.text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
              verticalalignment="center")

    plt.suptitle(title or "Activation Separation Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "activation_separation_analysis.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation separation between forget and retain datasets."
    )
    parser.add_argument("--model-a", required=True, help="Baseline model")
    parser.add_argument("--model-b", required=True, help="Unlearned/fine-tuned model")
    parser.add_argument("--forget-text", default="data/forget.txt")
    parser.add_argument("--retain-text", default="data/retain.txt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max texts per split to process (default: 500)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="activation_separation")

    init_wandb("activation_separation", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load texts
    print("[activation_separation] Loading forget/retain texts...")
    with open(args.forget_text, "r") as fh:
        forget_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]
    with open(args.retain_text, "r") as fh:
        retain_texts = [line.strip() for line in fh if line.strip()][:args.max_samples]

    print(
        f"[activation_separation] Loaded {len(forget_texts)} forget texts, "
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

    # Determine number of layers
    if hasattr(model_a, "gpt_neox") and hasattr(model_a.gpt_neox, "layers"):
        num_layers = len(model_a.gpt_neox.layers)          # GPT-NeoX
    elif hasattr(model_a, "model") and hasattr(model_a.model, "layers"):
        num_layers = len(model_a.model.layers)              # LLaMA
    elif hasattr(model_a, "transformer") and hasattr(model_a.transformer, "h"):
        num_layers = len(model_a.transformer.h)             # GPT-2
    elif hasattr(model_a, "encoder") and hasattr(model_a.encoder, "layer"):
        num_layers = len(model_a.encoder.layer)             # BERT
    else:
        raise ValueError("Could not determine number of layers for model architecture")

    os.makedirs(args.outdir, exist_ok=True)

    results_model_a: List[Dict] = []
    results_model_b: List[Dict] = []

    # Analyze each layer (including embedding layer → num_layers + 1)
    print(f"[activation_separation] Analyzing {num_layers + 1} layers (including embedding layer)...")
    for layer_index in tqdm(range(num_layers + 1), desc="Analyzing layer separation", unit="layer"):
        print(
            f"  Layer {layer_index}/{num_layers}: Extracting forget & retain activations for both models...",
            flush=True,
        )

        forget_activations_a = get_activations(
            model_a, tokenizer, forget_texts, layer_index, device, args.max_length, args.batch_size,
        )
        retain_activations_a = get_activations(
            model_a, tokenizer, retain_texts, layer_index, device, args.max_length, args.batch_size,
        )
        forget_activations_b = get_activations(
            model_b, tokenizer, forget_texts, layer_index, device, args.max_length, args.batch_size,
        )
        retain_activations_b = get_activations(
            model_b, tokenizer, retain_texts, layer_index, device, args.max_length, args.batch_size,
        )

        metrics_a = compute_separation_metrics(forget_activations_a, retain_activations_a)
        metrics_b = compute_separation_metrics(forget_activations_b, retain_activations_b)

        metrics_a["layer"] = layer_index
        metrics_b["layer"] = layer_index

        results_model_a.append(metrics_a)
        results_model_b.append(metrics_b)

        # Save every 4th layer's raw activations (disk-space management)
        if layer_index % 4 == 0:
            np.savez_compressed(
                os.path.join(args.outdir, f"activations_layer_{layer_index}.npz"),
                forget_a=forget_activations_a,
                retain_a=retain_activations_a,
                forget_b=forget_activations_b,
                retain_b=retain_activations_b,
            )

    # Write per-model CSVs
    write_csv(
        os.path.join(args.outdir, "separation_metrics_model_a.csv"),
        results_model_a,
        _METRIC_FIELDNAMES,
    )
    write_csv(
        os.path.join(args.outdir, "separation_metrics_model_b.csv"),
        results_model_b,
        _METRIC_FIELDNAMES,
    )

    # Plots
    plot_separation_analysis(results_model_a, results_model_b, args.outdir, title=args.title,
                              model_a=args.model_a, model_b=args.model_b)

    # JSON summary
    delta_cosine = [
        results_model_b[i]["cosine_distance"] - results_model_a[i]["cosine_distance"]
        for i in range(len(results_model_a))
    ]
    avg_auc_a = float(np.mean([r["linear_discriminability_auc"] for r in results_model_a]))
    avg_auc_b = float(np.mean([r["linear_discriminability_auc"] for r in results_model_b]))

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "avg_cosine_change": float(np.mean(delta_cosine)),
        "avg_auc_change": float(avg_auc_b - avg_auc_a),
        "max_separation_layer": int([r["layer"] for r in results_model_a][np.argmax(delta_cosine)]),
        "max_separation_value": float(max(delta_cosine)),
    }

    with open(os.path.join(args.outdir, "separation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[activation_separation] ✓ Results saved to {args.outdir}")
    print(f"[activation_separation] Average separation change (cosine): {summary['avg_cosine_change']:.3f}")
    print(f"[activation_separation] Maximum separation at layer {summary['max_separation_layer']}: {summary['max_separation_value']:.3f}")
    log_csv_as_table(os.path.join(args.outdir, "separation_metrics_model_a.csv"), "separation_metrics")
    log_plots(args.outdir, "activation_separation")
    finish_wandb()


if __name__ == "__main__":
    main()