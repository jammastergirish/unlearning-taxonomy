#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "matplotlib",
#   "datasets",
#   "tqdm",
#   "wandb",
#   "pandas",
# ]
# ///

"""
Layer-wise WMDP-Bio accuracy via logit lens or tuned lens.

For a single model, evaluates WMDP-Bio multiple-choice accuracy at every
transformer layer.  This reveals *where* hazardous knowledge becomes
accessible — and whether unlearning methods actually erase it or just
hide it from the final output head.

Two modes:
  logit  — project intermediate hidden states through the model's own
           final LayerNorm + unembedding head (lm_head).  Zero training cost.
  tuned  — train a per-layer affine transform (nn.Linear) to map hidden
           states → vocab logits.  More accurate at early layers but
           requires a training pass on held-out WMDP data.
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
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    model_outdir,
    resolve_device,
    resolve_dtype,
    write_csv,
    init_wandb,
    log_csv_as_table,
    log_plots,
    finish_wandb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_num_layers(model) -> int:
    """Auto-detect transformer layer count across common architectures."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    raise ValueError("Could not determine number of layers for model architecture")


def get_final_layer_norm(model):
    """Find the model's final layer norm (applied before lm_head)."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm       # GPT-NeoX
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm                       # LLaMA / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f                 # GPT-2
    return None


def get_lm_head(model):
    """Return the unembedding head, handling architecture naming differences.

    GPT-NeoX uses `embed_out`; GPT-2/LLaMA-style models use `lm_head`.
    """
    return getattr(model, "lm_head", None) or getattr(model, "embed_out", None)


def logit_lens_project(hidden_states: torch.Tensor, model) -> torch.Tensor:
    """Project hidden states to vocab logits using the model's own head."""
    final_norm = get_final_layer_norm(model)
    if final_norm is not None:
        hidden_states = final_norm(hidden_states)
    lm_head = get_lm_head(model)
    if lm_head is None:
        raise AttributeError(
            f"Could not find an unembedding head on {type(model).__name__}. "
            "Expected 'lm_head' or 'embed_out'."
        )
    return lm_head(hidden_states)


def load_wmdp_bio(max_samples: Optional[int] = None) -> List[Dict]:
    """Load WMDP-Bio multiple-choice dataset from HuggingFace.

    Returns list of dicts with keys: question, choices, answer (int index).
    """
    dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    items: List[Dict] = []
    for example in dataset:
        question = example.get("question", example.get("prompt", ""))
        choices = example.get("choices", [])
        answer = example.get("answer", 0)
        if question and choices:
            items.append({"question": question, "choices": choices, "answer": int(answer)})
    if max_samples:
        items = items[:max_samples]
    return items


# ---------------------------------------------------------------------------
# MCQ scoring (testable)
# ---------------------------------------------------------------------------

def score_single_mcq(
    model,
    tokenizer,
    question: str,
    choices: List[str],
    layer_index: int,
    device: str,
    max_length: int = 512,
    project_fn=None,
) -> int:
    """Return the index of the highest-scoring choice for one MCQ item.

    For each choice, computes the average per-token log-prob of the choice
    continuation appended to the question.

    Args:
        project_fn: callable(hidden_states) → logits.  If None, uses the
            model's native output logits (final layer only).

    Returns:
        predicted choice index (int).
    """
    choice_scores: List[float] = []
    for choice in choices:
        text = f"{question} {choice}"
        encoded = tokenizer(
            text, return_tensors="pt", max_length=max_length, truncation=True,
        ).to(device)
        input_ids = encoded["input_ids"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )
            if project_fn is not None:
                hidden = outputs.hidden_states[layer_index]  # (1, T, D)
                logits = project_fn(hidden)
            else:
                logits = outputs.logits  # final layer

        # Tokenize just the choice to find how many tokens it adds
        choice_encoded = tokenizer(f" {choice}", add_special_tokens=False)
        choice_length = len(choice_encoded["input_ids"])
        if choice_length == 0:
            choice_scores.append(float("-inf"))
            continue

        # Score: average log-prob over the last *choice_length* tokens
        sequence_length = input_ids.size(1)
        start = max(sequence_length - choice_length - 1, 0)
        end = sequence_length - 1  # logits are shifted by 1

        log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
        target_ids = input_ids[0, start + 1 : end + 1]
        token_log_probs = log_probs[torch.arange(log_probs.size(0)), target_ids]
        choice_scores.append(float(token_log_probs.mean().item()))

    return int(np.argmax(choice_scores)) if choice_scores else -1


def score_mcq_at_layer(
    model,
    tokenizer,
    items: List[Dict],
    layer_index: int,
    device: str,
    max_length: int = 512,
    project_fn=None,
) -> tuple:
    """Evaluate MCQ accuracy at a specific layer.

    Returns:
        (accuracy, num_correct, total)
    """
    correct = 0
    total = 0
    for item in items:
        predicted = score_single_mcq(
            model, tokenizer, item["question"], item["choices"],
            layer_index, device, max_length, project_fn,
        )
        if predicted == item["answer"]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


# ---------------------------------------------------------------------------
# Tuned lens
# ---------------------------------------------------------------------------

class TunedLensProbe(torch.nn.Module):
    """Per-layer affine transform: hidden_dim → vocab_size."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def train_tuned_lens(
    model,
    tokenizer,
    train_texts: List[str],
    layer_index: int,
    device: str,
    hidden_dim: int,
    vocab_size: int,
    max_length: int = 512,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    epochs: int = 3,
) -> TunedLensProbe:
    """Train a tuned lens probe for a single layer.

    Uses causal LM loss: the probe must predict the next token from the
    intermediate hidden state at the given layer.
    """
    probe = TunedLensProbe(hidden_dim, vocab_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    probe.train()
    for _epoch in range(epochs):
        np.random.shuffle(train_texts)
        for batch_start in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[batch_start : batch_start + batch_size]
            encoded = tokenizer(
                batch_texts, return_tensors="pt", max_length=max_length,
                truncation=True, padding=True,
            ).to(device)
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_index]  # (B, T, D)

            logits = probe(hidden)[:, :-1, :].contiguous()
            target_labels = encoded["input_ids"][:, 1:].contiguous()
            mask = encoded["attention_mask"][:, 1:].contiguous().float()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target_labels.view(-1), reduction="none",
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

_LENS_FIELDNAMES = ["layer", "accuracy", "correct", "total"]


def plot_wmdp_lens_results(
    results: List[Dict],
    final_accuracy: float,
    lens_type: str,
    outdir: str,
    model_name: str = "",
    title: Optional[str] = None,
) -> None:
    """Create the 1×2 panel of WMDP accuracy and delta plots."""
    layers = [r["layer"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Absolute accuracy by layer
    axis = axes[0]
    axis.plot(layers, accuracies, "o-", color="tab:blue", label=f"{lens_type} lens")
    axis.axhline(final_accuracy, color="tab:orange", ls="--", alpha=0.7,
                 label=f"Final layer ({final_accuracy:.3f})")
    axis.axhline(0.25, color="gray", ls=":", alpha=0.5,
                 label="Random chance (0.25)")
    axis.set_xlabel("Layer")
    axis.set_ylabel("WMDP-Bio Accuracy")
    axis.set_title("WMDP Accuracy by Layer")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    # 2. Delta from final layer
    axis = axes[1]
    deltas = [a - final_accuracy for a in accuracies]
    bar_colors = ["green" if d >= 0 else "red" for d in deltas]
    axis.bar(layers, deltas, color=bar_colors, alpha=0.6)
    axis.axhline(0, color="gray", ls="--", alpha=0.5)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Δ Accuracy (layer − final)")
    axis.set_title("Accuracy Delta from Final Layer")
    axis.grid(alpha=0.3)

    default_title = f"Layer-wise WMDP-Bio Accuracy ({lens_type} lens)"
    if model_name:
        default_title = f"{model_name}\n{default_title}"
    plt.suptitle(title or default_title, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "wmdp_lens_analysis.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise WMDP-Bio accuracy via logit/tuned lens.",
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--lens", choices=["logit", "tuned"], default="logit",
                        help="Lens type: logit (default) or tuned")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max WMDP questions to evaluate (default: 500)")
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: outputs/<model>/wmdp_<lens>_lens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuned-lr", type=float, default=1e-3,
                        help="Learning rate for tuned lens probes")
    parser.add_argument("--tuned-epochs", type=int, default=3,
                        help="Training epochs for tuned lens probes")
    parser.add_argument("--tuned-train-frac", type=float, default=0.3,
                        help="Fraction of WMDP data used to train tuned lens (rest for eval)")
    parser.add_argument("--title", default=None, help="Title for plots")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = model_outdir(args.model, suffix=f"wmdp_{args.lens}_lens")

    init_wandb(f"wmdp_{args.lens}_lens", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Load data
    print("[wmdp_lens] Loading WMDP-Bio dataset...")
    all_items = load_wmdp_bio(max_samples=args.max_samples)
    print(f"[wmdp_lens] {len(all_items)} MCQ items loaded")

    # Split for tuned lens training if needed
    if args.lens == "tuned":
        np.random.shuffle(all_items)
        num_train = int(len(all_items) * args.tuned_train_frac)
        train_items = all_items[:num_train]
        eval_items = all_items[num_train:]
        train_texts = [
            f"{item['question']} {item['choices'][item['answer']]}"
            for item in train_items
        ]
        print(f"[wmdp_lens] Tuned lens: {num_train} train, {len(eval_items)} eval")
    else:
        eval_items = all_items

    # Load model
    print(f"[wmdp_lens] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map_kwargs = {"device_map": "auto"} if device == "cuda" else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **device_map_kwargs)
    if not device_map_kwargs:
        model.to(device)
    model.eval()

    num_layers = get_num_layers(model)
    total_layers = num_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"[wmdp_lens] {total_layers} layers, hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    os.makedirs(args.outdir, exist_ok=True)

    # Evaluate per layer
    results: List[Dict] = []
    print(f"[wmdp_lens] Evaluating {args.lens} lens across {total_layers} layers...")

    for layer_index in tqdm(range(total_layers), desc=f"{args.lens} lens", unit="layer"):
        if args.lens == "logit":
            project_fn = lambda hidden, m=model: logit_lens_project(hidden, m)
        else:
            print(f"\n  Training tuned lens for layer {layer_index}...")
            probe = train_tuned_lens(
                model, tokenizer, train_texts, layer_index, device,
                hidden_dim, vocab_size, args.max_length, args.batch_size,
                args.tuned_lr, args.tuned_epochs,
            )
            project_fn = lambda hidden, p=probe: p(hidden)

        accuracy, num_correct, num_total = score_mcq_at_layer(
            model, tokenizer, eval_items, layer_index, device,
            args.max_length, project_fn=project_fn,
        )

        results.append({
            "layer": layer_index,
            "accuracy": round(float(accuracy), 4),
            "correct": num_correct,
            "total": num_total,
        })

    # Final-layer reference (using model's own logits, no lens)
    final_accuracy, final_correct, final_total = score_mcq_at_layer(
        model, tokenizer, eval_items, -1, device,
        args.max_length, project_fn=None,
    )
    print(f"\n[wmdp_lens] Final-layer accuracy (native): {final_accuracy:.4f} ({final_correct}/{final_total})")

    # Save CSV
    write_csv(os.path.join(args.outdir, "wmdp_lens_results.csv"), results, _LENS_FIELDNAMES)

    # Summary JSON
    best = max(results, key=lambda r: r["accuracy"])
    summary = {
        "model": args.model,
        "lens": args.lens,
        "num_layers": total_layers,
        "max_samples": args.max_samples,
        "final_layer_accuracy": final_accuracy,
        "best_layer": best["layer"],
        "best_layer_accuracy": best["accuracy"],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plots
    plot_wmdp_lens_results(results, final_accuracy, args.lens, args.outdir,
                           model_name=args.model, title=args.title)

    print(f"\n[wmdp_lens] ✓ Results saved to {args.outdir}")
    print(f"[wmdp_lens] Best layer: {best['layer']} (accuracy={best['accuracy']:.4f})")
    log_csv_as_table(os.path.join(args.outdir, "wmdp_lens_results.csv"), "wmdp_lens_results")
    log_plots(args.outdir, "wmdp_lens")
    finish_wandb()


if __name__ == "__main__":
    main()
