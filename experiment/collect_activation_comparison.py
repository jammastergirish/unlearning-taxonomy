#!/usr/bin/env python
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "tqdm",
#   "wandb",
#   "pandas",
#   "matplotlib",
# ]
# ///

"""
Collect per-layer activation norms for two models and compute their differences.

For each text split (forget / retain), this script:
  1. Runs model A, caches hidden states to disk, and records mean per-token norms.
  2. Runs model B on the same texts, loads cached model A hidden states, and
     records both model B norms and the element-wise difference norms.
  3. Writes a CSV with per-layer norm statistics.
  4. Optionally generates comparison plots (if --plot-outdir is provided).
"""

import argparse
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import tempfile
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import comparison_outdir, resolve_device, resolve_dtype, write_csv, init_wandb, infer_method_from_model_name, log_csv_as_table, log_plots, finish_wandb


def read_lines(path: str, max_samples: int) -> List[str]:
    """Read non-empty lines from a text file, up to max_samples."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_samples]


def _load_model(model_id: str, dtype: torch.dtype, device: str, method_name: str = "base"):
    """Load a causal language model and its tokenizer.

    Handles the dtype kwarg difference across transformers versions
    (some use ``dtype``, others require ``torch_dtype``).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[{method_name}] Loading model: {model_id}...")
    device_map_kwargs = {"device_map": "auto"} if device == "cuda" else {}
    
    # 2. Load the specified model
    if method_name in ("peft", "lora"):
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **device_map_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **device_map_kwargs
        )
    if not device_map_kwargs:
        model.to(device)
    model.eval()
    return model, tokenizer


def _tokenize_batch(tokenizer, texts: List[str], device: str, max_length: int) -> dict:
    """Tokenize a batch of texts and move tensors to the target device."""
    encoding = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return {key: value.to(device) for key, value in encoding.items()}


@torch.inference_mode()
def cache_hidden_states(
    model_id: str,
    texts: List[str],
    cache_dir: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    use_half_precision_cache: bool = False,
) -> Dict[str, object]:
    """
    Run a model on texts, cache hidden states to disk, and return mean per-token norms.

    Returns a dict with keys:
        - ``mean_l1_norm``: list of mean L1 norms per layer
        - ``mean_l2_norm``: list of mean L2 norms per layer
        - ``num_layers``: number of layers (including embedding layer)
    """
    model, tokenizer = _load_model(model_id, dtype, device)

    # Accumulators for per-layer mean norms (one entry per layer)
    sum_l1_norms = None
    sum_l2_norms = None
    total_tokens = 0.0

    short_name = model_id.split("/")[-1]

    for batch_idx, start in enumerate(tqdm(
        range(0, len(texts), batch_size),
        desc=f"Caching hidden states ({short_name})",
        unit="batch",
    )):
        batch_texts = texts[start : start + batch_size]
        inputs = _tokenize_batch(tokenizer, batch_texts, device, max_length)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError(
                "Model returned no hidden_states; check that output_hidden_states=True is supported."
            )

        if sum_l1_norms is None:
            sum_l1_norms = [0.0] * len(hidden_states)
            sum_l2_norms = [0.0] * len(hidden_states)

        attention_mask = inputs["attention_mask"].float()
        token_count = float(attention_mask.sum().item())
        total_tokens += token_count

        # Compute per-token norms at each layer, masked to real (non-padding) tokens
        for layer_idx, layer_hidden in enumerate(hidden_states):
            layer_float = layer_hidden.float()
            l1_per_token = layer_float.abs().sum(dim=-1)                        # [batch, seq_len]
            l2_per_token = torch.linalg.vector_norm(layer_float, ord=2, dim=-1)  # [batch, seq_len]
            sum_l1_norms[layer_idx] += float((l1_per_token * attention_mask).sum().item())
            sum_l2_norms[layer_idx] += float((l2_per_token * attention_mask).sum().item())

        # Save hidden states and mask for later diff computation against model B
        if use_half_precision_cache:
            batch_data = {
                "hidden_states": [h.cpu().half() for h in hidden_states],
                "attention_mask": inputs["attention_mask"].cpu(),
            }
        else:
            batch_data = {
                "hidden_states": [h.cpu() for h in hidden_states],
                "attention_mask": inputs["attention_mask"].cpu(),
            }
        torch.save(batch_data, os.path.join(cache_dir, f"batch_{batch_idx}.pt"))

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean_l1_norm": [s / total_tokens if total_tokens else 0.0 for s in sum_l1_norms],
        "mean_l2_norm": [s / total_tokens if total_tokens else 0.0 for s in sum_l2_norms],
        "num_layers": len(sum_l2_norms),
    }


@torch.inference_mode()
def compute_activation_diffs(
    model_id: str,
    texts: List[str],
    cache_dir: str,
    device: str,
    dtype: torch.dtype,
    max_length: int,
    batch_size: int,
    num_layers: int,
) -> Dict[str, List[float]]:
    """
    Run model B on texts, load cached model A hidden states, and compute diffs.

    Returns a dict with keys:
        - ``mean_l1_norm``: mean L1 norms for model B
        - ``mean_l2_norm``: mean L2 norms for model B
        - ``mean_diff_l1``: mean L1 norms of (model_B - model_A) hidden states
        - ``mean_diff_l2``: mean L2 norms of (model_B - model_A) hidden states
    """
    model, tokenizer = _load_model(model_id, dtype, device)

    # Accumulators
    sum_absolute_l1 = [0.0] * num_layers
    sum_absolute_l2 = [0.0] * num_layers
    sum_diff_l1 = [0.0] * num_layers
    sum_diff_l2 = [0.0] * num_layers
    total_tokens = 0.0

    short_name = model_id.split("/")[-1]
    for batch_idx, start in enumerate(tqdm(
        range(0, len(texts), batch_size),
        desc=f"Computing activation diffs ({short_name})",
        unit="batch",
    )):
        batch_texts = texts[start : start + batch_size]
        inputs = _tokenize_batch(tokenizer, batch_texts, device, max_length)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states_b = outputs.hidden_states

        # Load cached hidden states from model A
        cached = torch.load(
            os.path.join(cache_dir, f"batch_{batch_idx}.pt"), weights_only=True
        )
        hidden_states_a = cached["hidden_states"]
        cached_mask = cached["attention_mask"].float()

        attention_mask = cached_mask.to(device)
        token_count = float(attention_mask.sum().item())
        total_tokens += token_count

        for layer_idx in range(num_layers):
            layer_a = hidden_states_a[layer_idx].to(device=device, dtype=torch.float32)
            layer_b = hidden_states_b[layer_idx].float()

            # Handle potential sequence-length mismatch between cached and live run
            min_seq_len = min(layer_a.shape[1], layer_b.shape[1])
            layer_a = layer_a[:, :min_seq_len, :]
            layer_b = layer_b[:, :min_seq_len, :]
            layer_mask = attention_mask[:, :min_seq_len]

            # Absolute norms for model B
            l1_absolute = layer_b.abs().sum(dim=-1)
            l2_absolute = torch.linalg.vector_norm(layer_b, ord=2, dim=-1)
            sum_absolute_l1[layer_idx] += float((l1_absolute * layer_mask).sum().item())
            sum_absolute_l2[layer_idx] += float((l2_absolute * layer_mask).sum().item())

            # Difference norms: how much did activations change?
            diff = layer_b - layer_a
            diff_l1 = diff.abs().sum(dim=-1)
            diff_l2 = torch.linalg.vector_norm(diff, ord=2, dim=-1)
            sum_diff_l1[layer_idx] += float((diff_l1 * layer_mask).sum().item())
            sum_diff_l2[layer_idx] += float((diff_l2 * layer_mask).sum().item())

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean_l1_norm": [s / total_tokens if total_tokens else 0.0 for s in sum_absolute_l1],
        "mean_l2_norm": [s / total_tokens if total_tokens else 0.0 for s in sum_absolute_l2],
        "mean_diff_l1": [s / total_tokens if total_tokens else 0.0 for s in sum_diff_l1],
        "mean_diff_l2": [s / total_tokens if total_tokens else 0.0 for s in sum_diff_l2],
    }


# ---------------------------------------------------------------------------
# Plotting — generates activation norm comparison charts from the CSV
# ---------------------------------------------------------------------------
def plot_activation_comparison(csv_path: str, outdir: str, title: str = None,
                              model_a: str = "Model A (before)", model_b: str = "Model B (after)"):
    """Generate activation norm comparison plots from a per-layer CSV file.

    When the CSV contains ``*_std`` columns (produced by the multi-seed
    aggregator), shaded ±1 std error bands are drawn around each line.
    Falls back to plain lines when no ``_std`` columns are present.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe[dataframe["layer"] != "ALL_MEAN"]
    dataframe["layer"] = dataframe["layer"].astype(int)

    # Detect whether the aggregated CSV contains std columns
    has_std = any(c.endswith("_std") for c in dataframe.columns)

    print(f"[collect_activation_comparison] Generating plots from {csv_path} "
          f"({len(dataframe)} rows, error bars={'yes' if has_std else 'no'})")

    def _get_std(sub: "pd.DataFrame", col: str) -> "pd.Series | None":
        """Return the std series for *col* if available, else None."""
        std_col = f"{col}_std"
        return sub[std_col] if has_std and std_col in sub.columns else None

    def _plot_line_with_band(ax, x, y, std, label, color, marker):
        """Plot a line and, if std is given, a shaded ±1σ band."""
        ax.plot(x, y, marker=marker, linewidth=1.5, label=label, color=color)
        if std is not None:
            ax.fill_between(x, y - std, y + std, alpha=0.2, color=color)

    for split in ["forget", "retain"]:
        sub = dataframe[dataframe["split"] == split].sort_values("layer")
        if sub.empty:
            continue

        x = sub["layer"]

        # ------------------------------------------------------------------
        # Plot 1: Absolute norms — L1 and L2 side-by-side, model A vs model B
        # ------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        _plot_line_with_band(ax1, x, sub["model_a_l1_norm"],
                             _get_std(sub, "model_a_l1_norm"),
                             label=model_a, color="tab:blue", marker="o")
        _plot_line_with_band(ax1, x, sub["model_b_l1_norm"],
                             _get_std(sub, "model_b_l1_norm"),
                             label=model_b, color="tab:orange", marker="s")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel(r"Mean $\|h\|_1$ per token")
        ax1.set_title(f"$L_1$ Activation Magnitude ({split})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        _plot_line_with_band(ax2, x, sub["model_a_l2_norm"],
                             _get_std(sub, "model_a_l2_norm"),
                             label=model_a, color="tab:blue", marker="o")
        _plot_line_with_band(ax2, x, sub["model_b_l2_norm"],
                             _get_std(sub, "model_b_l2_norm"),
                             label=model_b, color="tab:orange", marker="s")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel(r"Mean $\|h\|_2$ per token")
        ax2.set_title(f"$L_2$ Activation Magnitude ({split})")
        ax2.legend()
        ax2.grid(alpha=0.3)

        suffix = " (mean ± 1σ)" if has_std else ""
        fig.suptitle((title or "") + suffix, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"activation_norms_{split}.png"))
        plt.close()

        # ------------------------------------------------------------------
        # Plot 2: Activation diffs — L1 and L2 side-by-side
        # ------------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        _plot_line_with_band(ax1, x, sub["mean_diff_l1"],
                             _get_std(sub, "mean_diff_l1"),
                             label=None, color="tab:green", marker="o")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel(r"Mean $\|\Delta h\|_1$ per token")
        ax1.set_title(f"Activation Diff $L_1$ Norm ({split})")
        ax1.grid(alpha=0.3)

        _plot_line_with_band(ax2, x, sub["mean_diff_l2"],
                             _get_std(sub, "mean_diff_l2"),
                             label=None, color="tab:red", marker="o")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel(r"Mean $\|\Delta h\|_2$ per token")
        ax2.set_title(f"Activation Diff $L_2$ Norm ({split})")
        ax2.grid(alpha=0.3)

        fig.suptitle((title or "") + suffix, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"activation_diffs_{split}.png"))
        plt.close()

    print(f"[collect_activation_comparison] ✓ All plots written to {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-layer activation norms for two models and compute their differences."
    )
    parser.add_argument("--model-a", required=False, default=None, help="Baseline model (before)")
    parser.add_argument("--model-b", required=False, default=None, help="Target model (after)")
    parser.add_argument("--forget-text", help="Path to forget-set prompts (one per line)")
    parser.add_argument("--retain-text", help="Path to retain-set prompts (one per line)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--outdir", default=None,
                        help="Output dir (default: auto-derived from model names)")
    parser.add_argument("--plot-outdir", default=None,
                        help="Plot dir (default: auto-derived from model names)")
    parser.add_argument("--title", default=None, help="Title for generated plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (accepted for pipeline compatibility).")
    parser.add_argument(
        "--cache-fp16",
        action="store_true",
        help="Use half-precision caching to reduce disk usage (default: full precision)",
    )
    parser.add_argument("--plot-from-csv", default=None,
                        help="Skip model loading — just regenerate plots from this CSV and exit")
    args = parser.parse_args()

    # ---- Plot-only mode (used by aggregate_multiseed_results.py) ----
    # Runs in the collect_activation_comparison venv which has torch/transformers,
    # so the module can be imported safely. The aggregator calls us as a subprocess
    # precisely to avoid needing torch in its own lighter venv.
    if args.plot_from_csv:
        if not args.plot_outdir:
            print("[collect_activation_comparison] --plot-from-csv requires --plot-outdir", flush=True)
            raise SystemExit(1)
        plot_activation_comparison(
            args.plot_from_csv,
            args.plot_outdir,
            title=args.title,
            model_a=args.model_a or "Model A (before)",
            model_b=args.model_b or "Model B (after)",
        )
        return

    if args.model_a is None or args.model_b is None:
        print("[collect_activation_comparison] --model-a and --model-b are required unless --plot-from-csv is used")
        raise SystemExit(1)

    if args.outdir is None:
        args.outdir = comparison_outdir(args.model_a, args.model_b, suffix="activation_comparison")
    if args.plot_outdir is None:
        args.plot_outdir = comparison_outdir(args.model_a, args.model_b, suffix="activation_plots")

    method = infer_method_from_model_name(args.model_b)
    init_wandb("activation_comparison", args, method=method)

    if not args.forget_text or not os.path.exists(args.forget_text):
        print("[collect_activation_comparison] Skipping: forget-text missing/not found")
        return
    if not args.retain_text or not os.path.exists(args.retain_text):
        print("[collect_activation_comparison] Skipping: retain-text missing/not found")
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    forget_texts = read_lines(args.forget_text, args.max_samples)
    retain_texts = read_lines(args.retain_text, args.max_samples)

    rows = []

    print(f"[collect_activation_comparison] Model A: {args.model_a}")
    print(f"[collect_activation_comparison] Model B: {args.model_b}")
    print(f"[collect_activation_comparison] Forget samples: {len(forget_texts)}  |  Retain samples: {len(retain_texts)}")
    if args.cache_fp16:
        print("[collect_activation_comparison] ⚠ Using FP16 caching — reduced precision for faster I/O")
    else:
        print("[collect_activation_comparison] Using full-precision caching")

    for split_name, texts in [("forget", forget_texts), ("retain", retain_texts)]:
        cache_dir = tempfile.mkdtemp(prefix="activation_cache_")

        try:
            print(f"\n[collect_activation_comparison] ── {split_name.upper()} split ──")

            # Phase 1: Cache model A hidden states and get its absolute norms
            result_a = cache_hidden_states(
                args.model_a, texts, cache_dir, device, dtype,
                args.max_length, args.batch_size, args.cache_fp16,
            )
            num_layers = result_a["num_layers"]

            # Phase 2: Run model B, compute absolute norms + diffs against model A
            result_b = compute_activation_diffs(
                args.model_b, texts, cache_dir, device, dtype,
                args.max_length, args.batch_size, num_layers,
            )

            # Collect per-layer results
            for layer_idx in range(num_layers):
                rows.append({
                    "layer": layer_idx,
                    "split": split_name,
                    "model_a_l1_norm": result_a["mean_l1_norm"][layer_idx],
                    "model_a_l2_norm": result_a["mean_l2_norm"][layer_idx],
                    "model_b_l1_norm": result_b["mean_l1_norm"][layer_idx],
                    "model_b_l2_norm": result_b["mean_l2_norm"][layer_idx],
                    "mean_diff_l1": result_b["mean_diff_l1"][layer_idx],
                    "mean_diff_l2": result_b["mean_diff_l2"][layer_idx],
                })

            # Summary row: average across all layers
            rows.append({
                "layer": "ALL_MEAN",
                "split": split_name,
                "model_a_l1_norm": float(np.mean(result_a["mean_l1_norm"])),
                "model_a_l2_norm": float(np.mean(result_a["mean_l2_norm"])),
                "model_b_l1_norm": float(np.mean(result_b["mean_l1_norm"])),
                "model_b_l2_norm": float(np.mean(result_b["mean_l2_norm"])),
                "mean_diff_l1": float(np.mean(result_b["mean_diff_l1"])),
                "mean_diff_l2": float(np.mean(result_b["mean_diff_l2"])),
            })

        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Write CSV output
    os.makedirs(args.outdir, exist_ok=True)

    output_path = os.path.join(args.outdir, "activation_comparison.csv")
    fieldnames = [
        "layer", "split",
        "model_a_l1_norm", "model_a_l2_norm",
        "model_b_l1_norm", "model_b_l2_norm",
        "mean_diff_l1", "mean_diff_l2",
    ]
    write_csv(output_path, rows, fieldnames)
    print(f"\n[collect_activation_comparison] ✓ Wrote activation stats to {output_path}")
    log_csv_as_table(output_path, "activation_comparison")

    # Generate plots if requested
    if args.plot_outdir:
        plot_activation_comparison(output_path, args.plot_outdir, args.title,
                                  model_a=args.model_a, model_b=args.model_b)
        log_plots(args.plot_outdir, "activation_plots")

    finish_wandb()


if __name__ == "__main__":
    main()
