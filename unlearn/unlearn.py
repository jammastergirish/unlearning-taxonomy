# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate>=0.27",
#     "tqdm",
#     "wandb",
# ]
# ///
"""
Multi-method LLM unlearning pipeline.

Supported methods:
  ga_simple  — Pure Gradient Ascent on forget set only (no retain loss)
  ga         — Gradient Ascent on forget set + Gradient Descent on retain set
  grad_diff  — Gradient Difference (weighted forget ascent + retain descent)
  dpo        — Direct Preference Optimization (forget=rejected, retain=chosen)
  npo        — Negative Preference Optimization (DPO-inspired, with reference model)
  simnpo     — Simple NPO (reference-free variant of NPO + retain NLL)
  rmu        — Representation Misdirection for Unlearning
  cb         — Circuit Breakers (representation rerouting via cosine similarity)
  lat        — Latent Adversarial Training (adversarial perturbations in hidden states)
  tar        — Task Arithmetic Removal (subtract forget fine-tuning direction)
  wt_dist    — Weight Distortion (Gaussian noise + retain fine-tuning)
  wt_dist_reg — Weight Distance Regularization (maximize L2 distance from pretrained)

Usage:
  uv run --script unlearn.py --model <HF_ID> --method ga --outdir outputs/test
"""

import argparse
import math
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# ---------------------------------------------------------------------------
# Reuse device / dtype helpers from utils.py if available, else inline
# ---------------------------------------------------------------------------
try:
    from utils import resolve_device, resolve_dtype, model_outdir, filter_gpus_by_free_vram, compute_training_max_memory
except ImportError as e:
    raise ImportError(f"Could not import utils.py from project root: {e}") from e


# ===================================================================
# Data loading
# ===================================================================
# The unlearning pipeline needs two datasets:
#   1. "forget" set — text the model should *un*-learn (e.g. hazardous knowledge)
#   2. "retain" set — text the model should *keep* performing well on
# Both are plain .txt files with one sample per line.
# ===================================================================

def load_lines(path: str, max_lines: int | None = None) -> list[str]:
    """Load non-empty lines from a text file.

    Each non-blank line becomes one training sample.  `max_lines` lets you
    cap the dataset size for quick debugging runs.
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_lines:
        lines = lines[:max_lines]
    return lines


def tokenize_texts(
    texts: list[str], tokenizer, max_length: int, device: str
) -> list[dict]:
    """Tokenize a list of strings into individual dicts of {input_ids, attention_mask}.

    Each text is tokenized independently and padded/truncated to `max_length`.
    Returns a list of single-sample dicts (batch dim = 1 each), which are
    later grouped into mini-batches by `make_batches()`.
    """
    batches = []
    vocab_size = len(tokenizer)
    for i, text in enumerate(texts):
        # HuggingFace tokenizer returns {"input_ids": (1, T), "attention_mask": (1, T)}
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Validate token IDs are within vocabulary range
        input_ids = enc["input_ids"]
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()

        if max_id >= vocab_size or min_id < 0:
            print(f"[WARNING] Invalid tokens in sample {i}: range [{min_id}, {max_id}], vocab_size={vocab_size}")
            print(f"  Text preview: {text[:100]}...")
            # Clip invalid tokens to valid range
            enc["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)

        # Move tensors to the target device (GPU/MPS/CPU)
        batches.append({k: v.to(device) for k, v in enc.items()})
    return batches


def make_batches(items: list[dict], batch_size: int, drop_last: bool = True) -> list[list[dict]]:
    """Group single-sample dicts into mini-batches by concatenating along dim 0.

    If drop_last is True (default), discard the final batch when it is
    smaller than batch_size.  This prevents size-mismatch errors when
    forget and retain batches are paired (e.g. in DPO / NPO).

    Example: 10 items with batch_size=4 → 2 full batches (items 0-3, 4-7),
    the remaining 2 items are dropped.
    """
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        if drop_last and len(chunk) < batch_size:
            break
        # Concatenate single-sample tensors → (batch_size, T) for each key
        batch = {
            k: torch.cat([c[k] for c in chunk], dim=0) for k in chunk[0]
        }
        batches.append(batch)
    return batches


# ===================================================================
# Loss functions for each method
# ===================================================================
# All unlearning methods ultimately manipulate two fundamental quantities:
#   1. NLL (negative log-likelihood) — how well the model predicts next tokens
#   2. Log-probabilities — per-token likelihood under a model
#
# "Unlearning" means making the model WORSE at predicting forget-set tokens
# (high NLL on forget) while staying GOOD at predicting retain-set tokens
# (low NLL on retain).  Each method below achieves this differently.
# ===================================================================


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Memory-efficient cross-entropy that never materialises the full token logit matrix.

    Background
    ----------
    Standard ``F.cross_entropy(logits.view(-1, V), labels.view(-1))`` needs to
    build an intermediate tensor of shape ``(B * T, vocab_size)`` on the *single
    GPU* that holds the final layer — typically ~3 GiB for a 7B model at
    batch=32, seq=512, vocab=50K.  This pushes an already-full A40 over the
    edge. 

    The fix
    -------
    https://github.com/karpathy/nanochat/pull/128
    
    Cross-entropy is **independent per token** — token ``i``'s loss depends
    only on ``logits[i]`` and ``labels[i]``, never on any other position.  So
    we can process ``chunk_size`` batch elements at a time, free each chunk
    before moving to the next, and concatenate the per-token losses at the end.
    Peak memory scales with ``chunk_size`` instead of ``B``, while the numbers
    are **bit-for-bit identical** to computing everything at once.

    Parameters
    ----------
    logits :
        Shape ``(B, T, vocab_size)`` — raw model output *before* softmax.
    labels :
        Shape ``(B, T)`` — ground-truth next-token IDs.
    chunk_size :
        Number of batch elements to process per chunk.  Smaller → less peak
        memory, slightly more kernel overhead.  Default of 4 keeps peak logit
        allocation to ~0.4 GiB for the above example.

    Returns
    -------
    Flat ``(B * T,)`` tensor of per-token cross-entropy values, in the same
    order as ``logits.view(-1, V)`` / ``labels.view(-1)`` — a drop-in
    replacement for ``F.cross_entropy(..., reduction="none")``.
    """
    B = logits.size(0)
    chunks = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        # logits_chunk: (chunk, T, V) — only a slice of the batch is live at once
        logits_chunk = logits[start:end].reshape(-1, logits.size(-1))
        labels_chunk = labels[start:end].reshape(-1)
        # F.cross_entropy does: log_softmax(logits) then NLL — no full-batch
        # allocation needed because we are already working on a small slice.
        chunk_loss = F.cross_entropy(logits_chunk, labels_chunk, reduction="none")
        chunks.append(chunk_loss)
        # Explicitly delete the slice so the allocator can reuse the memory
        # before the next iteration allocates a new one.
        del logits_chunk, labels_chunk
    return torch.cat(chunks)


def nll_loss(model, batch: dict) -> torch.Tensor:
    """Standard next-token prediction (causal LM) loss.

    This is the same cross-entropy loss used during normal pre-training.
    For unlearning, we either MINIMIZE it (on retain data) to preserve
    capability, or NEGATE it (on forget data) to degrade capability.
    """
    input_ids = batch["input_ids"]          # (B, T)
    attention_mask = batch["attention_mask"]  # (B, T) — 1 for real tokens, 0 for padding
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shift logits and labels for next-token prediction:
    #   logits[:, :-1] predicts token at position t+1
    #   labels[:, 1:]  is the actual token at position t+1
    logits = outputs.logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
    labels = input_ids[:, 1:].contiguous()            # (B, T-1)
    mask = attention_mask[:, 1:].contiguous()          # (B, T-1)

    # Check for vocabulary size mismatch before computing loss
    vocab_size = logits.size(-1)
    max_label = labels.max().item()
    min_label = labels.min().item()

    if max_label >= vocab_size or min_label < 0:
        print(f"\n[ERROR] Vocabulary size mismatch detected in nll_loss:")
        print(f"  Model vocab size: {vocab_size}")
        print(f"  Label range: [{min_label}, {max_label}]")

        # Find specific problematic tokens
        invalid_mask = (labels >= vocab_size) | (labels < 0)
        if invalid_mask.any():
            invalid_tokens = labels[invalid_mask].unique().tolist()[:10]  # Show first 10 unique invalid tokens
            print(f"  Invalid token IDs: {invalid_tokens}")
            print(f"  Batch shape: input_ids={batch['input_ids'].shape}")

            # This shouldn't happen if tokenization was done correctly
            # Return a large loss that requires grad to avoid CUDA crash
            print("  CRITICAL: Returning dummy loss to avoid CUDA crash")
            dummy_loss = torch.tensor(1e10, device=logits.device, dtype=logits.dtype, requires_grad=True)
            return dummy_loss

        # This path shouldn't be reached if above check works
        labels = torch.clamp(labels, 0, vocab_size - 1)

    # Per-token cross-entropy, then mask out padding and average over real tokens.
    # Using chunked_cross_entropy instead of F.cross_entropy to avoid
    # materialising the full (B*T, vocab_size) tensor at once on the GPU that
    # holds the lm_head — the peak allocation (~3 GiB at batch=32) would OOM
    # on an A40 that already has model weights + optimizer states loaded.
    loss = chunked_cross_entropy(logits, labels.view(logits.size(0), -1))
    loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
    return loss


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-token log-probabilities for the given labels.

    Args:
        logits: (B, T, vocab_size) — raw model output
        labels: (B, T) — ground-truth token indices

    Returns:
        (B, T) tensor where entry [b, t] = log P(labels[b,t] | context)
    """
    log_p = F.log_softmax(logits, dim=-1)  # normalize logits → log-probs
    # Gather only the log-prob of the correct token at each position
    return torch.gather(log_p, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def avg_log_prob(
    model, batch: dict, return_per_token: bool = False
) -> torch.Tensor:
    """Average per-token log-prob under `model` for the batch.

    Used by DPO, NPO, and SimNPO to compute how likely a sequence is
    under the current policy (or reference) model.  Higher values mean
    the model assigns higher probability to the sequence.

    Returns shape (B,) — one scalar per sample in the batch.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    # Use no_grad when called on a frozen reference model (not training),
    # but allow gradients when called on the policy model being trained.
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    per_token = log_probs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    mask = attention_mask[:, 1:].float()
    # Average log-prob over real (non-padding) tokens in each sample
    avg = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    if return_per_token:
        return avg, per_token, mask
    return avg


# ---- GA Simple ---------------------------------------------------------
# The simplest possible unlearning method.  During normal training we
# MINIMIZE cross-entropy (gradient descent).  Here we MAXIMIZE it on
# the forget set (gradient ascent) so the model becomes *worse* at
# predicting those tokens.  No retain term — so there’s a risk of
# catastrophic forgetting of useful capabilities.

def ga_simple_loss(model, forget_batch: dict) -> torch.Tensor:
    """Pure Gradient Ascent: negate NLL on forget set only.

    By returning the negative NLL, the optimizer’s gradient descent
    effectively becomes gradient ASCENT on the forget data.
    """
    return -nll_loss(model, forget_batch)


# ---- GA ----------------------------------------------------------------
# Same idea as GA Simple, but adds a standard NLL term on the retain set
# to preserve the model’s general capabilities.  The total loss is:
#   L = -NLL_forget + NLL_retain
# So the optimizer simultaneously pushes UP the forget loss and pushes
# DOWN the retain loss.

def ga_loss(model, forget_batch: dict, retain_batch: dict, retain_weight: float = 1.0) -> torch.Tensor:
    """Gradient Ascent: negate NLL on forget + standard NLL on retain."""
    l_forget = -nll_loss(model, forget_batch)  # ascent (maximise forget NLL)
    l_retain = nll_loss(model, retain_batch)   # descent (minimise retain NLL)
    return l_forget + retain_weight * l_retain


# ---- GradDiff ----------------------------------------------------------
# Gradient Difference is similar to GA but lets you control the relative
# importance of forgetting vs. retaining via `forget_weight`.
#   L = NLL_retain - weight * NLL_forget
# The minus sign on the forget term does gradient ascent (unlearn), while
# the retain term does gradient descent (preserve).  A higher
# `forget_weight` pushes harder on forgetting at the cost of retain quality.

def grad_diff_loss(
    model, forget_batch: dict, retain_batch: dict, forget_weight: float = 1.0
) -> torch.Tensor:
    """Gradient Difference: L = retain_NLL - weight * forget_NLL."""
    l_retain = nll_loss(model, retain_batch)
    l_forget = nll_loss(model, forget_batch)
    return l_retain - forget_weight * l_forget


# ---- DPO ---------------------------------------------------------------
# Direct Preference Optimization (Rafailov et al., 2023), repurposed for
# unlearning.  Originally designed to align LLMs with human preferences
# by treating one response as "chosen" (preferred) and another as
# "rejected" (dispreferred).  Here:
#   - "chosen"   = retain data  (model should still be good at this)
#   - "rejected" = forget data  (model should become bad at this)
#
# DPO requires a frozen *reference model* (π_ref) — a copy of the
# original pre-trained weights — to prevent the policy from drifting
# too far.  The loss encourages the policy to increase the probability
# gap between chosen and rejected relative to the reference.
#
# β (beta) is an inverse temperature: higher β = more aggressive updates.

def dpo_loss(
    model,
    ref_model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
) -> torch.Tensor:
    """
    DPO loss: chosen=retain, rejected=forget.
    L = -log σ(β * (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))
    """
    # Policy (trainable) model log-probs for chosen (retain) & rejected (forget)
    lp_chosen = avg_log_prob(model, retain_batch)
    lp_rejected = avg_log_prob(model, forget_batch)

    # Reference (frozen) model log-probs — serves as the baseline
    with torch.no_grad():
        ref_lp_chosen = avg_log_prob(ref_model, retain_batch)
        ref_lp_rejected = avg_log_prob(ref_model, forget_batch)

    # Core DPO formula: reward margin between chosen and rejected,
    # relative to the reference model.  The sigmoid maps this to [0, 1];
    # we minimise the negative log-sigmoid (i.e., maximise log-sigmoid).
    logits_diff = beta * (
        (lp_chosen - ref_lp_chosen) - (lp_rejected - ref_lp_rejected)
    )
    return -F.logsigmoid(logits_diff).mean()


# ---- NPO ---------------------------------------------------------------
# Negative Preference Optimization.  A DPO-inspired approach that only
# needs the forget set (no explicit "chosen" data in the preference term).
# It penalises the policy for assigning higher log-probs to forget data
# than the reference model does.  A separate NLL term on the retain set
# keeps general capability intact.
#
# Mathematically:
#   L = -(2/β) * E[log σ(-β * (log π_θ - log π_ref))]  on forget
#       + NLL on retain
#
# The -(2/β) scaling ensures the gradient magnitude stays reasonable
# across different β values.

def npo_loss(
    model,
    ref_model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
    retain_weight: float = 1.0,
) -> torch.Tensor:
    """
    NPO: L = -(2/β) * E[log σ(-β * log(π_θ / π_ref))]  on forget
         + NLL on retain
    """
    # Policy log-prob on forget data
    lp_forget = avg_log_prob(model, forget_batch)
    # Reference log-prob on forget data (frozen, no gradients)
    with torch.no_grad():
        ref_lp_forget = avg_log_prob(ref_model, forget_batch)

    # NPO term: penalise the policy if it assigns higher prob to forget
    # data than the reference does  (lp_forget - ref_lp_forget > 0 is bad)
    npo_term = -(2.0 / beta) * F.logsigmoid(
        -beta * (lp_forget - ref_lp_forget)
    ).mean()

    # Standard retain NLL to preserve general capabilities
    retain_nll = nll_loss(model, retain_batch)
    return npo_term + retain_weight * retain_nll


# ---- SimNPO ------------------------------------------------------------
# Simple NPO — a *reference-free* variant of NPO.  Instead of comparing
# the policy to a frozen reference, it directly penalises the policy’s
# absolute log-prob on the forget set.  This is cheaper (no ref model
# needed) but can be less stable since there’s no anchor.
#
#   L = -(2/β) * E[log σ(-β * avg_log_prob_θ(forget))]
#       + NLL on retain

def simnpo_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    beta: float,
    retain_weight: float = 1.0,
) -> torch.Tensor:
    """
    SimNPO (reference-free): L = -(2/β) * E[log σ(-β * avg_log_prob_θ)]
                                + NLL on retain
    """
    # No reference model — use the policy’s own log-prob directly
    lp_forget = avg_log_prob(model, forget_batch)

    # Penalise high log-prob on forget data (sigmoid pushes it down)
    simnpo_term = -(2.0 / beta) * F.logsigmoid(-beta * lp_forget).mean()
    retain_nll = nll_loss(model, retain_batch)
    return simnpo_term + retain_weight * retain_nll


# ---- RMU ---------------------------------------------------------------
# Representation Misdirection for Unlearning.
# Unlike the output-space methods above (which manipulate logits/losses),
# RMU operates in the model’s *internal representation space*.
#
# The idea: at selected intermediate layers, push the forget-set hidden
# states toward a fixed random direction (so the model can’t extract
# meaningful information from them), while keeping retain-set hidden
# states close to their original (pre-training) values.
#
# This requires:
#   1. Choosing target layers (e.g., layers 5, 6, 7)
#   2. Caching the retain-set activations from the *original* model
#   3. Generating a fixed random unit vector per layer as the “misdirection” target

def get_layer_activations(model, batch: dict, layer_ids: list[int]):
    """
    Run a forward pass and capture hidden states at specified layer indices.
    Returns dict {layer_id: tensor of shape (B, T, D)}.

    HuggingFace models return `output_hidden_states` as a tuple of
    (n_layers + 1) tensors: [embedding_output, layer_0_output, ..., layer_N_output].
    We add +1 to the layer ID to skip the embedding tensor.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    # hidden_states[0] = embedding output, hidden_states[i+1] = layer i output
    hidden = outputs.hidden_states
    return {lid: hidden[lid + 1] for lid in layer_ids}  # +1 to skip embedding


def rmu_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,  # {layer_id: (D,) tensor — fixed random unit vector}
    retain_targets: dict,  # {layer_id: (B, T, D) tensor — cached clean activations}
    steering_coeff: float,  # scales the random target magnitude
    alpha: float,           # weight for the retain preservation term
) -> torch.Tensor:
    """
    RMU: push forget-set activations toward random target,
         pull retain-set activations toward original (cached) activations.

    The loss has two terms per target layer:
      1. Forget MSE: ||h_forget - (coeff * random_dir)||^2
         → forces forget-set representations to become meaningless noise
      2. Retain MSE: α * ||h_retain - h_retain_original||^2
         → keeps retain-set representations close to their original values
    """
    # Get current hidden states for both forget and retain inputs
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    loss = torch.tensor(0.0, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

    for lid in layer_ids:
        # Forget: MSE toward (steering_coeff * random_direction)
        # Expand the (D,) random vector to match (B, T, D) activation shape
        target_f = random_targets[lid].unsqueeze(0).unsqueeze(0) * steering_coeff
        loss = loss + F.mse_loss(forget_acts[lid], target_f.expand_as(forget_acts[lid]))

        # Retain: MSE toward cached clean activations (from before training)
        target_r = retain_targets[lid].to(retain_acts[lid].device)  # move cached activations to current device
        # Handle batch-size mismatch by truncating to the smaller batch
        bsz = min(retain_acts[lid].size(0), target_r.size(0))
        loss = loss + alpha * F.mse_loss(
            retain_acts[lid][:bsz], target_r[:bsz].detach()
        )

    return loss


# ---- Circuit Breakers --------------------------------------------------
# Circuit Breakers (aka Representation Rerouting) is similar to RMU but
# uses *cosine similarity* instead of MSE to steer activations.
#
# Why cosine instead of MSE?  Cosine similarity only cares about the
# *direction* of the activation vector, not its magnitude.  This can be
# more robust because it doesn’t fight against the model’s natural norm
# scaling — it just ensures forget-set activations *point* in a random
# (meaningless) direction.
#
# Loss terms per layer:
#   Forget: -cos_sim(h_forget, random_dir)  → MAXIMIZE alignment with noise
#   Retain: α * (1 - cos_sim(h_retain, h_retain_orig))  → PRESERVE direction

def cb_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,
    retain_targets: dict,
    steering_coeff: float,
    alpha: float,
) -> torch.Tensor:
    """
    Circuit Breakers (Representation Rerouting):
    Forget: maximize cosine similarity toward random direction.
    Retain: minimize cosine distance from original activations.
    """
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    loss = torch.tensor(0.0, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

    for lid in layer_ids:
        # Forget: flatten (B, T, D) → (B*T, D) so each token position gets
        # its own cosine similarity measurement against the random target
        fa = forget_acts[lid].flatten(0, 1)  # (B*T, D)
        rt = random_targets[lid].unsqueeze(0).expand_as(fa) * steering_coeff
        # We want fa to ALIGN with rt, so minimize negative cosine sim
        cos_sim = F.cosine_similarity(fa, rt, dim=-1)
        loss = loss - cos_sim.mean()  # negate: gradient descent on this = push TOWARD rt

        # Retain: keep activations pointing in the same direction as the
        # cached original activations.  (1 - cos_sim) = 0 when perfectly aligned.
        ra = retain_acts[lid]
        tr = retain_targets[lid].to(ra.device)  # move cached activations to current device
        bsz = min(ra.size(0), tr.size(0))
        ra_flat = ra[:bsz].flatten(0, 1)
        tr_flat = tr[:bsz].detach().flatten(0, 1)
        retain_cos = F.cosine_similarity(ra_flat, tr_flat, dim=-1)
        loss = loss + alpha * (1.0 - retain_cos.mean())  # penalise directional drift

    return loss


# ---- LAT ---------------------------------------------------------------
# Latent Adversarial Training operates differently from all the above:
# it uses a *two-loop* (min-max) optimisation:
#
# INNER LOOP (adversary):
#   Find a small perturbation δ (added to a hidden layer) that HELPS the
#   model recall forget-set data.  Think of δ as an attacker trying to
#   "jailbreak" the unlearning by nudging internal representations.
#
# OUTER LOOP (defender):
#   Train the model so that even WITH the best adversarial δ, it still
#   can’t produce the forget-set outputs.  This makes unlearning robust
#   to representation-level attacks.
#
# The perturbation δ is constrained to an L∞ ball of radius `lat_eps`
# (similar to adversarial training in vision).  `lat_steps` controls
# how many PGD (Projected Gradient Descent) steps the adversary gets.

def lat_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    lat_eps: float,      # L∞ budget for the adversarial perturbation
    lat_steps: int,       # number of PGD steps for the inner adversary
    retain_weight: float = 1.0,
) -> torch.Tensor:
    """
    Latent Adversarial Training:
    1. Inner loop: find adversarial perturbation δ at target layers that
       MAXIMIZES the model's ability to produce forget-set outputs.
    2. Outer loop: train model to produce HIGH loss on forget even WITH δ,
       plus standard retain NLL.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # --- Forward pass to discover the hidden state shape ---
    # We need to know (B, T, D) at the target layer before creating δ
    with torch.no_grad():
        out = model(
            input_ids=forget_batch["input_ids"],
            attention_mask=forget_batch["attention_mask"],
            output_hidden_states=True,
        )
        # Pick the middle target layer for perturbation (a heuristic choice)
        target_lid = layer_ids[len(layer_ids) // 2]
        hidden_shape = out.hidden_states[target_lid + 1].shape  # (B, T, D)

    # --- Inner loop: PGD to find adversarial perturbation δ ---
    # δ starts as zeros and is iteratively updated via signed gradients
    delta = torch.zeros(hidden_shape, device=device, dtype=model_dtype, requires_grad=True)

    for _adv_step in range(lat_steps):
        # PyTorch "forward hooks" let us intercept & modify a layer’s
        # output during the forward pass.  We use this to inject δ
        # into the target layer’s hidden states.
        handle = None
        hook_layer_idx = [0]

        def make_hook(d):
            """Create a hook that adds perturbation `d` to the layer output."""
            def hook_fn(module, input, output):
                # Some layers return tuples (hidden_state, attention_weights, ...)
                if isinstance(output, tuple):
                    return ((output[0] + d).to(output[0].dtype),) + output[1:]
                return (output + d).to(output.dtype)
            return hook_fn

        # Find the target layer module in the model’s architecture.
        # Different HF model families use different attribute paths:
        #   - LLaMA/Mistral: model.layers
        #   - GPT-NeoX:      gpt_neox.layers
        #   - GPT-2:         transformer.h
        layers = None
        for attr in ["model.layers", "gpt_neox.layers", "transformer.h"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            # If we can’t find the layer structure, fall back to simple GA
            print(f"[WARNING] LAT: Could not find model layers structure. Falling back to gradient ascent loss.")
            print(f"[WARNING] LAT: This may happen with non-standard model architectures.")
            return -nll_loss(model, forget_batch) + nll_loss(model, retain_batch)

        # Register the hook to inject δ into the target layer
        handle = layers[target_lid].register_forward_hook(make_hook(delta))

        # Forward pass with perturbation active
        out = model(
            input_ids=forget_batch["input_ids"],
            attention_mask=forget_batch["attention_mask"],
        )
        logits = out.logits[:, :-1, :].contiguous()
        labels = forget_batch["input_ids"][:, 1:].contiguous()
        mask = forget_batch["attention_mask"][:, 1:].contiguous()
        adv_loss = chunked_cross_entropy(logits, labels.view(logits.size(0), -1))
        # The adversary wants to MINIMISE loss (keep the model good at forget data
        adv_loss = -(adv_loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)

        handle.remove()  # clean up the hook

        # PGD step: update δ using signed gradients, then clip to L∞ ball
        adv_loss.backward(inputs=[delta])  # only compute grad w.r.t. δ
        with torch.no_grad():
            delta.data = delta.data - lat_eps * delta.grad.sign()  # FGSM-style step
            delta.data = delta.data.clamp(-lat_eps, lat_eps)       # project back to L∞ ball
            delta.grad.zero_()

    # --- Outer loop: train model with the optimal (frozen) perturbation ---
    # The adversary found the best δ; now train the model to resist it.
    delta = delta.detach()  # freeze perturbation (no more adversary updates)
    handle = layers[target_lid].register_forward_hook(make_hook(delta))

    # Forget loss WITH perturbation (gradient ascent — maximize NLL)
    forget_loss = -nll_loss(model, forget_batch)
    handle.remove()

    # Retain loss (standard next-token prediction — no perturbation)
    retain_loss = nll_loss(model, retain_batch)

    return forget_loss + retain_weight * retain_loss


# ---- CB-LAT (combined) -------------------------------------------------
# The most robust method in this file: combines Circuit Breakers with
# Latent Adversarial Training.  The idea is:
#   1. Inner loop (LAT): find adversarial perturbation δ that helps the
#      model recall forget-set data (same as in LAT above)
#   2. Outer loop (CB): with δ injected, apply representation rerouting
#      to steer forget activations toward random noise
#
# This means the model learns to reroute representations even when an
# adversary is actively trying to recover the forgotten knowledge.

def cb_lat_loss(
    model,
    forget_batch: dict,
    retain_batch: dict,
    layer_ids: list[int],
    random_targets: dict,
    retain_targets: dict,
    steering_coeff: float,
    alpha: float,
    lat_eps: float,
    lat_steps: int,
) -> torch.Tensor:
    """
    CB-LAT: Circuit Breakers + Latent Adversarial Training.
    Inner loop: find adversarial perturbation that helps model recall forget data.
    Outer loop: apply CB representation rerouting with perturbation active,
                so model unlearns even under adversarial pressure.
    """
    device = next(model.parameters()).device

    # Find target layer module
    target_lid = layer_ids[len(layer_ids) // 2]
    layers = None
    for attr in ["model.layers", "gpt_neox.layers", "transformer.h"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            layers = obj
            break
        except AttributeError:
            continue

    if layers is None:
        # Fallback to plain CB
        print(f"[WARNING] CB-LAT: Could not find model layers structure. Falling back to plain Circuit Breakers.")
        print(f"[WARNING] CB-LAT: This may happen with non-standard model architectures.")
        return cb_loss(model, forget_batch, retain_batch, layer_ids,
                       random_targets, retain_targets, steering_coeff, alpha)

    # Get hidden shape
    with torch.no_grad():
        out = model(input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"],
                    output_hidden_states=True)
        hidden_shape = out.hidden_states[target_lid + 1].shape

    # Inner loop: PGD adversarial perturbation (identical to LAT inner loop)
    model_dtype = next(model.parameters()).dtype
    delta = torch.zeros(hidden_shape, device=device, dtype=model_dtype, requires_grad=True)

    def make_hook(d):
        """Hook that injects perturbation `d` into a layer’s output."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return ((output[0] + d).to(output[0].dtype),) + output[1:]
            return (output + d).to(output.dtype)
        return hook_fn

    for _ in range(lat_steps):
        handle = layers[target_lid].register_forward_hook(make_hook(delta))
        out = model(input_ids=forget_batch["input_ids"],
                    attention_mask=forget_batch["attention_mask"])
        logits = out.logits[:, :-1, :].contiguous()
        labels = forget_batch["input_ids"][:, 1:].contiguous()
        mask = forget_batch["attention_mask"][:, 1:].contiguous()
        adv_loss = chunked_cross_entropy(logits, labels.view(logits.size(0), -1))
        adv_loss = -(adv_loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
        handle.remove()
        adv_loss.backward(inputs=[delta])
        with torch.no_grad():
            delta.data = delta.data - lat_eps * delta.grad.sign()
            delta.data = delta.data.clamp(-lat_eps, lat_eps)
            delta.grad.zero_()

    # Outer loop: CB rerouting with the adversarial perturbation active
    # This is the key difference from plain CB: the forget activations have
    # been perturbed by the adversary, so the model must reroute *despite* that.
    delta = delta.detach()
    handle = layers[target_lid].register_forward_hook(make_hook(delta))

    # Get perturbed forget activations (with δ injected at the target layer)
    forget_acts = get_layer_activations(model, forget_batch, layer_ids)
    handle.remove()

    # Retain activations (clean, no perturbation — we only attack forget data)
    retain_acts = get_layer_activations(model, retain_batch, layer_ids)

    # CB loss computation: same as cb_loss() but using perturbed forget acts
    loss = torch.tensor(0.0, device=device, dtype=next(model.parameters()).dtype)
    for lid in layer_ids:
        # Forget: push perturbed activations toward random noise direction
        fa = forget_acts[lid].flatten(0, 1)
        rt = random_targets[lid].unsqueeze(0).expand_as(fa) * steering_coeff
        cos_sim = F.cosine_similarity(fa, rt, dim=-1)
        loss = loss - cos_sim.mean()

        # Retain: preserve original activation directions
        ra = retain_acts[lid]
        tr = retain_targets[lid].to(ra.device)  # move cached activations to current device
        bsz = min(ra.size(0), tr.size(0))
        retain_cos = F.cosine_similarity(
            ra[:bsz].flatten(0, 1), tr[:bsz].detach().flatten(0, 1), dim=-1)
        loss = loss + alpha * (1.0 - retain_cos.mean())

    return loss


# ---- Weight Distortion ---------------------------------------------------
# The simplest weight-space method.  Before training even begins, we add
# random Gaussian noise to ALL model parameters (controlled by --wt-noise-std).
# Then we fine-tune on the retain set only.  The intuition is:
#   - The noise destroys some of the model’s learned associations (including
#     forget-set knowledge)
#   - The retain fine-tuning recovers useful capabilities while leaving
#     the forget-set knowledge degraded
#
# The loss during training is just standard NLL on retain data.

def wt_dist_loss(model, retain_batch: dict) -> torch.Tensor:
    """Weight Distortion: just retain NLL (noise was added to weights before training)."""
    return nll_loss(model, retain_batch)


# ---- Weight Distance Regularization --------------------------------------
# Instead of adding noise upfront (like wt_dist), this method drives the
# model weights AWAY from their pretrained values by maximising L2 distance:
#   L = NLL_retain - λ * ||θ - θ_pretrained||²
#
# The NLL_retain term keeps the model useful on retain data, while the
# L2 penalty pushes the weights to change, hopefully destroying the
# specific associations that encode forget-set knowledge.
# λ (reg_lambda) controls how aggressively the weights are pushed apart.

def wt_dist_reg_loss(
    model,
    retain_batch: dict,
    pretrained_params: dict,  # {name: frozen tensor} — cached before training
    reg_lambda: float,
) -> torch.Tensor:
    """
    Weight Distance Regularization:
    Minimize retain NLL while MAXIMIZING L2 distance from pretrained weights.
    L = NLL_retain - λ * ||θ - θ_pretrained||_2^2
    """
    retain_nll = nll_loss(model, retain_batch)

    # Compute sum of squared differences between current and original weights
    l2_dist = torch.tensor(0.0, device=next(model.parameters()).device,
                           dtype=next(model.parameters()).dtype)
    for name, param in model.named_parameters():
        if param.requires_grad and name in pretrained_params:
            l2_dist = l2_dist + (param - pretrained_params[name]).pow(2).sum()

    # Subtract the L2 term: minimising this loss = retain quality UP, distance UP
    return retain_nll - reg_lambda * l2_dist


# ---- Task Arithmetic Removal (TAR) --------------------------------------
# TAR uses task arithmetic: θ_unlearned = θ_base - α(θ_forget_ft - θ_base)
# where θ_forget_ft is obtained by fine-tuning the base model on forget data.
# This is a one-time operation, not iterative training.

def apply_tar(model, forget_batches, alpha, lr, epochs, device, pt_dtype=None, args=None):
    """
    Apply Task Arithmetic Removal to the model using HF Trainer for robustness.

    Steps:
    1. Cache the original model weights (θ_base)
    2. Fine-tune on forget data to get θ_forget_ft using HF Trainer
    3. Compute task vector: τ = θ_forget_ft - θ_base
    4. Apply: θ_unlearned = θ_base - α * τ

    Args:
        model: The base model to modify in-place
        forget_batches: Batches of forget data for fine-tuning
        alpha: Scaling factor for task vector subtraction
        lr: Learning rate for forget fine-tuning
        epochs: Number of epochs for forget fine-tuning
        device: Device for computation
        pt_dtype: Data type for training (for mixed precision)
        args: Full args object for output directory
    """
    print(f"[TAR] Starting Task Arithmetic Removal (alpha={alpha})")

    # Step 1: Cache original weights (store on CPU to save GPU memory)
    print("[TAR] Caching original model weights...")
    original_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Create empty CPU tensor with same shape and dtype, then copy
            # This avoids allocating GPU memory for the clone
            cpu_tensor = torch.empty(param.shape, dtype=param.dtype, device='cpu')
            cpu_tensor.copy_(param.data, non_blocking=True)
            original_weights[name] = cpu_tensor

    # Step 2: Fine-tune on forget data using HF Trainer
    print(f"[TAR] Fine-tuning on forget data ({epochs} epochs, lr={lr}) using HF Trainer...")

    # Import trainer components
    import sys
    _unlearn_dir = os.path.dirname(os.path.abspath(__file__))
    if _unlearn_dir not in sys.path:
        sys.path.insert(0, _unlearn_dir)
    from trainer import UnlearningTrainer, UnlearningDataset, UnlearningCollator
    from transformers import TrainingArguments

    # Prepare dataset - only forget data for TAR
    # TAR only needs forget data, but UnlearningDataset expects pairs
    # We'll duplicate forget data as "retain" to satisfy the interface
    dataset = UnlearningDataset(
        forget_batches=forget_batches,
        retain_batches=forget_batches,  # Dummy - TAR won't use this
    )

    # Determine if we can use bf16
    if pt_dtype is None:
        pt_dtype = torch.float32  # Default to fp32 if not specified
    use_bf16 = (pt_dtype == torch.bfloat16) and torch.cuda.is_available()
    use_fp16 = False

    # Setup training arguments - match the main training loop's format
    training_args = TrainingArguments(
        output_dir=args.outdir if args else "/tmp/tar_training",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # Batches are already prepared
        learning_rate=lr,
        warmup_ratio=0.0,  # TAR doesn't use warmup
        lr_scheduler_type="cosine",
        max_grad_norm=0.0,  # TAR typically doesn't need gradient clipping
        gradient_accumulation_steps=1,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",  # Don't save checkpoints for TAR
        eval_strategy="no",  # No evaluation during TAR training
        push_to_hub=False,
        report_to="none",  # TAR doesn't need wandb logging
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # Use UnlearningTrainer for TAR - it handles device movement correctly
    # TAR just needs standard fine-tuning, which is the opposite of ga_simple
    # ga_simple does: return -nll_loss(forget_batch)  (gradient ascent)
    # TAR needs:      return nll_loss(forget_batch)   (gradient descent)
    # But ga_simple negates the loss, so we can't use it directly.
    # Let's use wt_dist which just does standard fine-tuning on retain data
    # but we'll pass forget data as both forget and retain

    class TARArgs:
        def __init__(self):
            self.method = "wt_dist"  # wt_dist just does nll_loss on retain data

    tar_args = TARArgs()

    # Create trainer using UnlearningTrainer for proper device handling
    trainer = UnlearningTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=UnlearningCollator(),
        unlearn_args=tar_args,
        ref_model=None,
        random_targets=None,
        retain_act_cache=None,
        layer_ids=None,
    )

    # Train
    print(f"[TAR] Starting fine-tuning with HF Trainer (mixed-precision bf16: {use_bf16})...")
    trainer.train()

    print(f"[TAR] Fine-tuning complete")

    # Step 3: Compute task vector (τ = θ_forget_ft - θ_base)
    print("[TAR] Computing task vector...")
    task_vectors = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in original_weights:
            task_vectors[name] = param.data - original_weights[name].to(param.device)

    # Step 4: Apply TAR update (θ_unlearned = θ_base - α * τ)
    print(f"[TAR] Applying task arithmetic (alpha={alpha})...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in task_vectors:
                param.data = original_weights[name].to(param.device) - alpha * task_vectors[name]

    print("[TAR] Task Arithmetic Removal completed")


# ===================================================================
# Validation function
# ===================================================================
# Periodically check how well unlearning is working by measuring NLL
# on held-out forget and retain samples.  Good unlearning means:
#   - HIGH forget_NLL  (model is bad at predicting forget data)
#   - LOW retain_NLL   (model is still good at predicting retain data)
#   - Positive "gap" (forget_NLL - retain_NLL)
# ===================================================================

def run_validation(model, eval_forget_batches, eval_retain_batches, epoch, step, device, verbose=True):
    """Run validation on held-out data and return metrics.

    Returns a dict with forget_nll, retain_nll, and the gap between them.
    The gap should be positive and increasing for successful unlearning.
    """
    if not eval_forget_batches or not eval_retain_batches:
        return None

    model.eval()  # switch to eval mode (disables dropout, etc.)
    with torch.no_grad():
        # Move eval batches to device when needed
        forget_nll = sum(nll_loss(model, {k: v.to(device) for k, v in b.items()}).item() for b in eval_forget_batches) / len(eval_forget_batches)
        retain_nll = sum(nll_loss(model, {k: v.to(device) for k, v in b.items()}).item() for b in eval_retain_batches) / len(eval_retain_batches)
    model.train()  # switch back to training mode

    gap = forget_nll - retain_nll

    if verbose:
        print(f"\n[VAL @ epoch {epoch+1}, step {step}]  forget_NLL={forget_nll:.4f}  retain_NLL={retain_nll:.4f}  gap={gap:.4f}")

    return {"forget_nll": forget_nll, "retain_nll": retain_nll, "gap": gap}


# Minimum free GPU VRAM (in bytes) required to attempt CUDA eval.
# If no GPU clears this bar we fall back to CPU.
_MIN_FREE_VRAM_FOR_EVAL = 10 * 1024 ** 3  # 10 GiB


def _pick_eval_device(requested_device: str) -> str:
    """Return a safe device for eval.py.

    If the caller asked for 'cpu' or a specific non-CUDA device, respect it.
    Otherwise check how much free VRAM is available *right now* (after the
    training model has been deleted and caches cleared).  If the best GPU
    has less than _MIN_FREE_VRAM_FOR_EVAL bytes free, fall back to CPU to
    avoid an immediate OOM during model reload inside eval.py.
    """
    if not torch.cuda.is_available() or requested_device == "cpu":
        return requested_device

    n = torch.cuda.device_count()
    best_free = 0
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            best_free = max(best_free, free)
        except Exception:
            pass

    if best_free < _MIN_FREE_VRAM_FOR_EVAL:
        free_gib = best_free / 1024 ** 3
        print(
            f"[unlearn] WARNING: Best GPU only has {free_gib:.1f} GiB free "
            f"(need ≥ {_MIN_FREE_VRAM_FOR_EVAL / 1024**3:.0f} GiB). "
            f"Falling back to --device cpu for eval."
        )
        return "cpu"

    return requested_device


def run_evaluation_benchmarks(outdir, device, dtype, no_eval=False):
    """Run evaluation benchmarks on the unlearned model.

    Args:
        outdir: Path to the model directory
        device: Device to run evaluation on
        dtype: Data type for evaluation
        no_eval: If True, skip evaluation

    Returns:
        True if evaluation succeeded, False otherwise
    """
    if no_eval:
        print("[unlearn] Skipping eval benchmarks (--no-eval specified)")
        return True

    print("[unlearn] Running eval benchmarks ...")
    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment", "eval.py")
    eval_device = _pick_eval_device(device)
    eval_cmd = [
        "uv", "run", "--script", eval_script,
        "--model", outdir,
        "--device", eval_device,
        "--dtype", dtype,
    ]

    try:
        import subprocess
        result = subprocess.run(eval_cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print("[unlearn] Eval complete ✓")

            # Log eval metrics to W&B
            from utils import model_outdir
            eval_summary_path = os.path.join(
                model_outdir(outdir, suffix="evals"), "summary.json"
            )
            if os.path.exists(eval_summary_path):
                import json
                with open(eval_summary_path) as f:
                    eval_data = json.load(f)
                try:
                    import wandb
                    if wandb.run is not None:
                        eval_results = eval_data.get("results", {})
                        flat = {}
                        for task, metrics in eval_results.items():
                            for metric_key, value in metrics.items():
                                if metric_key.endswith(",none") and not metric_key.startswith("alias"):
                                    clean = metric_key.replace(",none", "")
                                    flat[f"eval_bench/{task}/{clean}"] = value
                        wandb.log(flat)
                        wandb.summary.update(flat)
                        print(f"[unlearn] Eval metrics logged to W&B ✓ ({len(flat)} metrics)")
                except ImportError:
                    pass
            return True
        else:
            print(f"[unlearn] WARNING: eval returned exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"[unlearn] WARNING: eval failed: {e}")
        return False


# ===================================================================
# Output directory auto-generation
# ===================================================================
# The output folder name encodes the method plus all hyperparameters
# relevant to that method, so different runs never overwrite each other.
#
# Example:
#   unlearned_models/EleutherAI_deep-ignorance-unfiltered__cb_lat__ep2_lr5e-06_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7
# ===================================================================

# Which parameters are relevant for each method
METHOD_PARAMS: dict[str, list[str]] = {
    "ga_simple":    ["epochs", "lr", "batch_size", "max_lines"],
    "ga":           ["epochs", "lr", "batch_size", "retain_weight", "max_lines"],
    "grad_diff":    ["epochs", "lr", "batch_size", "forget_weight", "max_lines"],
    "dpo":          ["epochs", "lr", "batch_size", "beta", "max_lines"],
    "npo":          ["epochs", "lr", "batch_size", "beta", "retain_weight", "max_lines"],
    "simnpo":       ["epochs", "lr", "batch_size", "beta", "retain_weight", "max_lines"],
    "rmu":          ["epochs", "lr", "batch_size", "alpha", "steering_coeff", "layer_id", "max_lines"],
    "cb":           ["epochs", "lr", "batch_size", "alpha", "steering_coeff", "layer_id", "max_lines"],
    "lat":          ["epochs", "lr", "batch_size", "lat_eps", "lat_steps", "retain_weight", "layer_id", "max_lines"],
    "cb_lat":       ["epochs", "lr", "batch_size", "alpha", "steering_coeff", "lat_eps", "lat_steps", "layer_id", "max_lines"],
    "tar":          ["tar_alpha", "tar_lr", "tar_epochs", "max_lines"],
    "wt_dist":      ["epochs", "lr", "batch_size", "wt_noise_std", "max_lines"],
    "wt_dist_reg":  ["epochs", "lr", "batch_size", "wt_reg_lambda", "max_lines"],
}

# Short abbreviations for folder name suffixes
PARAM_ABBREV: dict[str, str] = {
    "epochs": "ep",
    "lr": "lr",
    "batch_size": "bs",
    "max_lines": "ml",
    "retain_weight": "rw",
    "forget_weight": "fw",
    "beta": "b",
    "alpha": "a",
    "steering_coeff": "sc",
    "layer_id": "ly",
    "lat_eps": "le",
    "lat_steps": "ls",
    "tar_alpha": "ta",
    "tar_lr": "tlr",
    "tar_epochs": "tep",
    "wt_noise_std": "wn",
    "wt_reg_lambda": "wr",
}


def build_outdir(args) -> str:
    """Build the output directory path from the method and its relevant parameters."""
    method = args.method

    # Build parameter suffix from all relevant params for this method
    parts = []
    for param in METHOD_PARAMS[method]:
        abbrev = PARAM_ABBREV[param]
        value = getattr(args, param)
        # layer_id is a string like "5,6,7" — use dashes for folder safety
        if param == "layer_id":
            value = str(value).replace(",", "-")
        parts.append(f"{abbrev}{value}")

    suffix = "_".join(parts)
    return model_outdir(args.model, root="unlearned_models", suffix=f"{method}__{suffix}")


# ===================================================================
# Main training loop
# ===================================================================
# The main() function orchestrates the full unlearning pipeline:
#   1. Parse CLI args to select method + hyperparameters
#   2. Load the pre-trained model + tokenizer
#   3. Prepare forget/retain datasets
#   4. Run method-specific setup (e.g., cache activations for RMU)
#   5. Train for the specified number of epochs
#   6. Evaluate unlearning quality on held-out data
#   7. Save the unlearned model
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-method LLM Unlearning")
    parser.add_argument("--model", required=True, help="Base model (HF ID or local path)")
    parser.add_argument(
        "--method",
        required=True,
        choices=["ga_simple", "ga", "grad_diff", "dpo", "npo", "simnpo", "rmu", "cb", "lat", "cb_lat", "tar", "wt_dist", "wt_dist_reg"],
        help="Unlearning method",
    )
    parser.add_argument("--forget-data", default="data/forget.txt")
    parser.add_argument("--retain-data", default="data/retain.txt")
    parser.add_argument("--max-lines", type=int, default=1024,
                        help="Maximum number of lines to load from each dataset (default: 1024 for fast sweeps, use 0 for unlimited)")
    # --outdir is auto-generated from the method and hyperparameters (see build_outdir)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1, help="Inverse temp for NPO/SimNPO/DPO")
    parser.add_argument("--alpha", type=float, default=100.0, help="Retain weight for RMU")
    parser.add_argument("--steering-coeff", type=float, default=20.0, help="Steering coeff for RMU")
    parser.add_argument(
        "--layer-id",
        default="5,6,7",
        help="Comma-separated target layer indices for RMU/CB/LAT",
    )
    parser.add_argument("--forget-weight", type=float, default=1.0,
                        help="Weight for forget loss in GradDiff")
    parser.add_argument("--lat-eps", type=float, default=0.1,
                        help="Perturbation budget for LAT")
    parser.add_argument("--lat-steps", type=int, default=5,
                        help="Number of adversarial inner steps for LAT")
    parser.add_argument("--retain-weight", type=float, default=1.0,
                        help="Multiplier for retain loss in ga/npo/simnpo/lat (higher = more stable, less forgetting)")
    parser.add_argument("--tar-alpha", type=float, default=1.0,
                        help="Scaling factor for TAR task vector subtraction")
    parser.add_argument("--tar-lr", type=float, default=1e-5,
                        help="Learning rate for TAR forget fine-tuning phase")
    parser.add_argument("--tar-epochs", type=int, default=1,
                        help="Number of epochs for TAR forget fine-tuning phase")
    parser.add_argument("--wt-noise-std", type=float, default=0.02,
                        help="Std of Gaussian noise for Weight Distortion (wt_dist)")
    parser.add_argument("--wt-reg-lambda", type=float, default=0.1,
                        help="Regularizer weight for Weight Dist Reg (wt_dist_reg)")
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction of data to hold out for evaluation (0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps for effective larger batch size")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Evaluate every N steps during training (0 to disable)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Upload finished model to HuggingFace (requires HF_TOKEN env var)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the final model weights to disk (useful for sweeps to save space)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip benchmark evaluation after training (useful when pushing to HuggingFace)")
    parser.add_argument("--check-wandb-only", action="store_true",
                        help="Check if this specific run configuration already finished in W&B and exit 0 if so, 1 otherwise.")
    args = parser.parse_args()

    # Auto-generate output directory from method + all relevant params
    args.outdir = build_outdir(args)
    
    # If we are just checking idempotency, do it now before allocating any GPU RAM
    if args.check_wandb_only:
        import utils
        full_display_name = utils._derive_run_name("unlearn", args)

        # Eval keys that analyze_runs.py expects — at least one must be present
        # for the run to be considered truly complete (not just "finished" but
        # crashed before the eval subprocess had a chance to log results).
        _EVAL_KEYS = [
            "eval_bench/mmlu/acc",
            "eval_bench/wmdp_bio_robust/acc",
            "eval_bench/wmdp_bio_cloze_verified/acc_norm",
            "eval_bench/wmdp_bio_categorized_mcqa/acc",
        ]

        try:
            import wandb
            api = wandb.Api()
            project_name = os.environ.get("WANDB_PROJECT", "cambridge_era")
            runs = api.runs(f"{project_name}", filters={"display_name": full_display_name})
            for r in runs:
                if r.state != "finished":
                    continue
                has_eval = any(r.summary.get(k) is not None for k in _EVAL_KEYS)
                if has_eval:
                    print(f"[unlearn] Idempotency check: Run '{full_display_name}' already finished with eval results. Skipping.")
                    sys.exit(0)
                else:
                    print(f"[unlearn] Idempotency check: Run '{full_display_name}' finished but missing eval results. Will re-run.")
        except Exception as e:
            # If W&B API drops or isn't authed, ignore and assume we need to run
            pass

        # If we didn't find a finished run with eval results, signal that training should proceed
        print(f"[unlearn] Idempotency check: Run '{full_display_name}' not found or incomplete. Needs training.")
        sys.exit(1)

    print(f"[unlearn] Output directory: {args.outdir}")

    # ---- W&B ----
    from utils import init_wandb, finish_wandb
    run = init_wandb("unlearn", args)

    # Log method-specific hyperparameters as a dedicated config group
    if run is not None:
        import wandb
        hyperparameters = {
            "model": args.model,
            "method": args.method,
            "outdir": args.outdir,
            "max_length": args.max_length,
            "seed": args.seed,
            "grad_clip": args.grad_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "eval_split": args.eval_split,
            "eval_interval": args.eval_interval,
            "forget_data": args.forget_data,
            "retain_data": args.retain_data,
        }
        for param in METHOD_PARAMS[args.method]:
            hyperparameters[param] = getattr(args, param)
        wandb.config.update({"hyperparameters": hyperparameters})

    # ---- Setup ----
    device = resolve_device(args.device)
    pt_dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"[unlearn] method={args.method}  device={device}  dtype={pt_dtype}")
    print(f"[unlearn] model={args.model}")
    print(f"[unlearn] forget={args.forget_data}  retain={args.retain_data}")

    # Log optimization settings for reproducibility
    if args.grad_accum_steps > 1:
        print(f"[unlearn] WARNING: Gradient accumulation enabled (steps={args.grad_accum_steps})")
        print(f"[unlearn]          Effective batch size = {args.batch_size * args.grad_accum_steps}")
        print(f"[unlearn]          This may affect convergence compared to true larger batch training")
    if args.grad_clip == 0:
        print(f"[unlearn] WARNING: Gradient clipping disabled - training may be less stable")

    # ---- Load tokenizer ----
    print("[unlearn] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load base model ----
    print(f"[unlearn] Loading base model: {args.model}")

    # If device is auto, let accelerate distribute it across available GPUs.
    # 1. Restrict CUDA_VISIBLE_DEVICES to GPUs with enough free VRAM so that
    #    device_map='auto' never places layers on a near-full GPU.
    # 2. Compute a per-GPU max_memory budget that leaves headroom for fp32
    #    optimizer states (~6× bf16 params) and forward-pass activations, so
    #    accelerate is forced to spread weights across all visible GPUs rather
    #    than packing everything onto the first one.
    if args.device == "auto" and torch.cuda.is_available():
        usable = filter_gpus_by_free_vram(min_free_gib=10.0)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in usable)
        print(f"[unlearn] Restricting to GPUs with ≥10 GiB free: {usable}")

        # Scale activation buffer with batch size (larger batches need more headroom).
        activation_buf_gib = max(8.0, args.batch_size * 0.4)
        mm = compute_training_max_memory(
            optimizer_state_multiplier=6.0,
            activation_buffer_gib=activation_buf_gib,
        )
        device_map_kwargs = {"device_map": "auto", **({"max_memory": mm} if mm else {})}
    else:
        device_map_kwargs = {}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=pt_dtype, trust_remote_code=True, **device_map_kwargs
    )
    if args.device != "auto":
        model.to(device)
    model.train()

    # Enable gradient checkpointing to save GPU memory at the cost of
    # recomputing activations during the backward pass.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ---- Load reference model (for DPO / NPO) ----
    # DPO and NPO compare the trainable "policy" model against a frozen
    # "reference" model (an unmodified copy of the pretrained weights).
    # This prevents the policy from drifting too far during unlearning.
    ref_model = None
    if args.method in ("dpo", "npo"):
        print("[unlearn] Loading reference model (frozen copy)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=pt_dtype, trust_remote_code=True, **device_map_kwargs
        )
        if args.device != "auto":
            ref_model.to(device)
        ref_model.eval()  # permanently in eval mode
        for p in ref_model.parameters():
            p.requires_grad = False  # freeze all weights

    # ---- Load and tokenize data ----
    print("[unlearn] Tokenizing data...")
    max_lines = args.max_lines if args.max_lines > 0 else None
    forget_texts = load_lines(args.forget_data, max_lines)
    retain_texts = load_lines(args.retain_data, max_lines)
    print(f"[unlearn]   forget samples: {len(forget_texts)}")
    print(f"[unlearn]   retain samples: {len(retain_texts)}")

    # ---- Train / eval split ----
    # Hold out a fraction of data for validation so we can monitor
    # unlearning quality during training (forget NLL should go UP,
    # retain NLL should stay LOW).
    eval_forget_batches = []
    eval_retain_batches = []
    if args.eval_split > 0:
        random.seed(args.seed)
        random.shuffle(forget_texts)
        random.shuffle(retain_texts)
        n_f_eval = max(1, int(len(forget_texts) * args.eval_split))
        n_r_eval = max(1, int(len(retain_texts) * args.eval_split))
        eval_forget_texts = forget_texts[:n_f_eval]
        eval_retain_texts = retain_texts[:n_r_eval]
        forget_texts = forget_texts[n_f_eval:]
        retain_texts = retain_texts[n_r_eval:]
        # Keep eval data on CPU as well to avoid GPU OOM
        eval_forget_items = tokenize_texts(eval_forget_texts, tokenizer, args.max_length, "cpu")
        eval_retain_items = tokenize_texts(eval_retain_texts, tokenizer, args.max_length, "cpu")
        eval_forget_batches = make_batches(eval_forget_items, args.batch_size)
        eval_retain_batches = make_batches(eval_retain_items, args.batch_size)
        print(f"[unlearn]   eval split: {n_f_eval} forget, {n_r_eval} retain")
        print(f"[unlearn]   train: {len(forget_texts)} forget, {len(retain_texts)} retain")

    # Keep tokenized data on CPU to avoid GPU OOM, move batches to device when needed
    forget_items = tokenize_texts(forget_texts, tokenizer, args.max_length, "cpu")
    retain_items = tokenize_texts(retain_texts, tokenizer, args.max_length, "cpu")

    forget_batches = make_batches(forget_items, args.batch_size)
    retain_batches = make_batches(retain_items, args.batch_size)

    # Each step pairs one forget batch with one retain batch, so the total
    # steps per epoch is limited by whichever dataset has fewer batches.
    n_steps = min(len(forget_batches), len(retain_batches))
    print(f"[unlearn]   steps/epoch: {n_steps}")

    # ---- RMU / CB / LAT setup: cache retain activations + random targets ----
    layer_ids = [int(x) for x in args.layer_id.split(",")]
    random_targets = {}
    retain_act_cache: list[dict] = []

    if args.method in ("rmu", "cb", "lat", "cb_lat"):
        n_layers = model.config.num_hidden_layers
        bad = [lid for lid in layer_ids if lid < 0 or lid >= n_layers]
        if bad:
            sys.exit(
                f"[unlearn] ERROR: --layer-id values {bad} out of range for "
                f"this model ({n_layers} layers, valid 0..{n_layers - 1})"
            )
        print(f"[unlearn] {args.method.upper()}: target layers={layer_ids}  (model has {n_layers})")

    if args.method in ("rmu", "cb", "cb_lat"):
        if n_steps == 0:
            sys.exit("[unlearn] ERROR: No training steps available (n_steps=0)")

        hidden_dim = model.config.hidden_size

        # Generate a random unit vector per target layer.  During training,
        # RMU/CB will push forget-set activations to align with these vectors.
        # Normalising ensures the direction is what matters, not magnitude.
        for lid in layer_ids:
            random_targets[lid] = torch.randn(hidden_dim, device=device, dtype=pt_dtype)
            random_targets[lid] = random_targets[lid] / random_targets[lid].norm()  # unit norm

        # Cache the retain-set activations from the ORIGINAL (pre-training)
        # model.  These become the targets that RMU/CB try to preserve.
        print(f"[unlearn] {args.method.upper()}: caching retain activations...")
        model.eval()
        with torch.no_grad():
            for rb in retain_batches[:n_steps]:
                # Move batch to device for activation computation
                rb_device = {k: v.to(device) for k, v in rb.items()}
                acts = get_layer_activations(model, rb_device, layer_ids)
                # Cache activations on CPU to avoid GPU OOM
                retain_act_cache.append({lid: a.detach().cpu().to(pt_dtype) for lid, a in acts.items()})
        model.train()

    # ---- Weight Distortion: add Gaussian noise to all parameters ----
    if args.method == "wt_dist":
        print(f"[unlearn] WT_DIST: adding Gaussian noise (std={args.wt_noise_std}) to all parameters...")
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * args.wt_noise_std)

    # ---- Weight Dist Reg: cache frozen copy of pretrained parameters ----
    pretrained_params = {}
    if args.method == "wt_dist_reg":
        print("[unlearn] WT_DIST_REG: caching pretrained parameters for L2 regularization...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                pretrained_params[name] = param.data.clone()

    # ---- Task Arithmetic Removal (TAR) ----
    # TAR is a one-time operation, not iterative training
    if args.method == "tar":
        apply_tar(model, forget_batches, args.tar_alpha, args.tar_lr, args.tar_epochs, device, pt_dtype, args)

        # After TAR is complete, we're done - no need for further training
        print("[unlearn] TAR completed. Skipping iterative training phase.")

        # Run final validation if eval data is available
        if eval_forget_batches and eval_retain_batches:
            final_metrics = run_validation(model, eval_forget_batches, eval_retain_batches, 0, 0, device)
            print(f"[FINAL] TAR metrics: {final_metrics}")

        # Save model and exit early
        model_out_path = Path(args.outdir) / "pytorch_model.bin"
        model_out_path.parent.mkdir(parents=True, exist_ok=True)

        # Always save for evaluation, then clean up later if NO_SAVE
        model.save_pretrained(args.outdir)
        tokenizer.save_pretrained(args.outdir)
        if args.no_save:
            print(f"[unlearn] TAR model saved temporarily for evaluation (will be deleted after due to --no-save): {args.outdir}")
        else:
            print(f"[unlearn] TAR model saved to: {args.outdir}")

        # Clear GPU memory before evaluation to avoid OOM
        print("[unlearn] Clearing GPU memory before evaluation...")
        del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        import gc; gc.collect()   # Python garbage collection

        # Set PyTorch memory allocator to avoid fragmentation
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

        # Run evaluation if needed
        run_evaluation_benchmarks(args.outdir, device, args.dtype, args.no_eval)

        # Clean up weights if --no-save (same as main path)
        if args.no_save:
            import glob
            print("\n[unlearn] Cleaning up TAR model weights (--no-save specified) to save disk space...")
            for ext in ["*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json", "*.pt"]:
                for f in glob.glob(os.path.join(args.outdir, ext)):
                    try:
                        os.remove(f)
                        print(f"[unlearn]   Deleted: {f}")
                    except Exception as e:
                        print(f"[unlearn]   Could not delete {f}: {e}")
            print("[unlearn] Cleanup complete ✓")

        return

    # ---- Train via HF Trainer (mixed-precision bf16) ----
    # The Trainer uses mixed precision when bf16=True:
    #   - Forward pass: bf16 (fast)
    #   - Master weight copy + Adam m/v states: fp32 (precise)
    # This is the key improvement over our old custom loop, which put
    # everything in bf16 (including Adam buffers), losing gradient precision.
    #
    # The Trainer also handles: grad accumulation, grad clipping,
    # cosine LR scheduler, WandB step logging, and multi-GPU via Accelerate.
    print(f"\n[unlearn] Starting training via HF Trainer: {args.epochs} epoch(s), {n_steps} steps/epoch")
    print(f"[unlearn] Mixed-precision bf16: ON (fp32 master weights + fp32 Adam states)\n")

    # trainer.py lives alongside unlearn.py in the unlearn/ directory.
    # Since unlearn.py is run as a uv --script (not an installed package),
    # we add its own directory to sys.path so `import trainer` resolves.
    _unlearn_dir = os.path.dirname(os.path.abspath(__file__))
    if _unlearn_dir not in sys.path:
        sys.path.insert(0, _unlearn_dir)
    from trainer import UnlearningTrainer, UnlearningDataset, UnlearningCollator

    # Determine whether we can actually use bf16
    # (bf16 requires CUDA; fall back to fp32 on CPU/MPS)
    use_bf16 = (pt_dtype == torch.bfloat16) and torch.cuda.is_available()
    use_fp16 = False
    if pt_dtype == torch.bfloat16 and not torch.cuda.is_available():
        print("[unlearn] WARNING: bf16 requested but CUDA not available — running in fp32")

    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        # batch_size=1 here because make_batches() already built the real batches;
        # each dataset item IS a full batch of size args.batch_size.
        per_device_train_batch_size=1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        max_grad_norm=args.grad_clip if args.grad_clip > 0 else 0.0,
        gradient_accumulation_steps=args.grad_accum_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_strategy="steps",
        logging_steps=1,
        # WandB is reported via our existing init_wandb() call above;
        # set report_to="none" so the Trainer doesn't try to re-init it,
        # but we override log() below to forward step metrics ourselves.
        report_to="none",
        save_strategy="no",   # we do our own save after training
        eval_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,  # our batch dicts have non-standard keys
        seed=args.seed,
        disable_tqdm=False,
    )

    dataset = UnlearningDataset(forget_batches, retain_batches)
    trainer = UnlearningTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=UnlearningCollator(),
        unlearn_args=args,
        ref_model=ref_model,
        random_targets=random_targets,
        retain_act_cache=retain_act_cache,
        layer_ids=layer_ids,
    )

    # Wire up pretrained_params for wt_dist_reg (set before train() is called)
    if args.method == "wt_dist_reg":
        trainer.set_pretrained_params(pretrained_params)

    trainer.train()

    # Forward the Trainer's logged metrics to WandB manually
    # (since we used report_to="none" to let our init_wandb() own the run)
    try:
        import wandb
        if wandb.run is not None and trainer.state.log_history:
            for log_entry in trainer.state.log_history:
                step = log_entry.get("step", None)
                metrics = {f"train/{k}": v for k, v in log_entry.items()
                           if k not in ("step", "epoch", "total_flos")}
                if metrics and step is not None:
                    wandb.log(metrics, step=step)
    except Exception:
        pass

    # ---- Post-training NLL evaluation on held-out split ----
    if eval_forget_batches and eval_retain_batches:
        print("[unlearn] Running NLL evaluation on held-out split...")
        model.eval()
        with torch.no_grad():
            # Move eval batches to device when needed
            forget_nll = sum(nll_loss(model, {k: v.to(device) for k, v in b.items()}).item() for b in eval_forget_batches) / len(eval_forget_batches)
            retain_nll = sum(nll_loss(model, {k: v.to(device) for k, v in b.items()}).item() for b in eval_retain_batches) / len(eval_retain_batches)
        print(f"[unlearn] Eval  forget_NLL={forget_nll:.4f}  retain_NLL={retain_nll:.4f}")
        print(f"[unlearn] Eval  gap (forget - retain) = {forget_nll - retain_nll:.4f}")
        print(f"[unlearn]   → Good unlearning: high forget_NLL + low retain_NLL\n")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"eval/forget_nll": forget_nll, "eval/retain_nll": retain_nll,
                           "eval/gap": forget_nll - retain_nll})
                wandb.summary.update({"final_forget_nll": forget_nll, "final_retain_nll": retain_nll,
                                      "final_gap": forget_nll - retain_nll})
        except Exception:
            pass

    # ---- Save ----
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"[unlearn] Saving model to {args.outdir} ...")
    # Disable gradient checkpointing before saving (not serializable)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    if args.no_save:
        print("[unlearn] Model saved temporarily for evaluation (will be deleted after due to --no-save) ✓")
    else:
        print("[unlearn] Model saved ✓")

    # ---- Auto-evaluate the unlearned model ----
    # Clear GPU memory before evaluation to avoid OOM
    print("[unlearn] Clearing GPU memory before evaluation...")
    if 'model' in locals():
        del model
    if 'tokenizer' in locals():
        del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    import gc; gc.collect()   # Python garbage collection

    # Set PyTorch memory allocator to avoid fragmentation
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    run_evaluation_benchmarks(args.outdir, args.device, args.dtype, args.no_eval)

    # ---- Upload to HuggingFace (only if --push-to-hub) ----
    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                username = api.whoami()["name"]
                repo_id = f"{username}/{os.path.basename(args.outdir)}"
                print(f"[unlearn] Uploading to HuggingFace: {repo_id}")
                api.create_repo(repo_id, exist_ok=True)
                api.upload_folder(folder_path=args.outdir, repo_id=repo_id)
                print(f"[unlearn] Upload complete ✓  https://huggingface.co/{repo_id}")
            except Exception as e:
                print(f"[unlearn] WARNING: HF upload failed: {e}")
        else:
            print("[unlearn] WARNING: --push-to-hub specified but HF_TOKEN not set")

    # ---- Clean up weights if --no-save ----
    if args.no_save:
        import glob
        print("\n[unlearn] Cleaning up model weights (--no-save specified) to save disk space...")
        for ext in ["*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json", "*.pt"]:
            for f in glob.glob(os.path.join(args.outdir, ext)):
                try:
                    os.remove(f)
                except OSError:
                    pass
        print("[unlearn] Model weights deleted ✓")

    print("===================================================================")
    print("===================================================================")
    finish_wandb()


if __name__ == "__main__":
    main()
