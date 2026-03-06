"""
Shared utilities for model_diffs analysis scripts.
"""
import csv
import gc
import json
import os
import re
from typing import Dict, List, Optional, Set

import torch


# --- Data utilities ---
def ensure_datasets_exist() -> None:
    """Ensure forget.txt and retain.txt datasets exist, create them if missing."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    forget_path = os.path.join(data_dir, "forget.txt")
    retain_path = os.path.join(data_dir, "retain.txt")

    # Check if both files exist
    if os.path.exists(forget_path) and os.path.exists(retain_path):
        return

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Run create_datasets.py to generate the files
    create_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_datasets.py")
    if os.path.exists(create_script):
        print("[utils] Data files missing. Running create_datasets.py to generate them...")
        import subprocess
        try:
            result = subprocess.run(["uv", "run", create_script],
                                  capture_output=True, text=True, cwd=os.path.dirname(create_script))
            if result.returncode == 0:
                print("[utils] Datasets created successfully ✓")
            else:
                print(f"[utils] Warning: create_datasets.py failed: {result.stderr}")
        except Exception as e:
            print(f"[utils] Warning: Could not run create_datasets.py: {e}")
    else:
        print("[utils] Warning: data files missing and create_datasets.py not found")


# --- Path utilities ---
def model_outdir(model: str, root: str = "outputs", suffix: str = "") -> str:
    """Derive output directory from a HuggingFace model ID.

    E.g. 'EleutherAI/deep-ignorance-unfiltered' → 'outputs/EleutherAI_deep-ignorance-unfiltered'
         with suffix='evals' → 'outputs/EleutherAI_deep-ignorance-unfiltered/evals'
    """
    sanitized = model.replace("/", "_")
    parts = [root, sanitized]
    if suffix:
        parts.append(suffix)
    return os.path.join(*parts)


def comparison_outdir(model_a: str, model_b: str, root: str = "outputs", suffix: str = "") -> str:
    """Derive output directory for a two-model comparison.

    E.g. comparison_outdir('org/base', 'org/filtered', suffix='weight_comparison')
         → 'outputs/org_base__to__org_filtered/weight_comparison'
    """
    san_a = model_a.replace("/", "_")
    san_b = model_b.replace("/", "_")
    parts = [root, f"{san_a}__to__{san_b}"]
    if suffix:
        parts.append(suffix)
    return os.path.join(*parts)


def load_dotenv(path: str = None):
    """Load .env file into environment. No external dependencies needed."""
    if path is None:
        # Look for .env in the same directory as this file
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and not os.environ.get(key):  # Don't override existing env vars
                os.environ[key] = value


# Auto-load .env on import so standalone scripts get HF_TOKEN etc.
load_dotenv()


# --- Tokenizer utilities ---
def setup_tokenizer_padding(tokenizer):
    """Set up tokenizer padding token if not already configured.

    Many models don't have a pad token defined by default. This function
    ensures the tokenizer has a pad token, using the eos_token as fallback.

    Args:
        tokenizer: HuggingFace tokenizer instance

    Returns:
        tokenizer: The same tokenizer instance (modified in-place)
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --- Loss computation utilities ---
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
    import torch.nn.functional as F

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


def nll_loss(model, batch: dict, chunk_size: int = 4) -> torch.Tensor:
    """Standard next-token prediction (causal LM) loss.

    This is the same cross-entropy loss used during normal pre-training.
    For unlearning, we either MINIMIZE it (on retain data) to preserve
    capability, or NEGATE it (on forget data) to degrade capability.

    Args:
        model: The language model to compute loss for
        batch: Dictionary with 'input_ids' and 'attention_mask' tensors
        chunk_size: Batch elements per chunk for memory-efficient computation

    Returns:
        Scalar loss tensor
    """
    # For distributed models, inputs need to be on the same device as the first layer
    # HF models with device_map handle this automatically in their forward()
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
    loss = chunked_cross_entropy(logits, labels.view(logits.size(0), -1), chunk_size=chunk_size)
    loss = (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)
    return loss


def compute_spectral_norm(A: torch.Tensor) -> float:
    """
    Compute spectral norm (largest singular value) using SVD.
    More stable than power iteration but potentially slower.
    """
    if A.numel() == 0 or min(A.shape) == 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(A.float())
        return float(s[0].item()) if len(s) > 0 else 0.0
    except:
        return spectral_norm_power(A)  # Fallback to power iteration


# --- Device / dtype resolution ---
def pick_best_gpu() -> int:
    """Return the index of the CUDA GPU with the most free VRAM.

    Uses torch.cuda.mem_get_info() which is fast (no subprocess).
    Falls back to GPU 0 if only one GPU is present or on any error.
    """
    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    best_idx, best_free = 0, 0
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_idx = i
        except Exception:
            pass
    return best_idx


def filter_gpus_by_free_vram(min_free_gib: float = 10.0) -> list[int] | None:
    """Return GPU indices that have at least *min_free_gib* GiB of free VRAM.

    Returns None when VRAM cannot be queried at all (e.g. MIG mode,
    exclusive-process mode, driver quirks on H200).  Callers should treat
    None as "do not restrict CUDA_VISIBLE_DEVICES".

    Falls back to [best_gpu_index] if GPUs can be queried but none meets the
    threshold, so that training can still proceed.
    """
    if not torch.cuda.is_available():
        return []
    min_free_bytes = int(min_free_gib * 1024 ** 3)
    n = torch.cuda.device_count()
    usable = []
    any_success = False
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            any_success = True
            if free >= min_free_bytes:
                usable.append(i)
        except Exception:
            pass

    if not any_success:
        # Could not query any GPU — MIG mode, exclusive-process mode, etc.
        # Returning None signals to the caller: don't touch CUDA_VISIBLE_DEVICES.
        print(
            "[device] WARNING: Could not query VRAM on any GPU "
            "(MIG/exclusive-process mode?). Skipping CUDA_VISIBLE_DEVICES restriction."
        )
        return None

    if not usable:
        # No GPU has enough free memory — pick the best one and warn.
        # mem_get_info can raise on some systems (MIG mode, exclusive-process
        # mode, driver quirks on H200, etc.) so wrap it defensively.
        best = pick_best_gpu()
        try:
            free_bytes, _ = torch.cuda.mem_get_info(best)
            free_gib = free_bytes / 1024 ** 3
            print(
                f"[device] WARNING: No GPU has ≥ {min_free_gib:.0f} GiB free. "
                f"Best GPU {best} has {free_gib:.1f} GiB free. Using it anyway."
            )
        except Exception:
            print(
                f"[device] WARNING: No GPU has ≥ {min_free_gib:.0f} GiB free "
                f"(could not query free VRAM). Using GPU {best} anyway."
            )
        return [best]
    return usable


def compute_training_max_memory(
    optimizer_state_multiplier: float = 6.0,
    activation_buffer_gib: float = 10.0,
) -> dict | None:
    """Compute per-GPU max_memory budget for from_pretrained(device_map='auto').

    accelerate's device_map='auto' places *model weights* using the max_memory
    budget, but fp32 optimizer states (Adam m + v + fp32 master weights ≈ 6×
    the bf16 parameter bytes) are allocated afterward on the same devices.
    Left unconstrained, accelerate packs all weights onto the fewest GPUs
    possible — optimizer states then overflow at training time.

    This function computes a per-GPU weight budget that leaves room for both
    optimizer states and activations:

        weight_budget = (free_vram - activation_buffer) / (1 + optimizer_multiplier)

    Args:
        optimizer_state_multiplier: bytes of optimizer state per byte of bf16
            model weight. fp32 Adam (m + v + master copy) = 6.0; bf16 Adam = 2.0.
        activation_buffer_gib: GiB to reserve per GPU for activation tensors
            during the forward/backward pass. Scale up for larger batches.

    Returns:
        Dict mapping GPU index → memory string (e.g. {0: "18GiB", 1: "18GiB"})
        for passing to from_pretrained(max_memory=...), or None if CUDA is
        unavailable.
    """
    if not torch.cuda.is_available():
        return None

    activation_buffer_bytes = int(activation_buffer_gib * 1024 ** 3)
    n = torch.cuda.device_count()
    max_memory = {}

    for i in range(n):
        try:
            _, total = torch.cuda.mem_get_info(i)
            # Use *total* VRAM (not free) as the budget ceiling.
            # filter_gpus_by_free_vram() already ensures the GPU has enough
            # headroom; using free VRAM here would make the budget tiny when
            # other processes are present, causing accelerate to spill weights
            # to disk (catastrophic for training speed).
            # Instead: weight_budget = total / (1 + optimizer_mult)
            # Subtract activation buffer from total first so forward-pass
            # tensors have headroom too.
            usable = max(0, total - activation_buffer_bytes)
            weight_budget_bytes = int(usable / (1.0 + optimizer_state_multiplier))
            weight_budget_gib = max(1, weight_budget_bytes // (1024 ** 3))
            max_memory[i] = f"{weight_budget_gib}GiB"
        except Exception:
            pass

    if not max_memory:
        return None

    indices = list(max_memory.keys())
    budgets = [max_memory[i] for i in indices]
    print(f"[device] max_memory per GPU (for device_map='auto'): "
          + ", ".join(f"GPU{i}={b}" for i, b in zip(indices, budgets)))
    return max_memory


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to the best available (cuda:N > mps > cpu).

    When multiple CUDA GPUs are present, picks the one with the most free
    VRAM so the model doesn't land on an already-busy GPU.
    """
    if device != "auto":
        resolved = device
    elif torch.cuda.is_available():
        best = pick_best_gpu()
        resolved = f"cuda:{best}"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        resolved = "mps"
    else:
        resolved = "cpu"
    print(f"[device] Using device: {resolved}" + (f" (resolved from '{device}')" if device == "auto" else ""))
    return resolved


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Resolve 'auto' dtype based on device, or parse explicit dtype string."""
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        if device == "mps":
            return torch.float16
        return torch.float32
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype '{dtype}'. Use auto|fp32|fp16|bf16")
    return mapping[dtype]


# --- Parameter name parsing ---
LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]
COARSE_ATTN_KEYS = ("attn", "attention", "self_attn")
COARSE_MLP_KEYS = ("mlp", "ffn", "feed_forward", "intermediate")


def extract_layer(param_name: str) -> Optional[int]:
    """Extract layer number from parameter name (e.g., 'layers.15.mlp' -> 15)."""
    for pat in LAYER_PATTERNS:
        m = pat.search(param_name)
        if m:
            return int(m.group(1))
    return None


def classify_coarse(param_name: str) -> str:
    # UNUSED AS WE DO THINGS MORE GRANULARLY NOW, but keeping for comparison if necessary in the future.
    """Classify parameter into coarse groups: 'attn', 'mlp', or 'other'."""
    s = param_name.lower()
    if any(k in s for k in COARSE_ATTN_KEYS):
        return "attn"
    if any(k in s for k in COARSE_MLP_KEYS):
        return "mlp"
    return "other"


# Component-level classification (granular sub-component within attn / mlp)
# Maps param name fragments → component label
_COMP_RULES = [
    # Attention QKV (fused or separate)
    ("attention.query_key_value", "qkv"),
    ("self_attn.qkv", "qkv"),
    ("self_attn.q_proj", "qkv"),
    ("self_attn.k_proj", "qkv"),
    ("self_attn.v_proj", "qkv"),
    ("attn.q_proj", "qkv"),
    ("attn.k_proj", "qkv"),
    ("attn.v_proj", "qkv"),
    ("query_key_value", "qkv"),
    # Attention output projection
    ("attention.dense", "proj"),
    ("self_attn.o_proj", "proj"),
    ("attn.o_proj", "proj"),
    ("attn.out_proj", "proj"),
    # MLP expand (hidden → 4h)
    ("mlp.dense_h_to_4h", "mlp_expand"),
    ("mlp.gate_proj", "mlp_expand"),
    ("mlp.up_proj", "mlp_expand"),
    ("mlp.fc1", "mlp_expand"),
    ("mlp.c_fc", "mlp_expand"),
    ("mlp.w1", "mlp_expand"),
    ("mlp.w3", "mlp_expand"),
    # MLP contract (4h → hidden)
    ("mlp.dense_4h_to_h", "mlp_contract"),
    ("mlp.down_proj", "mlp_contract"),
    ("mlp.fc2", "mlp_contract"),
    ("mlp.c_proj", "mlp_contract"),
    ("mlp.w2", "mlp_contract"),
]


def classify_granular(param_name: str) -> str:
    """Classify parameter into granular component: 'qkv', 'proj', 'mlp_expand', 'mlp_contract', or 'other'."""
    s = param_name.lower()
    for fragment, label in _COMP_RULES:
        if fragment in s:
            return label
    return "other"


# --- Math utilities ---
def frobenius_norm(A: torch.Tensor) -> float:
    """Compute Frobenius norm of a tensor."""
    return float(torch.norm(A.float(), p='fro').item())


def nuclear_norm(A: torch.Tensor) -> float:
    """Compute nuclear norm (sum of singular values)."""
    if A.numel() == 0 or A.ndim != 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(A.float())
        return float(s.sum().item())
    except:
        return 0.0


def spectral_norm_power(A: torch.Tensor, iters: int = 5, eps: float = 1e-12) -> float:
    """Estimate spectral norm using power iteration."""
    m, n = A.shape
    if m == 0 or n == 0:
        return 0.0
    v = torch.randn(n, 1, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + eps)
    for _ in range(iters):
        u = A @ v
        u = u / (u.norm() + eps)
        v = A.T @ u
        v = v / (v.norm() + eps)
    u = A @ v
    return float(u.norm().item())


def stable_rank_and_spectral(A: torch.Tensor, iters: int = 5, use_svd: bool = False) -> tuple[float, float]:
    """Compute stable rank AND spectral norm in one pass.

    Returns:
        (stable_rank, spectral_norm)
    """
    if A.numel() == 0:
        return 0.0, 0.0
    Af = A.float()
    fro_sq = float((Af * Af).sum(dtype=torch.float64).item())
    if fro_sq == 0.0:
        return 0.0, 0.0
    spec = compute_spectral_norm(Af) if use_svd else spectral_norm_power(Af, iters=iters)
    if spec <= 0:
        return 0.0, 0.0
    sr = fro_sq / (spec * spec)
    return sr, float(spec)


def stable_rank(A: torch.Tensor, iters: int = 5, use_svd: bool = False) -> float:
    """Compute stable rank = ||A||_F^2 / ||A||_2^2 (soft measure of matrix rank)."""
    sr, _ = stable_rank_and_spectral(A, iters=iters, use_svd=use_svd)
    return sr


def empirical_rank(A: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Compute empirical rank as the number of singular values needed to
    capture 'threshold' fraction of total variance (sum of squared singular values).

    Args:
        A: Input matrix
        threshold: Fraction of variance to capture (default 0.99)

    Returns:
        Number of singular values needed to capture threshold of variance
    """
    if A.numel() == 0:
        return 0

    # Compute SVD (we only need singular values)
    # Use float32 for memory efficiency
    Af = A.float()
    try:
        # torch.linalg.svdvals is more efficient when we only need singular values
        s = torch.linalg.svdvals(Af)
    except:
        # Fallback to standard SVD if svdvals not available
        _, s, _ = torch.linalg.svd(Af, full_matrices=False)

    # Compute squared singular values (these represent variance)
    s_squared = s * s
    total_variance = s_squared.sum().item()

    if total_variance == 0.0:
        return 0

    # Find how many singular values we need to capture threshold of variance
    cumsum = torch.cumsum(s_squared, dim=0)
    threshold_variance = threshold * total_variance

    # Find first index where cumsum exceeds threshold
    rank = torch.searchsorted(cumsum, threshold_variance).item() + 1

    # Ensure rank doesn't exceed matrix dimensions
    return min(rank, min(A.shape))


def condition_number(A: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute condition number (ratio of largest to smallest singular value).
    Large condition numbers indicate numerical instability.
    """
    if A.numel() == 0 or A.ndim != 2:
        return 1.0
    try:
        s = torch.linalg.svdvals(A.float())
        if len(s) == 0:
            return 1.0
        s_max = s[0].item()
        s_min = s[-1].item()
        return s_max / max(s_min, eps)
    except:
        return float('inf')


def compute_rank_deficiency(A: torch.Tensor, threshold: float = 1e-6) -> int:
    """
    Compute rank deficiency (how many dimensions are effectively zero).
    Returns min(m, n) - numerical_rank.
    """
    if A.numel() == 0:
        return min(A.shape) if A.ndim == 2 else 0
    try:
        s = torch.linalg.svdvals(A.float())
        numerical_rank = (s > threshold).sum().item()
        return min(A.shape) - numerical_rank
    except:
        return 0


# --- I/O utilities ---
def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """Write list of dicts to CSV, creating directories as needed."""
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if path has a directory component
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# --- Weights & Biases helpers ---
def _derive_run_name(script_name: str, args) -> str:
    """Derive a descriptive W&B run name from args.outdir (last 2 path segments)."""
    outdir = getattr(args, "outdir", None)
    if not outdir:
        return script_name
    # Normalise and grab last 2 non-empty components
    parts = [p for p in outdir.replace("\\", "/").split("/") if p]
    # Strip common root prefixes like "outputs" or "unlearned_models"
    while parts and parts[0] in ("outputs", "unlearned_models", "plots"):
        parts = parts[1:]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    if parts:
        return parts[-1]
    return script_name


def init_wandb(script_name: str, args, project: str = "cambridge_era", **kw):
    """Initialise a W&B run, logging to project "cambridge_era" by default.

    Silently no-ops if:
      - wandb is not installed, or
      - WANDB_API_KEY is not set in the environment (key absent from .env), or
      - WANDB_MODE=disabled is set explicitly.
    """
    try:
        import wandb
    except ImportError:
        print(f"[wandb] wandb not installed — skipping logging for {script_name}")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        return None
    run_name = _derive_run_name(script_name, args)
    group = os.environ.get("WANDB_RUN_GROUP", None)
    run = wandb.init(
        project=project,
        name=run_name,
        config=vars(args) if hasattr(args, "__dict__") else {},
        group=group,
        tags=[script_name],
        reinit=True,
        **kw,
    )
    return run


def log_csv_as_table(csv_path: str, key: str = "results"):
    """Upload a CSV file as a wandb.Table artefact."""
    try:
        import wandb
        if wandb.run is None:
            return
        import pandas as pd
        df = pd.read_csv(csv_path)
        wandb.log({key: wandb.Table(dataframe=df)})
    except Exception:
        pass


def log_plots(outdir: str, key_prefix: str = "plots"):
    """Glob all PNGs in *outdir* and log them as wandb.Images."""
    try:
        import wandb
        import glob as _glob
        if wandb.run is None:
            return
        for png in sorted(_glob.glob(os.path.join(outdir, "*.png"))):
            name = os.path.splitext(os.path.basename(png))[0]
            wandb.log({f"{key_prefix}/{name}": wandb.Image(png)})
    except Exception:
        pass


def finish_wandb():
    """Finish the current W&B run if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


# --- Model weight streaming ---
class SmartLoader:
    """Stream model weights one shard at a time from safetensors/bin files.

    We deliberately avoid AutoModelForCausalLM here because this script
    compares TWO models weight-by-weight.  Loading both full models would
    require ~2× model size in memory.  SmartLoader instead loads one shard
    file at a time, extracts individual weight tensors, computes stats, and
    frees them — so peak memory is roughly 2× one shard rather than 2× one
    full model.
    """

    def __init__(self, model_path: str):
        # Handle HF Hub IDs: If path doesn't exist locally, try downloading
        if not os.path.exists(model_path):
            print(f"'{model_path}' not found locally. Attempting HF Hub download...")
            from huggingface_hub import snapshot_download
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            try:
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors", "*.bin", "*.json"],
                    ignore_patterns=["*.msgpack", "*.h5"],
                    token=hf_token,
                )
            except Exception as e:
                print(f"[SmartLoader] Network download failed ({e}), trying local cache...")
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors", "*.bin", "*.json"],
                    ignore_patterns=["*.msgpack", "*.h5"],
                    token=hf_token,
                    local_files_only=True,
                )
            print(f"Downloaded/Found at: {model_path}")

        self.model_path = model_path
        self.index = {}
        self.is_safetensors = False
        self.single_file = None
        self._scan_structure()

        # Cache one shard at a time
        self.current_shard_path = None
        self.current_shard_data = {}

    def _scan_structure(self):
        safetensors_index = os.path.join(self.model_path, "model.safetensors.index.json")
        safetensors_file = os.path.join(self.model_path, "model.safetensors")
        pytorch_index = os.path.join(self.model_path, "pytorch_model.bin.index.json")
        pytorch_file = os.path.join(self.model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_index):
            self.is_safetensors = True
            with open(safetensors_index, "r") as f:
                data = json.load(f)
            self.index = data["weight_map"]
        elif os.path.exists(safetensors_file):
            self.is_safetensors = True
            self.single_file = safetensors_file
        elif os.path.exists(pytorch_index):
            self.is_safetensors = False
            with open(pytorch_index, "r") as f:
                data = json.load(f)
            self.index = data["weight_map"]
        elif os.path.exists(pytorch_file):
            self.is_safetensors = False
            self.single_file = pytorch_file
        else:
            if os.path.isfile(self.model_path):
                if self.model_path.endswith(".safetensors"):
                    self.is_safetensors = True
                    self.single_file = self.model_path
                else:
                    self.is_safetensors = False
                    self.single_file = self.model_path
            else:
                raise FileNotFoundError(f"Could not find model weights in {self.model_path}")

    def get_all_param_names(self) -> Set[str]:
        if self.index:
            return set(self.index.keys())

        # Single file case: we must peek
        if self.is_safetensors:
            from safetensors import safe_open
            with safe_open(self.single_file, framework="pt", device="cpu") as f:
                return set(f.keys())
        else:
            print(f"Warning: Loading full checkpoint {self.single_file} to list keys (Legacy PT format).")
            self.current_shard_data = torch.load(self.single_file, map_location="cpu", weights_only=True)
            self.current_shard_path = self.single_file
            return set(self.current_shard_data.keys())

    def get_param(self, name: str, device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.single_file:
            path = self.single_file
        else:
            if name not in self.index:
                return None
            filename = self.index[name]
            path = os.path.join(self.model_path, filename)

        if self.current_shard_path != path:
            del self.current_shard_data
            gc.collect()

            self.current_shard_path = path
            if self.is_safetensors:
                from safetensors.torch import load_file as load_safetensors_file
                self.current_shard_data = load_safetensors_file(path, device="cpu")
            else:
                self.current_shard_data = torch.load(path, map_location="cpu", weights_only=True)

        if name not in self.current_shard_data:
            return None

        tensor = self.current_shard_data[name]
        tensor = tensor.to(dtype=dtype, device=device)
        return tensor
