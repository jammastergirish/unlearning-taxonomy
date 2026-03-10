# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "python-dotenv",
# ]
# ///
"""
Extract metrics for the 6 best unlearning configs from wandb.

Training and eval are separate wandb runs with the same run name:
  - Training runs log: final_forget_nll, final_retain_nll, weight_l2_dist
  - Eval runs log:     eval_bench/mmlu/acc, eval_bench/wmdp_bio_robust_rewritten/acc

This script fetches both, matches them by run name, and prints a unified table.

Usage:
    uv run --script unlearn/analysis/verify_best_configs.py
"""

import wandb
from dotenv import load_dotenv

load_dotenv()
api = wandb.Api()

# --- Config ---

PROJECT = "cambridge_era"

# Run name substrings for the 6 best configs (wandb formats floats, e.g. a200.0)
BEST_CONFIGS = {
    "cb":        "cb__ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15",
    "cb_lat":    "cb_lat__ep1_lr1.3e-05_bs16_a200.0_sc10.0_le0.1_ls5_ly13-14-15",
    "ga":        "ga__ep1_lr2e-05_bs4_rw5.0",
    "grad_diff": "grad_diff__ep1_lr1e-05_bs4_fw1.0",
    "tar":       "tar__ta5.0_tlr1e-05_tep1",
    "wt_dist":   "wt_dist__ep1_lr2e-05_bs4_wn0.0001",
}

# --- Fetch runs ---

print(f"Fetching finished runs from '{PROJECT}'...")
runs = list(api.runs(PROJECT, filters={"state": "finished"}, per_page=500))
print(f"Total finished runs: {len(runs)}")

# --- For each best config, find its training + eval runs ---

print()
print(f"{'Method':<12} {'L2':>8} {'F_NLL':>8} {'R_NLL':>8} {'MMLU':>8} {'WMDP_RR':>8}")
print("-" * 62)

for method, substr in BEST_CONFIGS.items():
    # Find all runs whose name contains the config substring
    matching = [r for r in runs if substr in r.name]

    # Split into training runs (have NLL) and eval runs (have MMLU)
    train = [r for r in matching if r.summary.get("final_forget_nll") is not None]
    evals = [r for r in matching if r.summary.get("eval_bench/mmlu/acc") is not None]

    # Pick the earliest run of each type (= the original sweep run)
    t = sorted(train, key=lambda r: r.created_at)[0] if train else None
    e = sorted(evals, key=lambda r: r.created_at)[0] if evals else None

    l2   = f"{t.summary['weight_l2_dist']:.2f}"    if t and t.summary.get("weight_l2_dist") else "N/A"
    fnll = f"{t.summary['final_forget_nll']:.3f}"   if t else "N/A"
    rnll = f"{t.summary['final_retain_nll']:.3f}"   if t else "N/A"
    mmlu = f"{e.summary['eval_bench/mmlu/acc']:.4f}" if e else "N/A"
    wmdp = f"{e.summary['eval_bench/wmdp_bio_robust_rewritten/acc']:.4f}" if e and e.summary.get("eval_bench/wmdp_bio_robust_rewritten/acc") else "N/A"

    print(f"{method:<12} {l2:>8} {fnll:>8} {rnll:>8} {mmlu:>8} {wmdp:>8}")

    # Show all runs found for transparency
    if len(train) > 1 or len(evals) > 1:
        print(f"  ({len(train)} training runs, {len(evals)} eval runs found)")
