#!/usr/bin/env python
# /// script
# dependencies = [
#   "wandb",
# ]
# ///

"""
Check whether a W&B run for a given pipeline step has already completed.

Used by pipeline.sh to skip steps that have finished runs in W&B,
even if local sentinel files are missing.

Exit codes:
  0  — a matching finished run was found (step is complete)
  1  — no matching finished run found (step should run)
  2  — W&B unavailable or not configured (fall back to local check)

Usage:
  python experiment/check_wandb_complete.py \\
    --run-name "null_space_analysis/seed_42" \\
    --model-a "EleutherAI/deep-ignorance-unfiltered" \\
    --model-b "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"

  # For single-seed steps (no seed in run name):
  python experiment/check_wandb_complete.py \\
    --run-name "weight_comparison" \\
    --model-a "..." --model-b "..."

  # For per-model steps (only model-a matters, e.g. wmdp lens):
  python experiment/check_wandb_complete.py \\
    --run-name "wmdp_logit_lens/seed_42" \\
    --model-a "EleutherAI/deep-ignorance-unfiltered"
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Check if a W&B run for a pipeline step has completed."
    )
    parser.add_argument("--run-name", required=True,
                        help="Expected W&B run display name (e.g. 'null_space_analysis/seed_42')")
    parser.add_argument("--model-a", required=True, help="Model A identifier")
    parser.add_argument("--model-b", default=None, help="Model B identifier (omit for per-model steps)")
    parser.add_argument("--project", default="cambridge_era", help="W&B project name")
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        sys.exit(2)

    try:
        import wandb
    except ImportError:
        sys.exit(2)

    try:
        api = wandb.Api(timeout=10)
    except Exception:
        sys.exit(2)

    # W&B entity — derive from the API's default entity
    try:
        entity = api.default_entity
    except Exception:
        sys.exit(2)

    # Build filters: match on display_name and state
    filters = {
        "display_name": args.run_name,
        "state": "finished",
    }

    try:
        runs = api.runs(
            f"{entity}/{args.project}",
            filters=filters,
            per_page=50,
        )

        for run in runs:
            config = run.config or {}

            # Match model_a — check common config key names
            run_model_a = (
                config.get("model_a")
                or config.get("model")
                or ""
            )
            if not _models_match(run_model_a, args.model_a):
                continue

            # Match model_b if provided
            if args.model_b is not None:
                run_model_b = config.get("model_b", "")
                if not _models_match(run_model_b, args.model_b):
                    continue

            # Found a matching completed run
            sys.exit(0)

    except Exception:
        # API error — fall back to local check
        sys.exit(2)

    # No matching run found
    sys.exit(1)


def _models_match(run_value: str, expected: str) -> bool:
    """Check if model identifiers match, handling path normalization."""
    if not run_value or not expected:
        return False
    # Normalize: strip trailing slashes, compare case-insensitively
    run_clean = run_value.strip("/").lower()
    expected_clean = expected.strip("/").lower()
    # Direct match
    if run_clean == expected_clean:
        return True
    # Match after replacing / with _ (local path vs HF id)
    if run_clean.replace("/", "_") == expected_clean.replace("/", "_"):
        return True
    # Match on the last component (basename)
    if run_clean.split("/")[-1] == expected_clean.split("/")[-1]:
        return True
    return False


if __name__ == "__main__":
    main()
