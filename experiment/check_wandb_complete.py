#!/usr/bin/env python
# /// script
# dependencies = [
#   "wandb",
# ]
# ///

"""
Fetch all finished W&B runs for the project and cache them locally.

Called once at pipeline startup. Writes a simple text file listing all
finished run names + their model_a/model_b config values, one per line.
Subsequent completion checks are just grep against this file — no
network calls needed.

Usage:
  # Fetch and cache (run once at pipeline start):
  uv run experiment/check_wandb_complete.py --fetch \\
    --cache-file /tmp/wandb_finished_runs.txt

  # Check a specific step (fast, local grep):
  uv run experiment/check_wandb_complete.py --check \\
    --cache-file /tmp/wandb_finished_runs.txt \\
    --run-name "null_space_analysis/seed_42" \\
    --model-a "EleutherAI/deep-ignorance-unfiltered" \\
    --model-b "girishgupta/deep-ignorance-unfiltered_unlearned_dpo"

Exit codes:
  0  — (fetch) cache written successfully / (check) matching run found
  1  — (fetch) failed / (check) no matching run
  2  — W&B unavailable or not configured
"""

import argparse
import os
import sys


def fetch_finished_runs(project: str, cache_file: str) -> int:
    """Fetch all finished runs from W&B and write to cache file.

    Each line: run_name<TAB>model_a<TAB>model_b
    """
    if not os.environ.get("WANDB_API_KEY"):
        return 2

    try:
        import wandb
    except ImportError:
        return 2

    try:
        api = wandb.Api(timeout=30)
        entity = api.default_entity
    except Exception:
        return 2

    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"state": "finished"},
            per_page=1000,
        )

        lines = []
        for run in runs:
            name = run.display_name or run.name or ""
            config = run.config or {}
            model_a = config.get("model_a") or config.get("model") or ""
            model_b = config.get("model_b") or ""
            lines.append(f"{name}\t{model_a}\t{model_b}")

        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"[wandb-cache] Cached {len(lines)} finished runs to {cache_file}")
        return 0

    except Exception as exc:
        print(f"[wandb-cache] Failed to fetch runs: {exc}", file=sys.stderr)
        return 2


def check_cached(cache_file: str, run_name: str, model_a: str, model_b: str | None) -> int:
    """Check the cache file for a matching finished run.

    Returns 0 if found, 1 if not found, 2 if cache doesn't exist.
    """
    if not os.path.exists(cache_file):
        return 2

    model_a_normalized = _normalize(model_a)

    with open(cache_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            cached_name = parts[0]
            cached_model_a = parts[1] if len(parts) > 1 else ""
            cached_model_b = parts[2] if len(parts) > 2 else ""

            if cached_name != run_name:
                continue
            if not _models_match(cached_model_a, model_a_normalized):
                continue
            if model_b is not None:
                if not _models_match(cached_model_b, _normalize(model_b)):
                    continue

            return 0

    return 1


def _normalize(model_id: str) -> str:
    return model_id.strip("/").lower()


def _models_match(cached: str, expected: str) -> bool:
    if not cached or not expected:
        return False
    cached_clean = _normalize(cached)
    # Direct match
    if cached_clean == expected:
        return True
    # Match after replacing / with _ (local path vs HF id)
    if cached_clean.replace("/", "_") == expected.replace("/", "_"):
        return True
    # Basename match
    if cached_clean.split("/")[-1] == expected.split("/")[-1]:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch all finished runs from W&B and write cache")
    parser.add_argument("--check", action="store_true",
                        help="Check cache for a specific run")
    parser.add_argument("--cache-file", required=True,
                        help="Path to the cache file")
    parser.add_argument("--project", default="cambridge_era")
    # --check arguments
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--model-b", default=None)
    args = parser.parse_args()

    if args.fetch:
        sys.exit(fetch_finished_runs(args.project, args.cache_file))
    elif args.check:
        if not args.run_name or not args.model_a:
            print("--check requires --run-name and --model-a", file=sys.stderr)
            sys.exit(2)
        sys.exit(check_cached(args.cache_file, args.run_name, args.model_a, args.model_b))
    else:
        print("Specify --fetch or --check", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
