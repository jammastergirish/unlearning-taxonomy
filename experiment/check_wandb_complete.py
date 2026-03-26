#!/usr/bin/env python
# /// script
# dependencies = [
#   "wandb",
# ]
# ///

"""
Fetch all finished W&B run names and cache them locally.

Called once at pipeline startup. Writes one run name per line.
Subsequent checks are a simple grep — no network calls needed.

Usage:
  # Fetch (once at startup):
  uv run experiment/check_wandb_complete.py --fetch --cache-file /tmp/wandb_runs.txt

  # Check (instant, per step):
  python3 experiment/check_wandb_complete.py --check --cache-file /tmp/wandb_runs.txt \\
    --run-name "null_space_analysis/seed_42"

Exit codes:
  0  — (fetch) success / (check) run found
  1  — (fetch) failed  / (check) not found
  2  — W&B unavailable
"""

import argparse
import os
import sys


def fetch_finished_runs(project: str, cache_file: str) -> int:
    """Fetch all finished run display names from W&B into a text file."""
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

        names = set()
        for run in runs:
            name = run.display_name or run.name or ""
            if name:
                names.add(name)

        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as fh:
            fh.write("\n".join(sorted(names)) + "\n")

        print(f"[wandb-cache] Cached {len(names)} finished run names to {cache_file}")
        return 0

    except Exception as exc:
        print(f"[wandb-cache] Failed to fetch runs: {exc}", file=sys.stderr)
        return 2


def check_cached(cache_file: str, run_name: str) -> int:
    """Check if a run name exists in the cache. 0=found, 1=not found, 2=no cache."""
    if not os.path.exists(cache_file):
        return 2

    with open(cache_file) as fh:
        for line in fh:
            if line.strip() == run_name:
                return 0
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--project", default="cambridge_era")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.fetch:
        sys.exit(fetch_finished_runs(args.project, args.cache_file))
    elif args.check:
        if not args.run_name:
            print("--check requires --run-name", file=sys.stderr)
            sys.exit(2)
        sys.exit(check_cached(args.cache_file, args.run_name))
    else:
        print("Specify --fetch or --check", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
