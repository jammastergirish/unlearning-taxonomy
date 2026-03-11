#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "python-dotenv",
#     "tqdm",
#     "torch",
# ]
# ///
"""
Backfill method tags on all finished W&B runs in the cambridge_era project.

For each run, the method is inferred (in priority order) from:
  1. run.config["hyperparameters"]["method"]
  2. run.config["method"]
  3. A regex search of the run name for a known method slug

A tag "method:<name>" is then added to the run (idempotently — existing
tags are preserved, and the tag is only written if not already present).

Usage:
  uv run tag_wandb_runs.py
  uv run tag_wandb_runs.py --dry-run   # print changes without applying them
"""

import argparse
import os
import re

from dotenv import load_dotenv
from tqdm import tqdm
import wandb

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import infer_method_from_model_name


def infer_method(run) -> str | None:
    """Return the method slug for this run, or None if unknown."""
    # 1. Prefer explicit config field written by unlearn.py
    hp = run.config.get("hyperparameters", {})
    method = hp.get("method") or run.config.get("method")
    if method and method != "unknown":
        return method

    # 2. Fall back to regex on the run name
    return infer_method_from_model_name(run.name)


def main():
    parser = argparse.ArgumentParser(description="Backfill method tags on W&B runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing to W&B")
    parser.add_argument("--project", default="cambridge_era")
    args = parser.parse_args()

    load_dotenv()

    api = wandb.Api()
    runs = api.runs(
        args.project,
        filters={"state": "finished"},
        per_page=500,
    )

    updated = skipped = unknown = 0

    for run in tqdm(runs, desc="Tagging runs"):
        method = infer_method(run)
        if method is None:
            tqdm.write(f"  [UNKNOWN] {run.name}")
            unknown += 1
            continue

        tag = f"method:{method}"
        existing_tags = list(run.tags or [])

        if tag in existing_tags:
            skipped += 1
            continue

        new_tags = existing_tags + [tag]
        if args.dry_run:
            tqdm.write(f"  [DRY RUN] {run.name}  →  add tag '{tag}'")
        else:
            run.tags = new_tags
            run.update()
            tqdm.write(f"  [TAGGED]  {run.name}  →  '{tag}'")
        updated += 1

    print(f"\nDone. updated={updated}  skipped={skipped}  unknown={unknown}")
    if args.dry_run:
        print("(dry-run — no changes were written)")


if __name__ == "__main__":
    main()
