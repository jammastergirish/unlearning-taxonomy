#!/usr/bin/env python3
"""
check_dataset_stats.py
----------------------
Prints basic statistics for the forget and retain datasets.
Tokenizes each line to give accurate token-level counts (not just char counts).

Usage:
    python check_dataset_stats.py                          # default tokenizer
    python check_dataset_stats.py --model EleutherAI/deep-ignorance-unfiltered
    python check_dataset_stats.py --max-length 256         # change truncation threshold
"""

import argparse
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def load_lines(path: Path) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def word_lengths(lines: list[str]) -> list[int]:
    """Rough whitespace-tokenized length as a fast proxy."""
    return [len(l.split()) for l in lines]


def char_lengths(lines: list[str]) -> list[int]:
    return [len(l) for l in lines]


def print_stats(name: str, lines: list[str], lengths: list[int], threshold: int):
    over = [l for l in lengths if l > threshold]
    print(f"\n{'='*55}")
    print(f"  {name}  ({len(lines):,} samples)")
    print(f"{'='*55}")
    print(f"  Total samples     : {len(lines):,}")
    print(f"  Min length        : {min(lengths):,}")
    print(f"  Max length        : {max(lengths):,}")
    print(f"  Mean length       : {statistics.mean(lengths):.1f}")
    print(f"  Median length     : {statistics.median(lengths):.1f}")
    print(f"  Stdev             : {statistics.stdev(lengths):.1f}")
    print(f"  --- Truncation (threshold={threshold}) ---")
    print(f"  > {threshold}              : {len(over):,}  ({100*len(over)/len(lengths):.1f}%)")
    if over:
        print(f"  Max over-length   : {max(over):,}")
        print(f"  Mean over-length  : {statistics.mean(over):.1f}")

    # Percentiles
    sorted_l = sorted(lengths)
    n = len(sorted_l)
    for pct in [50, 75, 90, 95, 99]:
        idx = min(int(pct / 100 * n), n - 1)
        print(f"  p{pct:<2}               : {sorted_l[idx]:,}")


def token_stats(name: str, lines: list[str], tokenizer, threshold: int):
    """Run the real tokenizer for accurate token counts."""
    print(f"\n  [Token counts via tokenizer — {name}]")
    token_lengths = []
    for text in lines:
        ids = tokenizer.encode(text, add_special_tokens=True)
        token_lengths.append(len(ids))

    over = [l for l in token_lengths if l > threshold]
    print(f"  Min tokens        : {min(token_lengths):,}")
    print(f"  Max tokens        : {max(token_lengths):,}")
    print(f"  Mean tokens       : {statistics.mean(token_lengths):.1f}")
    print(f"  Median tokens     : {statistics.median(token_lengths):.1f}")
    print(f"  > {threshold} tokens       : {len(over):,}  ({100*len(over)/len(token_lengths):.1f}%)")

    sorted_l = sorted(token_lengths)
    n = len(sorted_l)
    for pct in [50, 75, 90, 95, 99]:
        idx = min(int(pct / 100 * n), n - 1)
        print(f"  p{pct:<2} tokens        : {sorted_l[idx]:,}")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for forget/retain datasets")
    parser.add_argument("--forget", default=str(ROOT / "data" / "forget.txt"), help="Path to forget.txt")
    parser.add_argument("--retain", default=str(ROOT / "data" / "retain.txt"), help="Path to retain.txt")
    parser.add_argument("--max-length", type=int, default=512, help="Truncation threshold (default: 512)")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model ID to load tokenizer from. "
                             "If omitted, uses word-count as a proxy.")
    args = parser.parse_args()

    forget_path = Path(args.forget)
    retain_path = Path(args.retain)

    for p in [forget_path, retain_path]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    forget_lines = load_lines(forget_path)
    retain_lines = load_lines(retain_path)

    # --- Word-count stats (always fast, no model needed) ---
    print(f"\nThreshold: {args.max_length} tokens   |   Measuring: word count (proxy)")
    print_stats("FORGET SET", forget_lines, word_lengths(forget_lines), args.max_length)
    print_stats("RETAIN SET", retain_lines, word_lengths(retain_lines), args.max_length)

    # --- Real tokenizer stats (optional) ---
    if args.model:
        try:
            from transformers import AutoTokenizer
            print(f"\nLoading tokenizer: {args.model} ...")
            tok = AutoTokenizer.from_pretrained(args.model)
            token_stats("FORGET SET", forget_lines, tok, args.max_length)
            token_stats("RETAIN SET", retain_lines, tok, args.max_length)
        except Exception as e:
            print(f"\n[WARNING] Could not load tokenizer: {e}")
            print("  Falling back to word-count proxy only.")
    else:
        print("\n[TIP] Pass --model <HF_ID> to get accurate token counts instead of word counts.")

    print()


if __name__ == "__main__":
    main()
