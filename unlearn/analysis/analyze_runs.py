# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "pandas",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

import os
import re

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

KNOWN_METHODS = [
    "tar", "cb_lat", "cb", "lat", "rmu", "npo", "simnpo",
    "wt_dist_reg", "wt_dist", "ga_simple", "ga", "grad_diff", "dpo",
]

# Columns shown in every table
TABLE_COLS = [
    "Method", "Config", "L2 Dist",
    "MMLU", "WMDP (Robust)", "WMDP (Cloze)", "WMDP (Categorized)", "WMDP (Robust Rewritten)",
    "MMLU-WMDP", "Forget NLL", "Retain NLL", "Run Name",
]

# Abbreviated param names in run names → readable labels
_ABBREV_TO_LABEL = {
    "ep": "Epochs", "lr": "Learning Rate", "bs": "Batch Size",
    "mle": "Max Length", "mli": "Max Lines", "ml": "Max Length",
    "rw": "Retain Weight", "fw": "Forget Weight",
    "b": "Beta", "a": "Alpha", "sc": "Steering Coeff", "ly": "Layers",
    "le": "LAT Epsilon", "ls": "LAT Steps",
    "ta": "TAR Alpha", "tlr": "TAR LR", "tep": "TAR Epochs",
    "wn": "Noise Std", "wr": "Reg Lambda", "opt": "Optimizer",
}
_ABBREV_PATTERN = sorted(_ABBREV_TO_LABEL.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _extract_config(run_name: str) -> str:
    """'EleutherAI_.../ga__ep1_lr2e-05_bs4_rw5.0' → 'ep1_lr2e-05_bs4_rw5.0'"""
    if "/" in run_name:
        run_name = run_name.split("/", 1)[1]
    if "__" in run_name:
        run_name = run_name.split("__", 1)[1]
    return run_name


def _expand_config(config_str: str) -> str:
    """Expand abbreviated config into readable '<br>'-separated labels."""
    parts = config_str.split("_")
    result = []
    for part in parts:
        matched = False
        for abbrev in _ABBREV_PATTERN:
            if part.startswith(abbrev):
                value = part[len(abbrev):]
                result.append(f"**{_ABBREV_TO_LABEL[abbrev]}:** {value}")
                matched = True
                break
        if not matched:
            if result:
                result[-1] += f"-{part}"
            else:
                result.append(part)
    return "<br>".join(result)


def _score(mmlu, wmdp):
    if mmlu is not None and wmdp is not None and not np.isnan(float(mmlu)) and not np.isnan(float(wmdp)):
        return f"{mmlu - wmdp:.4f}"
    return "N/A"


def _run_to_row(run_row) -> list[str]:
    """Convert a DataFrame row into a list of formatted cell strings."""
    mmlu = run_row.get("MMLU")
    wmdp = run_row.get("WMDP (Robust)")
    return [
        str(run_row.get("Method", "")),
        _expand_config(_extract_config(str(run_row.get("Name", "")))),
        _fmt(run_row.get("L2 Dist"), 2),
        _fmt(mmlu),
        _fmt(wmdp),
        _fmt(run_row.get("WMDP (Cloze)")),
        _fmt(run_row.get("WMDP (Categorized)")),
        _fmt(run_row.get("WMDP (Robust Rewritten)")),
        _score(mmlu, wmdp),
        _fmt(run_row.get("Forget NLL"), 3),
        _fmt(run_row.get("Retain NLL"), 3),
        str(run_row.get("Name", "")),
    ]


def _md_table(rows: list[list[str]], headers: list[str] | None = None) -> str:
    """Render a list of row-lists as a GitHub-Flavoured Markdown table."""
    if headers is None:
        headers = TABLE_COLS
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert a DataFrame of runs to a markdown table."""
    if df.empty:
        return "No runs found."
    rows = [_run_to_row(row) for _, row in df.iterrows()]
    return _md_table(rows)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_runs(api) -> pd.DataFrame:
    project_name = "cambridge_era"
    print(f"Fetching runs from project '{project_name}'...")

    runs = api.runs(
        project_name,
        filters={
            "$and": [
                {"state": "finished"},
                {"summary_metrics.eval_bench/mmlu/acc": {"$exists": True}},
            ]
        },
        per_page=500,
    )

    data = []
    for run in tqdm(runs, desc="Processing runs"):
        method = run.config.get("hyperparameters", {}).get("method", "unknown")
        if method == "unknown":
            pattern = r'[/_](' + '|'.join(KNOWN_METHODS) + r')__'
            match = re.search(pattern, run.name)
            if match:
                method = match.group(1)

        mmlu   = run.summary.get("eval_bench/mmlu/acc")
        wmdp_1 = run.summary.get("eval_bench/wmdp_bio_robust/acc")
        wmdp_2 = run.summary.get("eval_bench/wmdp_bio_cloze_verified/acc_norm")
        wmdp_3 = run.summary.get("eval_bench/wmdp_bio_categorized_mcqa/acc")
        wmdp_4 = run.summary.get("eval_bench/wmdp_bio_robust_rewritten/acc")

        if mmlu is None and wmdp_1 is None and wmdp_2 is None and wmdp_3 is None and wmdp_4 is None:
            continue

        is_base = not any(f"/{m}__" in run.name or f"_{m}__" in run.name for m in KNOWN_METHODS)

        data.append({
            "Run ID":                  run.id,
            "Name":                    run.name,
            "Created":                 run.created_at,
            "Method":                  method,
            "MMLU":                    mmlu,
            "WMDP (Robust)":           wmdp_1,
            "WMDP (Cloze)":            wmdp_2,
            "WMDP (Categorized)":      wmdp_3,
            "WMDP (Robust Rewritten)": wmdp_4,
            "L2 Dist":                 run.summary.get("weight_l2_dist"),
            "Forget NLL":              run.summary.get("final_forget_nll"),
            "Retain NLL":              run.summary.get("final_retain_nll"),
            "IsBase":                  is_base,
        })

    if not data:
        print("No finished runs with evaluation metrics found.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Deduplicate old-style vs new-style run names with identical metrics
    df["_style"] = df["Name"].apply(lambda n: 0 if "/" in n else 1)
    df["_fp"] = (
        df["MMLU"].round(4).astype(str) + "|"
        + df["WMDP (Robust)"].round(4).astype(str) + "|"
        + df["WMDP (Cloze)"].round(4).astype(str) + "|"
        + df["WMDP (Categorized)"].round(4).astype(str) + "|"
        + df["WMDP (Robust Rewritten)"].round(4).astype(str)
    )
    df = (
        df.sort_values("_style")
          .drop_duplicates(subset="_fp", keep="first")
          .drop(columns=["_style", "_fp"])
    )

    df["Score"] = df["MMLU"] - df["WMDP (Robust)"]
    print(f"\nProcessed {len(df)} runs.\n")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    api = wandb.Api()
    df = _fetch_runs(api)
    if df.empty:
        return

    baselines_df = df[df["IsBase"]].sort_values("Name")
    sweeps_df = df[~df["IsBase"]].sort_values(
        by=["Method", "Score", "MMLU"], ascending=[True, False, False],
    )

    out_file = os.path.join(os.path.dirname(__file__), "best_unlearning_models.md")
    print(f"Writing results to {out_file}...")

    with open(out_file, "w") as f:
        # 1. Best config per method
        f.write("## Best Config Per Method\n\n")
        f.write("*Best run per method ranked by Score = MMLU − WMDP (Robust)*\n\n")
        best_rows = []
        for method, group in sweeps_df.groupby("Method"):
            ranked = group.dropna(subset=["Score"]).sort_values(
                by=["Score", "MMLU"], ascending=[False, False],
            )
            if ranked.empty:
                ranked = group.sort_values("MMLU", ascending=False)
            if ranked.empty:
                continue
            best_rows.append(_run_to_row(ranked.iloc[0]))
        if best_rows:
            f.write(_md_table(best_rows) + "\n\n")
        else:
            f.write("No sweep runs found.\n\n")

        # 2. Baselines
        f.write("## Baselines\n\n")
        if not baselines_df.empty:
            f.write(_df_to_md(baselines_df) + "\n\n")
        else:
            f.write("No baseline runs found.\n\n")

        # 3. All runs, grouped by method then newest first within each group
        f.write("## All Runs (Grouped by Method)\n\n")
        # Build a method-order mapping; unknowns go last
        method_order = {m: i for i, m in enumerate(KNOWN_METHODS)}
        all_runs = df.copy()
        all_runs["_method_order"] = all_runs["Method"].map(
            lambda m: method_order.get(m, len(KNOWN_METHODS))
        )
        all_runs = all_runs.sort_values(
            ["_method_order", "Created"], ascending=[True, False]
        ).drop(columns=["_method_order"])

        # Emit one sub-section per method
        for method, group in all_runs.groupby("Method", sort=False):
            f.write(f"### {method}\n\n")
            f.write(_df_to_md(group) + "\n\n")

    print("Done!")


if __name__ == "__main__":
    main()
