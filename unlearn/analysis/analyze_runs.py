# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "pandas",
#     "numpy",
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
    "MMLU", "WMDP (Robust Rewritten)", "WMDP (Robust)", "WMDP (Cloze)", "WMDP (Categorized)",
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
    "wn": "Noise Std", "wr": "Reg Lambda",
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


def _wmdp_primary(run_row):
    """Return the best available WMDP score: prefer robust_rewritten, fall back to robust."""
    rr = run_row.get("WMDP (Robust Rewritten)")
    if rr is not None and not (isinstance(rr, float) and np.isnan(rr)):
        return rr
    return run_row.get("WMDP (Robust)")


def _run_to_row(run_row) -> list[str]:
    """Convert a DataFrame row into a list of formatted cell strings."""
    mmlu = run_row.get("MMLU")
    wmdp = _wmdp_primary(run_row)
    return [
        str(run_row.get("Method", "")),
        _expand_config(_extract_config(str(run_row.get("Name", "")))),
        _fmt(run_row.get("L2 Dist"), 2),
        _fmt(mmlu),
        _fmt(run_row.get("WMDP (Robust Rewritten)")),
        _fmt(run_row.get("WMDP (Robust)")),
        _fmt(run_row.get("WMDP (Cloze)")),
        _fmt(run_row.get("WMDP (Categorized)")),
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

def _detect_method(run_name: str) -> str:
    pattern = r'[/_](' + '|'.join(KNOWN_METHODS) + r')__'
    match = re.search(pattern, run_name)
    return match.group(1) if match else "unknown"


def _fetch_runs(api) -> pd.DataFrame:
    project_name = "cambridge_era"
    print(f"Fetching runs from project '{project_name}'...")

    # Fetch ALL finished runs — training runs have NLL/L2, eval runs have MMLU/WMDP.
    # They share the same run name but are separate wandb runs.
    all_runs = api.runs(
        project_name,
        filters={"state": "finished"},
        per_page=500,
    )

    eval_data = []   # runs with MMLU (from eval.py)
    train_data = []  # runs with NLL/L2 (from unlearn.py)

    for run in tqdm(all_runs, desc="Processing runs"):
        method = run.config.get("hyperparameters", {}).get("method", "unknown")
        if method == "unknown":
            method = _detect_method(run.name)

        mmlu   = run.summary.get("eval_bench/mmlu/acc")
        wmdp_1 = run.summary.get("eval_bench/wmdp_bio_robust/acc")
        wmdp_2 = run.summary.get("eval_bench/wmdp_bio_cloze_verified/acc_norm")
        wmdp_3 = run.summary.get("eval_bench/wmdp_bio_categorized_mcqa/acc")
        wmdp_4 = run.summary.get("eval_bench/wmdp_bio_robust_rewritten/acc")
        l2     = run.summary.get("weight_l2_dist")
        fnll   = run.summary.get("final_forget_nll")
        rnll   = run.summary.get("final_retain_nll")

        is_base = not any(f"/{m}__" in run.name or f"_{m}__" in run.name for m in KNOWN_METHODS)

        has_eval = mmlu is not None or wmdp_1 is not None or wmdp_4 is not None
        has_train = fnll is not None or l2 is not None

        if has_eval:
            eval_data.append({
                "Name":                    run.name,
                "Created":                 run.created_at,
                "Method":                  method,
                "MMLU":                    mmlu,
                "WMDP (Robust)":           wmdp_1,
                "WMDP (Cloze)":            wmdp_2,
                "WMDP (Categorized)":      wmdp_3,
                "WMDP (Robust Rewritten)": wmdp_4,
                "IsBase":                  is_base,
            })

        if has_train:
            train_data.append({
                "Name":        run.name,
                "Created":     run.created_at,
                "L2 Dist":     l2,
                "Forget NLL":  fnll,
                "Retain NLL":  rnll,
            })

    if not eval_data:
        print("No finished runs with evaluation metrics found.")
        return pd.DataFrame()

    eval_df = pd.DataFrame(eval_data)
    print(f"\nFound {len(eval_df)} eval runs, {len(train_data)} training runs.")

    # Deduplicate eval runs: keep earliest per unique run name
    eval_df = eval_df.sort_values("Created").drop_duplicates(subset="Name", keep="first")

    # Join training metrics (NLL, L2) by run name — take earliest training run per name
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_df = train_df.sort_values("Created").drop_duplicates(subset="Name", keep="first")
        df = eval_df.merge(
            train_df[["Name", "L2 Dist", "Forget NLL", "Retain NLL"]],
            on="Name", how="left",
        )
    else:
        df = eval_df
        df["L2 Dist"] = np.nan
        df["Forget NLL"] = np.nan
        df["Retain NLL"] = np.nan

    # Deduplicate: old-style flat names ("unlearned_models_EleutherAI_..._npo__...")
    # vs new-style slash names ("EleutherAI_.../npo__...") that point to the same model.
    # Extract the config portion (everything after "__") and dedup on that.
    def _config_key(name: str) -> str:
        """Extract config portion for dedup: 'org/model/method__config' -> 'method__config'"""
        if "/" in name:
            name = name.rsplit("/", 1)[-1]  # take last path segment
        return name

    df["_config_key"] = df["Name"].apply(_config_key)
    df["_style"] = df["Name"].apply(lambda n: 0 if "/" in n else 1)
    df = (
        df.sort_values("_style")          # prefer new-style (has "/")
          .drop_duplicates(subset="_config_key", keep="first")
          .drop(columns=["_style", "_config_key"])
    )

    # Score: MMLU - WMDP. Prefer wmdp_bio_robust_rewritten (what eval.py runs),
    # fall back to wmdp_bio_robust for older runs that only have that variant.
    wmdp_for_score = df["WMDP (Robust Rewritten)"].fillna(df["WMDP (Robust)"])
    df["Score"] = df["MMLU"] - wmdp_for_score
    print(f"After dedup: {len(df)} unique runs.\n")
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
        f.write("*Best run per method ranked by Score = MMLU − WMDP (Robust Rewritten, with fallback to Robust)*\n\n")
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

        # 3. All runs, newest first
        all_runs = df.sort_values("Created", ascending=False)
        f.write("## All Runs (Newest First)\n\n")
        f.write(_df_to_md(all_runs) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
