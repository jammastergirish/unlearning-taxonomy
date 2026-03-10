# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "wandb",
#     "pandas",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

import wandb

KNOWN_METHODS = ["tar", "cb_lat", "cb", "lat", "rmu", "npo", "simnpo", "wt_dist_reg", "wt_dist", "ga_simple", "ga", "grad_diff", "dpo"]
import pandas as pd
from dotenv import load_dotenv


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a DataFrame subset as a GitHub-Flavoured Markdown table."""
    def fmt(val):
        if isinstance(val, float):
            return f"{val:.4f}" if pd.notnull(val) else "N/A"
        return str(val)

    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = [
        "| " + " | ".join(fmt(row[c]) for c in cols) + " |"
        for _, row in df[cols].iterrows()
    ]
    return "\n".join([header, sep] + rows)


def _extract_best_config(run_name: str) -> str:
    """Extract a short human-readable config string from the run name.

    Run names look like:
      EleutherAI_deep-ignorance-unfiltered/ga__ep3_lr2e-05_bs32_rw5.0_ml2048
    We strip the model prefix and the method prefix, leaving just the params.
    """
    # Strip model prefix (everything up to and including the slash)
    if "/" in run_name:
        run_name = run_name.split("/", 1)[1]
    # Strip method prefix (everything up to and including the first __)
    if "__" in run_name:
        run_name = run_name.split("__", 1)[1]
    return run_name


# Mapping from abbreviated parameter names to readable labels
_ABBREV_TO_LABEL = {
    "ep": "Epochs",
    "lr": "Learning Rate",
    "bs": "Batch Size",
    "mle": "Max Length",
    "mli": "Max Lines",
    "ml": "Max Length",
    "rw": "Retain Weight",
    "fw": "Forget Weight",
    "b": "Beta",
    "a": "Alpha",
    "sc": "Steering Coeff",
    "ly": "Layers",
    "le": "LAT Epsilon",
    "ls": "LAT Steps",
    "ta": "TAR Alpha",
    "tlr": "TAR LR",
    "tep": "TAR Epochs",
    "wn": "Noise Std",
    "wr": "Reg Lambda",
}

# Ordered by longest abbreviation first so e.g. "mle" matches before "ml"
_ABBREV_PATTERN = sorted(_ABBREV_TO_LABEL.keys(), key=len, reverse=True)


def _expand_config(config_str: str) -> str:
    """Convert an abbreviated config like 'ep1_lr1.3e-05_bs16_a200.0_sc10.0_ly13-14-15'
    into a readable multi-line string using <br> for line breaks in markdown tables.
    """
    import re
    parts = config_str.split("_")

    result = []
    i = 0
    while i < len(parts):
        part = parts[i]
        matched = False
        for abbrev in _ABBREV_PATTERN:
            if part.startswith(abbrev):
                value = part[len(abbrev):]
                # Handle layer values with dashes (already fine)
                label = _ABBREV_TO_LABEL[abbrev]
                result.append(f"**{label}:** {value}")
                matched = True
                break
        if not matched:
            # Could be a continuation of a previous value (shouldn't happen
            # with current naming, but handle gracefully)
            if result:
                result[-1] += f"-{part}"
            else:
                result.append(part)
        i += 1

    return "<br>".join(result)


def main():
    load_dotenv()
    api = wandb.Api()

    project_name = "cambridge_era"

    print(f"Fetching runs from project '{project_name}'...")
    try:
        # Filter server-side:
        #   - only finished runs
        #   - only runs where MMLU was logged (proxy for "eval completed")
        # This avoids fetching every in-progress/crashed/no-eval run entirely.
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
    except Exception as e:
        print(f"Error fetching runs: {e}")
        print(f"Please check your WANDB_API_KEY and project name ({project_name}).")
        return

    data = []

    from tqdm import tqdm
    for run in tqdm(runs, desc="Processing runs"):
        method = run.config.get("hyperparameters", {}).get("method", "unknown")

        # If method is unknown, try to parse it from the run name
        if method == "unknown":
            import re
            # Match patterns like "/simnpo__" or "_simnpo__" in the run name
            pattern = r'[/_](' + '|'.join(KNOWN_METHODS) + r')__'
            match = re.search(pattern, run.name)
            if match:
                method = match.group(1)

        mmlu   = run.summary.get("eval_bench/mmlu/acc",                         None)
        wmdp_1 = run.summary.get("eval_bench/wmdp_bio_robust/acc",              None)
        wmdp_2 = run.summary.get("eval_bench/wmdp_bio_cloze_verified/acc_norm", None)
        wmdp_3 = run.summary.get("eval_bench/wmdp_bio_categorized_mcqa/acc",    None)
        wmdp_4 = run.summary.get("eval_bench/wmdp_bio_robust_rewritten/acc",    None)

        if mmlu is None and wmdp_1 is None and wmdp_2 is None and wmdp_3 is None and wmdp_4 is None:
            continue

        data.append({
            "Run ID":                  run.id,
            "Name":                    run.name,
            "Method":                  method,
            "MMLU":                    mmlu,
            "WMDP (Robust)":           wmdp_1,
            "WMDP (Cloze)":            wmdp_2,
            "WMDP (Categorized)":      wmdp_3,
            "WMDP (Robust Rewritten)": wmdp_4,
            "Loss":                    run.summary.get("train/loss", None),
            # L2 weight distance from base model (logged as wandb summary)
            "L2 Dist":                 run.summary.get("weight_l2_dist", None),
            # NLL on held-out forget/retain split
            "Forget NLL":              run.summary.get("final_forget_nll", None),
            "Retain NLL":              run.summary.get("final_retain_nll", None),
            # Sweep runs have method names like "/tar__", "/cb__", etc. in their name
            # Some runs may use underscore instead of slash (e.g., "_simnpo__")
            # Baselines are plain model runs without any method suffix
            "IsBase":             not any(f"/{m}__" in run.name or f"_{m}__" in run.name for m in KNOWN_METHODS),
            # Timestamp for deduplication — latest run wins when the same
            # config has been run more than once.
            "CreatedAt":               run.created_at,
        })

    if not data:
        print("No finished runs with evaluation metrics found.")
        return

    df = pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Merge runs that share the same display Name.
    #
    # unlearn.py (training) and eval.py (eval subprocess) both call
    # wandb.init() with the same display name, producing two separate runs:
    #   - training run: has weight_l2_dist / NLL, also gets eval_bench
    #     metrics logged back onto it after eval.py finishes
    #   - eval run:     has eval_bench metrics only
    #
    # Simple "keep latest" would drop the training run and lose L2.
    # Instead: group by Name and take the first non-null value per column
    # (rows sorted latest-first, so newer values win when both are non-null).
    # ------------------------------------------------------------------
    df = df.sort_values("CreatedAt", ascending=False)

    def _first_valid(s):
        valid = s.dropna()
        return valid.iloc[0] if not valid.empty else s.iloc[0]

    merged_rows = []
    for name, group in df.groupby("Name", sort=False):
        row = {col: _first_valid(group[col]) for col in df.columns}
        # Timestamp: always the latest run
        row["CreatedAt"] = group["CreatedAt"].iloc[0]
        merged_rows.append(row)

    df = pd.DataFrame(merged_rows)

    # Format timestamp for display — keep timezone explicit
    df["Date (UTC)"] = pd.to_datetime(df["CreatedAt"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    df = df.drop(columns=["CreatedAt"])

    # Calculate a combined score
    # Goal: Maximize MMLU (retain) and Minimize WMDP_Robust (forget)
    # So we want to maximize: (MMLU) - (WMDP_Robust)
    # This gives equal weight to a 1% gain in MMLU and a 1% drop in WMDP
    df["Score"] = df["MMLU"] - df["WMDP (Robust)"]

    print(f"\nSuccessfully processed {len(df)} runs with evaluation metrics.\n")

    cols = ["Name", "Score", "L2 Dist", "MMLU", "WMDP (Robust)", "WMDP (Robust Rewritten)", "WMDP (Cloze)", "WMDP (Categorized)", "Forget NLL", "Retain NLL", "Date (UTC)"]

    baselines_df = df[df["IsBase"]].sort_values("Name")
    sweeps_df    = df[~df["IsBase"]]

    import os
    out_file = os.path.join(os.path.dirname(__file__), "best_unlearning_models.md")
    print(f"Writing results to {out_file}...")

    sweeps_df = sweeps_df.sort_values(
        by=["Method", "Score", "MMLU"],
        ascending=[True, False, False],
    )

    with open(out_file, 'w') as f:
        # ------------------------------------------------------------------ #
        # Cross-method summary table (at the top for quick reference)
        # ------------------------------------------------------------------ #
        summary = _build_summary_table(sweeps_df)
        if summary:
            f.write(summary + "\n")

        # ------------------------------------------------------------------ #
        # Baselines
        # ------------------------------------------------------------------ #
        f.write("## Baselines\n\n")
        if not baselines_df.empty:
            f.write(_md_table(baselines_df, cols) + "\n\n")
        else:
            f.write("No baseline runs found.\n")
            f.write("\nTo generate baselines, run:\n")
            f.write("```bash\n")
            f.write("uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered \\\n")
            f.write("    --wandb-project cambridge_era --wandb-name EleutherAI/deep-ignorance-unfiltered\n")
            f.write("```\n\n")

        # ------------------------------------------------------------------ #
        # Best models by method
        # ------------------------------------------------------------------ #
        f.write("## Best Models By Method\n\n")
        f.write("*Ranked by Score = MMLU - WMDP (Robust)*\n\n")

        for method, group in sweeps_df.groupby("Method"):
            f.write(f"### {method}\n\n")
            best = group.sort_values(
                by=["Score", "MMLU"],
                ascending=[False, False],
            )
            f.write(_md_table(best, cols) + "\n\n")

    print("Done!")


def _build_summary_table(sweeps_df: "pd.DataFrame") -> str:
    """
    For each method, pick the single best run (by Score then MMLU) and
    return a cross-method summary markdown table string.

    Columns: Method | Best Config | L2 Dist | MMLU | WMDP | MMLU-WMDP | Forget NLL | Retain NLL
    """
    import numpy as np

    if sweeps_df.empty:
        print("No sweep runs available – skipping summary table.")
        return ""

    def _f(val, decimals=4):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.{decimals}f}"

    rows = []
    for method, group in sweeps_df.groupby("Method"):
        ranked = group.dropna(subset=["Score"]).sort_values(
            by=["Score", "MMLU"], ascending=[False, False]
        )
        if ranked.empty:
            ranked = group.sort_values("MMLU", ascending=False)
        if ranked.empty:
            continue

        # Prefer runs that have L2 Dist, but only if they're within 0.01
        # Score of the overall best.  If the L2 run is much worse, show
        # the true best honestly (even if L2 is N/A for that row).
        best_overall = ranked.iloc[0]
        best_score   = best_overall.get("Score") or 0
        with_l2 = ranked[ranked["L2 Dist"].notna()]
        if not with_l2.empty and (best_score - with_l2.iloc[0].get("Score", 0)) <= 0.01:
            best = with_l2.iloc[0]
        else:
            best = best_overall
        mmlu    = best.get("MMLU")
        wmdp    = best.get("WMDP (Robust)")
        l2      = best.get("L2 Dist")
        fgt_nll = best.get("Forget NLL")
        ret_nll = best.get("Retain NLL")

        mmlu_minus_wmdp = (
            f"{mmlu - wmdp:.4f}"
            if (mmlu is not None and wmdp is not None
                and not np.isnan(float(mmlu)) and not np.isnan(float(wmdp)))
            else "N/A"
        )

        rows.append([
            method,
            _expand_config(_extract_best_config(best["Name"])),
            _f(l2, 2),
            _f(mmlu),
            _f(wmdp),
            mmlu_minus_wmdp,
            _f(fgt_nll, 3),
            _f(ret_nll, 3),
            str(best.get("Date (UTC)", "N/A")),
            str(best.get("Name", "")),
        ])

    if not rows:
        print("No rows for summary table.")
        return ""

    headers = ["Method", "Best Config", "L2 Dist", "MMLU", "WMDP (Robust)", "MMLU−WMDP (Robust)", "Forget NLL", "Retain NLL", "Date (UTC)", "Run Name"]
    header  = "| " + " | ".join(headers) + " |"
    sep     = "| " + " | ".join(["---"] * len(headers)) + " |"
    body    = ["| " + " | ".join(r) + " |" for r in rows]
    table   = "\n".join([header, sep] + body)

    result = "## Cross-Method Comparison — Best Config Per Method\n\n"
    result += "*Best run per method ranked by Score = MMLU − WMDP (Robust)*\n\n"
    result += table + "\n"
    return result


if __name__ == "__main__":
    main()
