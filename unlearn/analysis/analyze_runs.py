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
            pattern = r'[/_](tar|cb|rmu|npo|simnpo|wt_dist|ga|grad_diff)__'
            match = re.search(pattern, run.name)
            if match:
                method = match.group(1)

        mmlu   = run.summary.get("eval_bench/mmlu/acc",                     None)
        wmdp_1 = run.summary.get("eval_bench/wmdp_bio_robust/acc",          None)
        wmdp_2 = run.summary.get("eval_bench/wmdp_bio_cloze_verified/acc_norm", None)
        wmdp_3 = run.summary.get("eval_bench/wmdp_bio_categorized_mcqa/acc", None)

        if mmlu is None and wmdp_1 is None and wmdp_2 is None and wmdp_3 is None:
            continue

        data.append({
            "Run ID":             run.id,
            "Name":               run.name,
            "Method":             method,
            "MMLU":               mmlu,
            "WMDP (Robust)":      wmdp_1,
            "WMDP (Cloze)":       wmdp_2,
            "WMDP (Categorized)": wmdp_3,
            "Loss":               run.summary.get("train/loss", None),
            # Sweep runs have method names like "/tar__", "/cb__", etc. in their name
            # Some runs may use underscore instead of slash (e.g., "_simnpo__")
            # Baselines are plain model runs without any method suffix
            "IsBase":             not any(f"/{method}__" in run.name or f"_{method}__" in run.name for method in ["tar", "cb", "rmu", "npo", "simnpo", "wt_dist"]),
        })

    if not data:
        print("No finished runs with evaluation metrics found.")
        return

    df = pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Deduplicate runs that represent the same trained model.
    #
    # Two naming schemes exist across code versions:
    #   Old  →  "unlearned_models_EleutherAI_deep-ignorance-unfiltered_npo__..."
    #   New  →  "EleutherAI_deep-ignorance-unfiltered/npo__..."
    #
    # Runs from both eras may appear in the same project with identical
    # metrics.  We detect them by rounding MMLU+WMDP to 4 dp (runs with
    # exactly the same checkpoint produce bit-identical eval results), then
    # keep whichever name contains a "/" (new-style) over the flat one.
    # ------------------------------------------------------------------
    def _name_style_rank(name: str) -> int:
        """Return 0 for new-style (has '/'), 1 for old-style."""
        return 0 if "/" in name else 1

    df["_style"] = df["Name"].apply(_name_style_rank)
    df["_fp"] = (
        df["MMLU"].round(4).astype(str)
        + "|" + df["WMDP (Robust)"].round(4).astype(str)
        + "|" + df["WMDP (Cloze)"].round(4).astype(str)
        + "|" + df["WMDP (Categorized)"].round(4).astype(str)
    )
    df = (
        df.sort_values("_style")          # new-style (0) first
          .drop_duplicates(subset="_fp", keep="first")
          .drop(columns=["_style", "_fp"])
    )

    # Calculate a combined score
    # Goal: Maximize MMLU (retain) and Minimize WMDP_Robust (forget)
    # So we want to maximize: (MMLU) - (WMDP_Robust)
    # This gives equal weight to a 1% gain in MMLU and a 1% drop in WMDP
    df["Score"] = df["MMLU"] - df["WMDP (Robust)"]

    print(f"\nSuccessfully processed {len(df)} runs with evaluation metrics.\n")

    cols = ["Name", "Score", "MMLU", "WMDP (Robust)", "WMDP (Cloze)", "WMDP (Categorized)"]

    baselines_df = df[df["IsBase"]].sort_values("Name")
    sweeps_df    = df[~df["IsBase"]]

    import os
    out_file = os.path.join(os.path.dirname(__file__), "best_unlearning_models.md")
    print(f"Writing results to {out_file}...")

    with open(out_file, 'w') as f:
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

        sweeps_df = sweeps_df.sort_values(
            by=["Method", "Score", "MMLU"],
            ascending=[True, False, False],
        )

        for method, group in sweeps_df.groupby("Method"):
            f.write(f"### {method}\n\n")
            best = group.sort_values(
                by=["Score", "MMLU"],
                ascending=[False, False],
            ).head(5)
            f.write(_md_table(best, cols) + "\n\n")

    print("Done!")

if __name__ == "__main__":
    main()

