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
            # Sweep runs always have "__<method>__" in their name (e.g. "__tar__ta1.0_…",
            # "__cb__ep3_…").  Baselines are plain model-name-only runs with no "/__" segment.
            "IsBase":             "/__" not in run.name,
        })

    if not data:
        print("No finished runs with evaluation metrics found.")
        return

    df = pd.DataFrame(data)

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

