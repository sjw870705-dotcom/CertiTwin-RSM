import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step2_7"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
CLASSICAL_SUMMARY = EVAL_DIR / "step2_3_classical_shielding_summary.csv"
PPO_UTILITY_SUMMARY = EVAL_DIR / "step2_5b_ppo_utility_shielding_summary.csv"
PPO_PENALTY_SUMMARY = EVAL_DIR / "step2_6_ppo_penalty_summary.csv"

# Outputs
OUT_MAIN = RESULT_DIR / "table_step2_7_main_raw_vs_certitwin.csv"
OUT_PAPER = RESULT_DIR / "table_step2_7_main_raw_vs_certitwin_paper.csv"
OUT_JSON = RESULT_DIR / "step2_7_main_summary.json"
OUT_MD = RESULT_DIR / "step2_7_report.md"


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def normalize_controller_names(df):
    df = df.copy()

    name_map = {
        "PF": "PF",
        "Greedy-SLA": "Greedy-SLA",
        "DT-Top1": "DT-Top1",
        "PPO-Utility": "PPO",
        "PPO-Penalty": "PPO-Penalty",
    }

    shield_name_map = {
        "CertiTwin-PF": "CertiTwin-PF",
        "CertiTwin-Greedy-SLA": "CertiTwin-Greedy-SLA",
        "CertiTwin-DT-Top1": "CertiTwin-DT",
        "CertiTwin-PPO-Utility": "CertiTwin-PPO",
        "CertiTwin-PPO-Penalty": "CertiTwin-PPO-Penalty",
    }

    if "controller" in df.columns:
        df["controller"] = df["controller"].replace(name_map)

    if "shielded_controller" in df.columns:
        df["shielded_controller"] = df["shielded_controller"].replace(shield_name_map)

    return df


def load_all_results():
    for p in [CLASSICAL_SUMMARY, PPO_UTILITY_SUMMARY, PPO_PENALTY_SUMMARY]:
        assert_exists(p)

    classical = pd.read_csv(CLASSICAL_SUMMARY)
    ppo_utility = pd.read_csv(PPO_UTILITY_SUMMARY)
    ppo_penalty = pd.read_csv(PPO_PENALTY_SUMMARY)

    classical = normalize_controller_names(classical)
    ppo_utility = normalize_controller_names(ppo_utility)
    ppo_penalty = normalize_controller_names(ppo_penalty)

    all_df = pd.concat([classical, ppo_utility, ppo_penalty], ignore_index=True)

    order = {
        "PF": 1,
        "Greedy-SLA": 2,
        "DT-Top1": 3,
        "PPO": 4,
        "PPO-Penalty": 5,
    }

    all_df["order"] = all_df["controller"].map(order).fillna(99)
    all_df = all_df.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    return all_df


def add_derived_metrics(df):
    df = df.copy()

    # Normalize important missing columns if any.
    for c in [
        "raw_true_unsafe_rate",
        "shielded_true_unsafe_rate",
        "unsafe_reduction_abs",
        "unsafe_reduction_rel",
        "utility_raw_mean",
        "utility_shielded_mean",
        "utility_retention",
        "action_changed_rate",
        "pass_through_rate",
        "fallback_rate",
        "raw_V_embb_mean",
        "shielded_V_embb_mean",
        "raw_V_urllc_mean",
        "shielded_V_urllc_mean",
        "raw_V_total_mean",
        "shielded_V_total_mean",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # If unsafe reduction was not already present, compute it.
    df["unsafe_reduction_abs"] = df["raw_true_unsafe_rate"] - df["shielded_true_unsafe_rate"]

    df["unsafe_reduction_rel"] = np.where(
        df["raw_true_unsafe_rate"] > 1e-12,
        df["unsafe_reduction_abs"] / df["raw_true_unsafe_rate"],
        0.0,
    )

    df["utility_loss_abs"] = df["utility_raw_mean"] - df["utility_shielded_mean"]

    df["V_embb_change"] = df["shielded_V_embb_mean"] - df["raw_V_embb_mean"]
    df["V_urllc_change"] = df["shielded_V_urllc_mean"] - df["raw_V_urllc_mean"]
    df["V_total_change"] = df["shielded_V_total_mean"] - df["raw_V_total_mean"]

    return df


def make_paper_table(df):
    paper = df.copy()

    # Convert to percentage-style values.
    paper["Raw unsafe (%)"] = 100 * paper["raw_true_unsafe_rate"]
    paper["Shielded unsafe (%)"] = 100 * paper["shielded_true_unsafe_rate"]
    paper["Unsafe reduction (%)"] = 100 * paper["unsafe_reduction_rel"]
    paper["Utility retention (%)"] = 100 * paper["utility_retention"]
    paper["Action changed (%)"] = 100 * paper["action_changed_rate"]
    paper["Pass-through (%)"] = 100 * paper["pass_through_rate"]
    paper["Fallback (%)"] = 100 * paper["fallback_rate"]

    paper["Raw utility"] = paper["utility_raw_mean"]
    paper["Shielded utility"] = paper["utility_shielded_mean"]

    paper = paper[
        [
            "controller",
            "shielded_controller",
            "Raw unsafe (%)",
            "Shielded unsafe (%)",
            "Unsafe reduction (%)",
            "Raw utility",
            "Shielded utility",
            "Utility retention (%)",
            "Action changed (%)",
            "Pass-through (%)",
            "Fallback (%)",
        ]
    ].copy()

    rename = {
        "controller": "Raw controller",
        "shielded_controller": "Shielded controller",
    }
    paper = paper.rename(columns=rename)

    # Round for paper table.
    for c in paper.columns:
        if c not in ["Raw controller", "Shielded controller"]:
            paper[c] = paper[c].astype(float).round(4)

    return paper


def evaluate_table_quality(df):
    checks = {}

    required_controllers = {"PF", "Greedy-SLA", "DT-Top1", "PPO", "PPO-Penalty"}
    actual_controllers = set(df["controller"].tolist())

    checks["has_all_required_controllers"] = bool(required_controllers.issubset(actual_controllers))

    checks["fallback_rate_ok"] = bool((df["fallback_rate"] <= 0.05).all())

    checks["utility_retention_ok"] = bool((df["utility_retention"] >= 0.90).all())

    # For controllers with non-zero raw unsafe, shielded unsafe should be much lower.
    risky = df[df["raw_true_unsafe_rate"] > 0.01]
    if len(risky) > 0:
        checks["risky_controllers_improved"] = bool(
            (risky["shielded_true_unsafe_rate"] < risky["raw_true_unsafe_rate"]).all()
        )
        checks["risky_controllers_large_reduction"] = bool(
            (risky["unsafe_reduction_rel"] >= 0.90).all()
        )
    else:
        checks["risky_controllers_improved"] = False
        checks["risky_controllers_large_reduction"] = False

    # Safe controllers should not be over-modified.
    safeish = df[df["raw_true_unsafe_rate"] <= 0.01]
    if len(safeish) > 0:
        checks["safe_controllers_not_overmodified"] = bool(
            (safeish["action_changed_rate"] <= 0.05).all()
        )
    else:
        checks["safe_controllers_not_overmodified"] = True

    checks["overall_main_table_ready"] = bool(all(checks.values()))

    return checks


def headline_summary(df):
    risky = df[df["raw_true_unsafe_rate"] > 0.01]

    summary = {
        "num_rows": int(len(df)),
        "controllers": df["controller"].tolist(),
        "max_raw_unsafe_rate": float(df["raw_true_unsafe_rate"].max()),
        "max_shielded_unsafe_rate": float(df["shielded_true_unsafe_rate"].max()),
        "min_utility_retention": float(df["utility_retention"].min()),
        "max_fallback_rate": float(df["fallback_rate"].max()),
        "mean_utility_retention": float(df["utility_retention"].mean()),
    }

    if len(risky) > 0:
        summary["mean_unsafe_reduction_rel_on_risky_controllers"] = float(
            risky["unsafe_reduction_rel"].mean()
        )
        summary["min_unsafe_reduction_rel_on_risky_controllers"] = float(
            risky["unsafe_reduction_rel"].min()
        )

    return summary


def main():
    df = load_all_results()
    df = add_derived_metrics(df)

    paper_table = make_paper_table(df)
    checks = evaluate_table_quality(df)
    headlines = headline_summary(df)

    df.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")
    paper_table.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

    summary = {
        "input_files": {
            "classical_summary": str(CLASSICAL_SUMMARY),
            "ppo_utility_summary": str(PPO_UTILITY_SUMMARY),
            "ppo_penalty_summary": str(PPO_PENALTY_SUMMARY),
        },
        "headline_summary": headlines,
        "quality_checks": checks,
        "outputs": {
            "main_table": str(OUT_MAIN),
            "paper_table": str(OUT_PAPER),
            "report_md": str(OUT_MD),
        },
        "important_note": (
            "Step 2.7 aggregates the main raw-vs-CertiTwin paired comparison table. "
            "This table is the current main experimental result before optional SAC or robustness experiments."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.7 Main Raw-vs-CertiTwin Result Aggregation\n\n")

        f.write("## Aggregated controllers\n\n")
        for c in df["controller"].tolist():
            shielded = df[df["controller"] == c]["shielded_controller"].iloc[0]
            f.write(f"- {c} vs {shielded}\n")

        f.write("\n## Headline summary\n\n")
        for k, v in headlines.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Quality checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Main paper table\n\n")
        f.write(paper_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Interpretation\n\n")
        f.write("- This table combines classical, digital-twin, utility-oriented PPO, and PPO-Penalty baselines.\n")
        f.write("- CertiTwin is evaluated as an execution-time shield using the same raw controller actions.\n")
        f.write("- The key expected pattern is: large unsafe reduction for risky controllers, high utility retention, and no over-modification for already safe controllers.\n")

    print("Step 2.7 completed.")
    print(json.dumps(summary, indent=2))
    print("\nPaper table:")
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()