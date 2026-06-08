import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

PPO_PENALTY_MODEL_DIR = PROJECT_ROOT / "models" / "ppo_raw_controller"
RESULT_DIR = PROJECT_ROOT / "results" / "step2_6"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_METRICS = PPO_PENALTY_MODEL_DIR / "step2_5_ppo_metrics.json"
INPUT_PAIRWISE = EVAL_DIR / "step2_5_ppo_raw_vs_shielded_pairwise.csv"
INPUT_SUMMARY = EVAL_DIR / "step2_5_ppo_shielding_summary.csv"

OUTPUT_PAIRWISE = EVAL_DIR / "step2_6_ppo_penalty_pairwise.csv"
OUTPUT_SUMMARY = EVAL_DIR / "step2_6_ppo_penalty_summary.csv"
OUTPUT_REPORT_JSON = EVAL_DIR / "step2_6_ppo_penalty_report.json"
OUTPUT_REPORT_MD = RESULT_DIR / "step2_6_report.md"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main():
    for p in [INPUT_METRICS, INPUT_PAIRWISE, INPUT_SUMMARY]:
        assert_exists(p)

    metrics = load_json(INPUT_METRICS)
    pairwise = pd.read_csv(INPUT_PAIRWISE)
    summary = pd.read_csv(INPUT_SUMMARY)

    # Rename controller labels from generic PPO to PPO-Penalty / PPO-Safe.
    pairwise = pairwise.copy()
    summary = summary.copy()

    if "controller" in pairwise.columns:
        pairwise["controller"] = "PPO-Penalty"
    if "shielded_controller" in pairwise.columns:
        pairwise["shielded_controller"] = "CertiTwin-PPO-Penalty"

    if "controller" in summary.columns:
        summary["controller"] = "PPO-Penalty"
    if "shielded_controller" in summary.columns:
        summary["shielded_controller"] = "CertiTwin-PPO-Penalty"

    # Add explicit reward description.
    summary["reward_type"] = "risk_penalized"
    summary["lambda_risk"] = 0.80
    summary["lambda_unsafe"] = 0.30
    summary["lambda_reconfig"] = 0.05
    summary["baseline_role"] = (
        "Constrained/risk-penalized DRL baseline; used to verify that CertiTwin "
        "does not over-modify an already safe controller."
    )

    pairwise.to_csv(OUTPUT_PAIRWISE, index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    row = summary.iloc[0].to_dict()

    report = {
        "source_step": "Step 2.5",
        "source_metrics": str(INPUT_METRICS),
        "source_pairwise": str(INPUT_PAIRWISE),
        "source_summary": str(INPUT_SUMMARY),
        "renamed_controller": "PPO-Penalty",
        "renamed_shielded_controller": "CertiTwin-PPO-Penalty",
        "reward_definition": {
            "type": "risk_penalized",
            "expression": "utility - 0.8 * joint_risk - 0.3 * unsafe_penalty - 0.05 * reconfiguration",
            "lambda_risk": 0.80,
            "lambda_unsafe": 0.30,
            "lambda_reconfig": 0.05,
        },
        "raw_test_metrics": metrics.get("test_metrics", {}),
        "shielding_summary": row,
        "outputs": {
            "ppo_penalty_pairwise": str(OUTPUT_PAIRWISE),
            "ppo_penalty_summary": str(OUTPUT_SUMMARY),
            "ppo_penalty_report": str(OUTPUT_REPORT_JSON),
        },
        "important_note": (
            "This step does not retrain PPO. It formalizes the Step 2.5 risk-penalized PPO "
            "as PPO-Penalty / PPO-Safe. Its role is to show that CertiTwin passes through "
            "an already safe controller without unnecessary intervention."
        ),
    }

    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.6 PPO-Penalty / PPO-Safe Baseline Report\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "This step formalizes the previous risk-penalized PPO run as a constrained DRL baseline. "
            "It is used to verify that CertiTwin does not over-modify an already safe raw controller.\n\n"
        )

        f.write("## Reward definition\n\n")
        f.write("- reward type: risk_penalized\n")
        f.write("- expression: utility - 0.8 * joint_risk - 0.3 * unsafe_penalty - 0.05 * reconfiguration\n")
        f.write("- lambda_risk: 0.80\n")
        f.write("- lambda_unsafe: 0.30\n")
        f.write("- lambda_reconfig: 0.05\n\n")

        f.write("## PPO-Penalty raw test metrics\n\n")
        for k, v in metrics.get("test_metrics", {}).items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## PPO-Penalty vs CertiTwin-PPO-Penalty\n\n")
        for k, v in row.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- PPO-Penalty and CertiTwin-PPO-Penalty use the same trained PPO checkpoint.\n")
        f.write("- CertiTwin only checks and potentially modifies the PPO-Penalty raw action at execution time.\n")
        f.write("- Since PPO-Penalty is already safe, CertiTwin should pass through almost all actions.\n")
        f.write("- This baseline is complementary to PPO-Utility, where CertiTwin demonstrates unsafe-action reduction.\n")

    print("Step 2.6 completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()