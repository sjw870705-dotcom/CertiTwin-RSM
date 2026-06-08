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

RESULT_DIR = PROJECT_ROOT / "results" / "step2_2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = EVAL_DIR / "controller_candidate_groups.csv"

OUTPUT_ACTIONS = EVAL_DIR / "raw_controller_actions_classical.csv"
OUTPUT_SUMMARY = EVAL_DIR / "raw_controller_actions_classical_summary.csv"
OUTPUT_REPORT_JSON = EVAL_DIR / "step2_2_raw_controller_report.json"
OUTPUT_REPORT_MD = RESULT_DIR / "step2_2_report.md"


CONTROLLERS = {
    "PF": "pf_like_score",
    "Greedy-SLA": "greedy_sla_score",
    "DT-Top1": "pred_mean_management_utility",
}


REQUIRED_COLUMNS = [
    "candidate_group_id",
    "candidate_id",
    "candidate_embb_slice_prb",
    "candidate_urllc_slice_prb",

    "true_V_embb",
    "true_V_urllc",
    "true_V_total",
    "true_management_utility",

    "pred_mean_V_embb",
    "pred_mean_V_urllc",
    "pred_mean_V_total",
    "pred_mean_management_utility",

    "upper_V_embb_q90",
    "upper_V_urllc_q90",
    "upper_V_total_q90",

    "true_joint_safe",
    "raw_pred_joint_safe",
    "cert_joint_safe",

    "pf_like_score",
    "greedy_sla_score",
]


def check_required_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def select_raw_actions(df, controller_name, score_col):
    # Select the candidate with highest score in each group.
    idx = df.groupby("candidate_group_id")[score_col].idxmax()
    selected = df.loc[idx].copy()

    selected["controller"] = controller_name
    selected["raw_candidate_id"] = selected["candidate_id"]
    selected["raw_score_col"] = score_col
    selected["raw_score"] = selected[score_col]

    # Rename fields explicitly for raw controller output.
    rename_map = {
        "candidate_embb_slice_prb": "raw_embb_slice_prb",
        "candidate_urllc_slice_prb": "raw_urllc_slice_prb",

        "true_V_embb": "raw_true_V_embb",
        "true_V_urllc": "raw_true_V_urllc",
        "true_V_total": "raw_true_V_total",
        "true_management_utility": "raw_true_management_utility",

        "pred_mean_V_embb": "raw_pred_mean_V_embb",
        "pred_mean_V_urllc": "raw_pred_mean_V_urllc",
        "pred_mean_V_total": "raw_pred_mean_V_total",
        "pred_mean_management_utility": "raw_pred_mean_management_utility",

        "upper_V_embb_q90": "raw_upper_V_embb_q90",
        "upper_V_urllc_q90": "raw_upper_V_urllc_q90",
        "upper_V_total_q90": "raw_upper_V_total_q90",

        "true_joint_safe": "raw_true_joint_safe",
        "raw_pred_joint_safe": "raw_pred_joint_safe",
        "cert_joint_safe": "raw_cert_joint_safe",
    }

    selected = selected.rename(columns=rename_map)

    keep_cols = [
        "controller",
        "candidate_group_id",
        "raw_candidate_id",
        "raw_score_col",
        "raw_score",

        "raw_embb_slice_prb",
        "raw_urllc_slice_prb",

        "raw_true_V_embb",
        "raw_true_V_urllc",
        "raw_true_V_total",
        "raw_true_management_utility",

        "raw_pred_mean_V_embb",
        "raw_pred_mean_V_urllc",
        "raw_pred_mean_V_total",
        "raw_pred_mean_management_utility",

        "raw_upper_V_embb_q90",
        "raw_upper_V_urllc_q90",
        "raw_upper_V_total_q90",

        "raw_true_joint_safe",
        "raw_pred_joint_safe",
        "raw_cert_joint_safe",
    ]

    keep_cols = [c for c in keep_cols if c in selected.columns]
    return selected[keep_cols].copy()


def summarize_controller(selected):
    rows = []

    for controller, g in selected.groupby("controller"):
        raw_true_safe = g["raw_true_joint_safe"].astype(int).to_numpy()
        raw_cert_safe = g["raw_cert_joint_safe"].astype(int).to_numpy()
        raw_pred_safe = g["raw_pred_joint_safe"].astype(int).to_numpy()

        rec = {
            "controller": controller,
            "num_groups": int(len(g)),

            "raw_true_safe_rate": float(raw_true_safe.mean()),
            "raw_true_unsafe_rate": float(1.0 - raw_true_safe.mean()),

            "raw_certified_safe_rate": float(raw_cert_safe.mean()),
            "raw_certified_unsafe_rate": float(1.0 - raw_cert_safe.mean()),

            "raw_pred_safe_rate": float(raw_pred_safe.mean()),

            "raw_true_V_embb_mean": float(g["raw_true_V_embb"].mean()),
            "raw_true_V_urllc_mean": float(g["raw_true_V_urllc"].mean()),
            "raw_true_V_total_mean": float(g["raw_true_V_total"].mean()),
            "raw_true_management_utility_mean": float(g["raw_true_management_utility"].mean()),

            "raw_embb_slice_prb_mean": float(g["raw_embb_slice_prb"].mean()),
            "raw_urllc_slice_prb_mean": float(g["raw_urllc_slice_prb"].mean()),
            "raw_embb_slice_prb_p95": float(g["raw_embb_slice_prb"].quantile(0.95)),
            "raw_urllc_slice_prb_p95": float(g["raw_urllc_slice_prb"].quantile(0.95)),
        }

        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    print(f"Reading controller candidate groups: {INPUT_CANDIDATES}")
    df = pd.read_csv(INPUT_CANDIDATES)

    check_required_columns(df)

    print(f"Candidate rows: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")

    selected_list = []

    for controller_name, score_col in CONTROLLERS.items():
        print(f"Selecting raw actions for {controller_name} using {score_col}")
        selected = select_raw_actions(df, controller_name, score_col)
        selected_list.append(selected)

    raw_actions = pd.concat(selected_list, ignore_index=True)

    # Check every controller selected exactly one action per group.
    group_counts = (
        raw_actions.groupby(["controller", "candidate_group_id"])
        .size()
        .reset_index(name="count")
    )

    if not (group_counts["count"] == 1).all():
        raise RuntimeError("Some controller/group pairs do not have exactly one selected action.")

    summary_df = summarize_controller(raw_actions)

    raw_actions.to_csv(OUTPUT_ACTIONS, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    report = {
        "input_candidates": str(INPUT_CANDIDATES),
        "num_candidate_rows": int(len(df)),
        "num_candidate_groups": int(df["candidate_group_id"].nunique()),
        "controllers": list(CONTROLLERS.keys()),
        "controller_score_columns": CONTROLLERS,
        "num_raw_action_rows": int(len(raw_actions)),
        "expected_raw_action_rows": int(df["candidate_group_id"].nunique() * len(CONTROLLERS)),
        "selection_check_exactly_one_per_group": True,
        "outputs": {
            "raw_controller_actions": str(OUTPUT_ACTIONS),
            "raw_controller_summary": str(OUTPUT_SUMMARY),
        },
        "important_note": (
            "Step 2.2 implements non-training raw controllers under the frozen Step 2.1 interface. "
            "CertiTwin shielding will be applied in Step 2.3 using the same candidate groups and selected raw actions."
        ),
    }

    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.2 Classical Raw Controller Actions Report\n\n")

        f.write("## Controllers\n\n")
        for controller_name, score_col in CONTROLLERS.items():
            f.write(f"- {controller_name}: selects max `{score_col}`\n")

        f.write("\n## Dataset summary\n\n")
        f.write(f"- candidate rows: {len(df)}\n")
        f.write(f"- candidate groups: {df['candidate_group_id'].nunique()}\n")
        f.write(f"- raw action rows: {len(raw_actions)}\n")
        f.write(f"- expected raw action rows: {report['expected_raw_action_rows']}\n")
        f.write(f"- exactly one action per controller/group: True\n\n")

        f.write("## Raw controller summary\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"### {row['controller']}\n")
            f.write(f"- raw true unsafe rate: {row['raw_true_unsafe_rate']}\n")
            f.write(f"- raw certified unsafe rate: {row['raw_certified_unsafe_rate']}\n")
            f.write(f"- utility mean: {row['raw_true_management_utility_mean']}\n")
            f.write(f"- V_embb mean: {row['raw_true_V_embb_mean']}\n")
            f.write(f"- V_urllc mean: {row['raw_true_V_urllc_mean']}\n\n")

        f.write("## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This step evaluates only raw action selection, not shielding.\n")
        f.write("- Step 2.3 will compare each raw controller against its CertiTwin-shielded version.\n")

    print("Step 2.2 completed.")
    print(json.dumps(report, indent=2))
    print("\nRaw controller summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()