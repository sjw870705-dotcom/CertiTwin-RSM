import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAFE_SET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
)

INPUT_CANDIDATES = SAFE_SET_DIR / "step1_5_safe_labeled_candidates_patched.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step2_1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

Q_LEVEL = 90

EPS_EMBB = 0.29489304824809814
EPS_URLLC = 0.09153290412936986

REQUIRED_COLUMNS = [
    "candidate_group_id",
    "candidate_id",

    "candidate_embb_slice_prb",
    "candidate_urllc_slice_prb",
    "candidate_embb_share",
    "candidate_urllc_share",

    "true_V_embb",
    "true_V_urllc",
    "true_V_total",
    "true_management_utility",

    "pred_mean_V_embb",
    "pred_mean_V_urllc",
    "pred_mean_V_total",
    "pred_mean_management_utility",

    "pred_std_V_embb",
    "pred_std_V_urllc",
    "pred_std_V_total",

    "upper_V_embb_q90",
    "upper_V_urllc_q90",
    "upper_V_total_q90",

    "true_joint_safe",
    "raw_pred_joint_safe",
    "cert_joint_safe",
]


def check_required_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def add_controller_scores(df):
    out = df.copy()

    # Predicted risk scores.
    out["pred_joint_risk"] = (
        0.5 * out["pred_mean_V_embb"]
        + 0.5 * out["pred_mean_V_urllc"]
    )

    out["upper_joint_risk_q90"] = (
        0.5 * out["upper_V_embb_q90"]
        + 0.5 * out["upper_V_urllc_q90"]
    )

    out["true_joint_risk"] = (
        0.5 * out["true_V_embb"]
        + 0.5 * out["true_V_urllc"]
    )

    # Utility-risk score for DT/Greedy-like controllers.
    out["pred_balanced_score"] = (
        out["pred_mean_management_utility"]
        - 0.5 * out["pred_mean_V_embb"]
        - 0.5 * out["pred_mean_V_urllc"]
    )

    out["upper_balanced_score"] = (
        out["pred_mean_management_utility"]
        - 0.5 * out["upper_V_embb_q90"]
        - 0.5 * out["upper_V_urllc_q90"]
    )

    # PF-like score: approximate proportional fairness using normalized candidate shares.
    # This is not final PF scheduling, but provides a deterministic interface-ready score.
    out["pf_like_score"] = (
        np.log1p(out["candidate_embb_share"].clip(lower=0))
        + np.log1p(out["candidate_urllc_share"].clip(lower=0))
    )

    # Greedy-SLA score: prioritize certified/safe low risk, then utility.
    out["greedy_sla_score"] = (
        out["pred_mean_management_utility"]
        - 1.0 * out["upper_joint_risk_q90"]
    )

    # Candidate action distance from group default / candidate 0.
    # This will help later compute action deviation.
    first_actions = (
        out.sort_values(["candidate_group_id", "candidate_id"])
        .groupby("candidate_group_id")
        .first()[["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]]
        .rename(columns={
            "candidate_embb_slice_prb": "group_ref_embb_prb",
            "candidate_urllc_slice_prb": "group_ref_urllc_prb",
        })
        .reset_index()
    )

    out = out.merge(first_actions, on="candidate_group_id", how="left")

    out["distance_to_group_ref"] = (
        (out["candidate_embb_slice_prb"] - out["group_ref_embb_prb"]).abs()
        + (out["candidate_urllc_slice_prb"] - out["group_ref_urllc_prb"]).abs()
    )

    return out


def summarize_groups(df):
    g = (
        df.groupby("candidate_group_id")
        .agg(
            num_candidates=("candidate_id", "count"),
            num_true_safe=("true_joint_safe", "sum"),
            num_raw_pred_safe=("raw_pred_joint_safe", "sum"),
            num_cert_safe=("cert_joint_safe", "sum"),
            max_true_utility=("true_management_utility", "max"),
            max_pred_utility=("pred_mean_management_utility", "max"),
            min_pred_risk=("pred_joint_risk", "min"),
            min_upper_risk=("upper_joint_risk_q90", "min"),
        )
        .reset_index()
    )

    g["has_true_safe"] = (g["num_true_safe"] > 0).astype(int)
    g["has_raw_pred_safe"] = (g["num_raw_pred_safe"] > 0).astype(int)
    g["has_cert_safe"] = (g["num_cert_safe"] > 0).astype(int)

    return g


def main():
    print(f"Reading patched safe candidates: {INPUT_CANDIDATES}")
    df = pd.read_csv(INPUT_CANDIDATES)

    check_required_columns(df)

    print(f"Candidate rows: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")

    df = add_controller_scores(df)

    # Ensure each group has the same number of candidates.
    group_summary = summarize_groups(df)

    output_candidates = OUT_DIR / "controller_candidate_groups.csv"
    output_group_summary = OUT_DIR / "controller_group_summary.csv"

    df.to_csv(output_candidates, index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_group_summary, index=False, encoding="utf-8-sig")

    metadata = {
        "input_candidates": str(INPUT_CANDIDATES),
        "output_candidates": str(output_candidates),
        "output_group_summary": str(output_group_summary),
        "num_candidate_rows": int(len(df)),
        "num_candidate_groups": int(df["candidate_group_id"].nunique()),
        "mean_candidates_per_group": float(group_summary["num_candidates"].mean()),
        "q_level": Q_LEVEL,
        "eps_embb": EPS_EMBB,
        "eps_urllc": EPS_URLLC,
        "controller_interface": {
            "input": "state + candidate_set",
            "output": "raw_action_id",
            "action_id_field": "candidate_id",
            "group_field": "candidate_group_id",
            "required_selection_rule": "Each raw controller must select exactly one candidate_id per candidate_group_id.",
        },
        "planned_controllers": [
            "PF",
            "Greedy-SLA",
            "DT-Top1",
            "PPO",
            "SAC",
            "PPO-Penalty",
        ],
        "available_scores_for_classical_controllers": [
            "pf_like_score",
            "greedy_sla_score",
            "pred_mean_management_utility",
            "pred_joint_risk",
            "upper_joint_risk_q90",
            "pred_balanced_score",
            "upper_balanced_score",
        ],
        "group_quality": {
            "group_has_true_safe_rate": float(group_summary["has_true_safe"].mean()),
            "group_has_raw_pred_safe_rate": float(group_summary["has_raw_pred_safe"].mean()),
            "group_has_cert_safe_rate": float(group_summary["has_cert_safe"].mean()),
            "no_cert_safe_group_rate": float((group_summary["num_cert_safe"] == 0).mean()),
            "mean_cert_safe_candidates_per_group": float(group_summary["num_cert_safe"].mean()),
        },
        "important_note": (
            "Step 2.1 freezes the controller action interface. "
            "All raw controllers must select one candidate from the same candidate group. "
            "CertiTwin shielding will then check and project the selected raw action."
        )
    }

    metadata_path = OUT_DIR / "controller_interface_metadata.json"
    report_json_path = OUT_DIR / "step2_1_interface_report.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_md = RESULT_DIR / "step2_1_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Step 2.1 Raw Controller Action Interface Report\n\n")

        f.write("## Interface\n\n")
        f.write("- Input: state + candidate set\n")
        f.write("- Output: one raw action id per candidate group\n")
        f.write("- Action field: `candidate_id`\n")
        f.write("- Group field: `candidate_group_id`\n")
        f.write("- Requirement: every raw controller selects exactly one candidate per group\n\n")

        f.write("## Dataset summary\n\n")
        f.write(f"- candidate rows: {len(df)}\n")
        f.write(f"- candidate groups: {df['candidate_group_id'].nunique()}\n")
        f.write(f"- mean candidates per group: {group_summary['num_candidates'].mean()}\n\n")

        f.write("## Group quality\n\n")
        for k, v in metadata["group_quality"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Planned controllers\n\n")
        for c in metadata["planned_controllers"]:
            f.write(f"- {c}\n")

        f.write("\n## Output files\n\n")
        f.write(f"- controller candidate groups: `{output_candidates}`\n")
        f.write(f"- controller group summary: `{output_group_summary}`\n")
        f.write(f"- metadata: `{metadata_path}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This step does not evaluate controllers yet.\n")
        f.write("- It only freezes the common interface and candidate-set input used by all controllers.\n")
        f.write("- Step 2.2 will implement PF, Greedy-SLA, and DT-Top1 under this interface.\n")

    print("Step 2.1 completed.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()