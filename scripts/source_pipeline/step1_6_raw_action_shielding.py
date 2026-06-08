import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAFE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_shielding"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step1_6"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = SAFE_DIR / "step1_5_safe_labeled_candidates_patched.csv"

Q_LEVEL = 90

# Same thresholds as Step 1.5 summary.
EPS_EMBB = 0.29489304824809814
EPS_URLLC = 0.09153290412936986

# Projection tradeoff.
# Higher lambda_distance means closer-to-raw action is preferred.
LAMBDA_DISTANCE = 0.02
LAMBDA_UPPER_RISK = 0.50


def check_required_columns(df):
    required = [
        "candidate_group_id",
        "candidate_id",
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
        "true_V_embb",
        "true_V_urllc",
        "true_management_utility",
        "pred_mean_V_embb",
        "pred_mean_V_urllc",
        "pred_mean_management_utility",
        f"upper_V_embb_q{Q_LEVEL}",
        f"upper_V_urllc_q{Q_LEVEL}",
        "true_joint_safe",
        "cert_joint_safe",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def action_distance(row_a, row_b):
    de = float(row_a["candidate_embb_slice_prb"]) - float(row_b["candidate_embb_slice_prb"])
    du = float(row_a["candidate_urllc_slice_prb"]) - float(row_b["candidate_urllc_slice_prb"])
    return abs(de) + abs(du)


def raw_score(row, policy):
    if policy == "Raw-Utility":
        return float(row["pred_mean_management_utility"])

    if policy == "Raw-LowRisk":
        return -0.5 * float(row["pred_mean_V_embb"]) - 0.5 * float(row["pred_mean_V_urllc"])

    if policy == "Raw-Balanced":
        return (
            float(row["pred_mean_management_utility"])
            - 0.5 * float(row["pred_mean_V_embb"])
            - 0.5 * float(row["pred_mean_V_urllc"])
        )

    if policy == "Raw-eMBB":
        return float(row["candidate_embb_slice_prb"])

    if policy == "Raw-URLLC":
        return float(row["candidate_urllc_slice_prb"])

    raise ValueError(f"Unknown policy: {policy}")


def shield_score(candidate_row, raw_row):
    # Certified replacement score:
    # maximize predicted utility,
    # penalize calibrated risk upper bound,
    # penalize distance from raw action.
    distance = action_distance(candidate_row, raw_row)

    upper_risk = (
        0.5 * float(candidate_row[f"upper_V_embb_q{Q_LEVEL}"])
        + 0.5 * float(candidate_row[f"upper_V_urllc_q{Q_LEVEL}"])
    )

    score = (
        float(candidate_row["pred_mean_management_utility"])
        - LAMBDA_UPPER_RISK * upper_risk
        - LAMBDA_DISTANCE * distance
    )

    return score


def pick_raw_action(group, policy):
    scores = group.apply(lambda r: raw_score(r, policy), axis=1)
    return group.loc[scores.idxmax()]


def pick_shielded_action(group, raw_row):
    # If raw action is certified safe, pass through.
    if int(raw_row["cert_joint_safe"]) == 1:
        return raw_row, "pass"

    cert_safe = group[group["cert_joint_safe"] == 1].copy()

    if len(cert_safe) > 0:
        scores = cert_safe.apply(lambda r: shield_score(r, raw_row), axis=1)
        return cert_safe.loc[scores.idxmax()], "shield_to_certified_safe"

    # Fallback: no certified safe action.
    # Choose least calibrated upper risk.
    fallback_scores = -(
        0.5 * group[f"upper_V_embb_q{Q_LEVEL}"]
        + 0.5 * group[f"upper_V_urllc_q{Q_LEVEL}"]
    )
    return group.loc[fallback_scores.idxmax()], "fallback_no_certified_safe"


def evaluate_policy(policy, raw_rows, shield_rows, shield_modes):
    raw_true_safe = raw_rows["true_joint_safe"].astype(int).to_numpy()
    shield_true_safe = shield_rows["true_joint_safe"].astype(int).to_numpy()

    raw_cert_safe = raw_rows["cert_joint_safe"].astype(int).to_numpy()
    shield_cert_safe = shield_rows["cert_joint_safe"].astype(int).to_numpy()

    raw_u = raw_rows["true_management_utility"].to_numpy()
    shield_u = shield_rows["true_management_utility"].to_numpy()

    raw_v_embb = raw_rows["true_V_embb"].to_numpy()
    shield_v_embb = shield_rows["true_V_embb"].to_numpy()

    raw_v_urllc = raw_rows["true_V_urllc"].to_numpy()
    shield_v_urllc = shield_rows["true_V_urllc"].to_numpy()

    raw_candidate_id = raw_rows["candidate_id"].to_numpy()
    shield_candidate_id = shield_rows["candidate_id"].to_numpy()

    changed = raw_candidate_id != shield_candidate_id

    # Action distance.
    dist = (
        np.abs(raw_rows["candidate_embb_slice_prb"].to_numpy() - shield_rows["candidate_embb_slice_prb"].to_numpy())
        + np.abs(raw_rows["candidate_urllc_slice_prb"].to_numpy() - shield_rows["candidate_urllc_slice_prb"].to_numpy())
    )

    utility_retention = np.mean(shield_u / (raw_u + 1e-9))

    modes = np.array(shield_modes)

    result = {
        "policy": policy,
        "num_groups": int(len(raw_rows)),

        "raw_true_safe_rate": float(raw_true_safe.mean()),
        "shielded_true_safe_rate": float(shield_true_safe.mean()),
        "raw_true_unsafe_rate": float(1.0 - raw_true_safe.mean()),
        "shielded_true_unsafe_rate": float(1.0 - shield_true_safe.mean()),

        "raw_certified_safe_rate": float(raw_cert_safe.mean()),
        "shielded_certified_safe_rate": float(shield_cert_safe.mean()),

        "raw_V_embb_mean": float(raw_v_embb.mean()),
        "shielded_V_embb_mean": float(shield_v_embb.mean()),
        "raw_V_urllc_mean": float(raw_v_urllc.mean()),
        "shielded_V_urllc_mean": float(shield_v_urllc.mean()),

        "utility_raw_mean": float(raw_u.mean()),
        "utility_shielded_mean": float(shield_u.mean()),
        "utility_retention": float(utility_retention),

        "action_changed_rate": float(changed.mean()),
        "mean_action_distance": float(dist.mean()),
        "p95_action_distance": float(np.quantile(dist, 0.95)),

        "pass_through_rate": float(np.mean(modes == "pass")),
        "shield_to_safe_rate": float(np.mean(modes == "shield_to_certified_safe")),
        "fallback_rate": float(np.mean(modes == "fallback_no_certified_safe")),
    }

    # Improvement metrics.
    result["unsafe_reduction_abs"] = result["raw_true_unsafe_rate"] - result["shielded_true_unsafe_rate"]

    if result["raw_true_unsafe_rate"] > 1e-12:
        result["unsafe_reduction_rel"] = result["unsafe_reduction_abs"] / result["raw_true_unsafe_rate"]
    else:
        result["unsafe_reduction_rel"] = 0.0

    result["utility_loss_abs"] = result["utility_raw_mean"] - result["utility_shielded_mean"]

    return result


def main():
    print(f"Reading certified candidates: {INPUT_CANDIDATES}")
    df = pd.read_csv(INPUT_CANDIDATES)

    check_required_columns(df)

    policies = [
        "Raw-Utility",
        "Raw-LowRisk",
        "Raw-Balanced",
        "Raw-eMBB",
        "Raw-URLLC",
    ]

    results = []
    selection_rows = []

    grouped = df.groupby("candidate_group_id", sort=True)
    num_groups = df["candidate_group_id"].nunique()

    print(f"Candidate rows: {len(df)}")
    print(f"Candidate groups: {num_groups}")

    for policy in policies:
        print(f"Evaluating policy: {policy}")

        raw_selected = []
        shield_selected = []
        shield_modes = []

        for gid, group in grouped:
            raw = pick_raw_action(group, policy)
            shielded, mode = pick_shielded_action(group, raw)

            raw_selected.append(raw)
            shield_selected.append(shielded)
            shield_modes.append(mode)

            selection_rows.append({
                "policy": policy,
                "candidate_group_id": int(gid),

                "raw_candidate_id": int(raw["candidate_id"]),
                "shield_candidate_id": int(shielded["candidate_id"]),
                "shield_mode": mode,

                "raw_true_joint_safe": int(raw["true_joint_safe"]),
                "shield_true_joint_safe": int(shielded["true_joint_safe"]),

                "raw_cert_joint_safe": int(raw["cert_joint_safe"]),
                "shield_cert_joint_safe": int(shielded["cert_joint_safe"]),

                "raw_true_V_embb": float(raw["true_V_embb"]),
                "shield_true_V_embb": float(shielded["true_V_embb"]),
                "raw_true_V_urllc": float(raw["true_V_urllc"]),
                "shield_true_V_urllc": float(shielded["true_V_urllc"]),

                "raw_true_management_utility": float(raw["true_management_utility"]),
                "shield_true_management_utility": float(shielded["true_management_utility"]),

                "raw_embb_prb": float(raw["candidate_embb_slice_prb"]),
                "raw_urllc_prb": float(raw["candidate_urllc_slice_prb"]),
                "shield_embb_prb": float(shielded["candidate_embb_slice_prb"]),
                "shield_urllc_prb": float(shielded["candidate_urllc_slice_prb"]),
            })

        raw_df = pd.DataFrame(raw_selected).reset_index(drop=True)
        shield_df = pd.DataFrame(shield_selected).reset_index(drop=True)

        result = evaluate_policy(policy, raw_df, shield_df, shield_modes)
        results.append(result)

    result_df = pd.DataFrame(results)
    selection_df = pd.DataFrame(selection_rows)

    summary_path = OUT_DIR / "step1_6_raw_shielding_summary.csv"
    selection_path = OUT_DIR / "step1_6_raw_shielding_selections.csv"

    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    selection_df.to_csv(selection_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_candidates": str(INPUT_CANDIDATES),
        "q_level": Q_LEVEL,
        "eps_embb": EPS_EMBB,
        "eps_urllc": EPS_URLLC,
        "lambda_distance": LAMBDA_DISTANCE,
        "lambda_upper_risk": LAMBDA_UPPER_RISK,
        "num_candidate_rows": int(len(df)),
        "num_candidate_groups": int(num_groups),
        "policies": policies,
        "outputs": {
            "raw_shielding_summary": str(summary_path),
            "raw_shielding_selections": str(selection_path),
        },
        "important_note": (
            "Step 1.6 applies raw-action shielding on the candidate groups. "
            "This is still a candidate-set experiment, not yet the final PF/PPO/SAC paired controller experiment."
        )
    }

    json_path = OUT_DIR / "step1_6_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step1_6_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 1.6 Raw-Action Shielding with Candidate Set Report\n\n")

        f.write("## Setup\n\n")
        f.write(f"- q level: {Q_LEVEL}\n")
        f.write(f"- eps_embb: {EPS_EMBB}\n")
        f.write(f"- eps_urllc: {EPS_URLLC}\n")
        f.write(f"- lambda_distance: {LAMBDA_DISTANCE}\n")
        f.write(f"- lambda_upper_risk: {LAMBDA_UPPER_RISK}\n")
        f.write(f"- candidate rows: {len(df)}\n")
        f.write(f"- candidate groups: {num_groups}\n\n")

        f.write("## Policies\n\n")
        for p in policies:
            f.write(f"- {p}\n")

        f.write("\n## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- If raw action is certified safe, CertiTwin passes it through.\n")
        f.write("- If raw action is not certified safe, CertiTwin projects it to a certified-safe candidate.\n")
        f.write("- If no certified-safe candidate exists, it falls back to the least upper-risk candidate.\n")
        f.write("- The next step will summarize whether shielding reduces unsafe actions with high utility retention.\n")

    print("Step 1.6 completed.")
    print(json.dumps(summary, indent=2))
    print("\nRaw-shielding summary:")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()