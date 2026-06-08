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

RESULT_DIR = PROJECT_ROOT / "results" / "step2_3"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_FILE = EVAL_DIR / "controller_candidate_groups.csv"
RAW_ACTIONS_FILE = EVAL_DIR / "raw_controller_actions_classical.csv"

OUTPUT_PAIRWISE = EVAL_DIR / "step2_3_classical_raw_vs_shielded_pairwise.csv"
OUTPUT_SUMMARY = EVAL_DIR / "step2_3_classical_shielding_summary.csv"
OUTPUT_REPORT_JSON = EVAL_DIR / "step2_3_classical_shielding_report.json"
OUTPUT_REPORT_MD = RESULT_DIR / "step2_3_report.md"

Q_LEVEL = 90

LAMBDA_DISTANCE = 0.02
LAMBDA_UPPER_RISK = 0.50

REQUIRED_CANDIDATE_COLUMNS = [
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
]

REQUIRED_RAW_COLUMNS = [
    "controller",
    "candidate_group_id",
    "raw_candidate_id",
    "raw_embb_slice_prb",
    "raw_urllc_slice_prb",
    "raw_true_V_embb",
    "raw_true_V_urllc",
    "raw_true_V_total",
    "raw_true_management_utility",
    "raw_true_joint_safe",
    "raw_cert_joint_safe",
]


def check_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def action_distance(candidate_row, raw_row):
    de = float(candidate_row["candidate_embb_slice_prb"]) - float(raw_row["raw_embb_slice_prb"])
    du = float(candidate_row["candidate_urllc_slice_prb"]) - float(raw_row["raw_urllc_slice_prb"])
    return abs(de) + abs(du)


def shield_score(candidate_row, raw_row):
    distance = action_distance(candidate_row, raw_row)

    upper_risk = (
        0.5 * float(candidate_row[f"upper_V_embb_q{Q_LEVEL}"])
        + 0.5 * float(candidate_row[f"upper_V_urllc_q{Q_LEVEL}"])
    )

    return (
        float(candidate_row["pred_mean_management_utility"])
        - LAMBDA_UPPER_RISK * upper_risk
        - LAMBDA_DISTANCE * distance
    )


def pick_shielded_action(group, raw_row):
    # Pass-through if raw action is certified safe.
    if int(raw_row["raw_cert_joint_safe"]) == 1:
        raw_candidate_id = int(raw_row["raw_candidate_id"])
        raw_match = group[group["candidate_id"] == raw_candidate_id]
        if len(raw_match) != 1:
            raise RuntimeError(
                f"Cannot find raw candidate id {raw_candidate_id} "
                f"in group {raw_row['candidate_group_id']}"
            )
        return raw_match.iloc[0], "pass"

    cert_safe = group[group["cert_joint_safe"] == 1].copy()

    if len(cert_safe) > 0:
        scores = cert_safe.apply(lambda r: shield_score(r, raw_row), axis=1)
        return cert_safe.loc[scores.idxmax()], "shield_to_certified_safe"

    fallback_scores = -(
        0.5 * group[f"upper_V_embb_q{Q_LEVEL}"]
        + 0.5 * group[f"upper_V_urllc_q{Q_LEVEL}"]
    )
    return group.loc[fallback_scores.idxmax()], "fallback_no_certified_safe"


def evaluate_controller(pair_df):
    rows = []

    for controller, g in pair_df.groupby("controller"):
        raw_safe = g["raw_true_joint_safe"].astype(int).to_numpy()
        shield_safe = g["shield_true_joint_safe"].astype(int).to_numpy()

        raw_cert_safe = g["raw_cert_joint_safe"].astype(int).to_numpy()
        shield_cert_safe = g["shield_cert_joint_safe"].astype(int).to_numpy()

        raw_u = g["raw_true_management_utility"].to_numpy()
        shield_u = g["shield_true_management_utility"].to_numpy()

        changed = g["raw_candidate_id"].to_numpy() != g["shield_candidate_id"].to_numpy()

        dist = (
            np.abs(g["raw_embb_slice_prb"].to_numpy() - g["shield_embb_slice_prb"].to_numpy())
            + np.abs(g["raw_urllc_slice_prb"].to_numpy() - g["shield_urllc_slice_prb"].to_numpy())
        )

        modes = g["shield_mode"].astype(str).to_numpy()

        rec = {
            "controller": controller,
            "shielded_controller": f"CertiTwin-{controller}",
            "num_groups": int(len(g)),

            "raw_true_safe_rate": float(raw_safe.mean()),
            "shielded_true_safe_rate": float(shield_safe.mean()),
            "raw_true_unsafe_rate": float(1.0 - raw_safe.mean()),
            "shielded_true_unsafe_rate": float(1.0 - shield_safe.mean()),

            "raw_certified_safe_rate": float(raw_cert_safe.mean()),
            "shielded_certified_safe_rate": float(shield_cert_safe.mean()),

            "raw_V_embb_mean": float(g["raw_true_V_embb"].mean()),
            "shielded_V_embb_mean": float(g["shield_true_V_embb"].mean()),
            "raw_V_urllc_mean": float(g["raw_true_V_urllc"].mean()),
            "shielded_V_urllc_mean": float(g["shield_true_V_urllc"].mean()),
            "raw_V_total_mean": float(g["raw_true_V_total"].mean()),
            "shielded_V_total_mean": float(g["shield_true_V_total"].mean()),

            "utility_raw_mean": float(raw_u.mean()),
            "utility_shielded_mean": float(shield_u.mean()),
            "utility_retention": float(np.mean(shield_u / (raw_u + 1e-9))),

            "action_changed_rate": float(changed.mean()),
            "mean_action_distance": float(dist.mean()),
            "p95_action_distance": float(np.quantile(dist, 0.95)),

            "pass_through_rate": float(np.mean(modes == "pass")),
            "shield_to_safe_rate": float(np.mean(modes == "shield_to_certified_safe")),
            "fallback_rate": float(np.mean(modes == "fallback_no_certified_safe")),
        }

        rec["unsafe_reduction_abs"] = rec["raw_true_unsafe_rate"] - rec["shielded_true_unsafe_rate"]
        if rec["raw_true_unsafe_rate"] > 1e-12:
            rec["unsafe_reduction_rel"] = rec["unsafe_reduction_abs"] / rec["raw_true_unsafe_rate"]
        else:
            rec["unsafe_reduction_rel"] = 0.0

        rec["utility_loss_abs"] = rec["utility_raw_mean"] - rec["utility_shielded_mean"]

        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    print(f"Reading candidates: {CANDIDATES_FILE}")
    candidates = pd.read_csv(CANDIDATES_FILE)

    print(f"Reading raw actions: {RAW_ACTIONS_FILE}")
    raw_actions = pd.read_csv(RAW_ACTIONS_FILE)

    check_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "candidates")
    check_columns(raw_actions, REQUIRED_RAW_COLUMNS, "raw_actions")

    print(f"Candidate rows: {len(candidates)}")
    print(f"Candidate groups: {candidates['candidate_group_id'].nunique()}")
    print(f"Raw action rows: {len(raw_actions)}")

    # Pre-group candidates for faster lookup.
    cand_groups = {gid: g.copy() for gid, g in candidates.groupby("candidate_group_id", sort=False)}

    pair_rows = []

    for idx, raw in raw_actions.iterrows():
        gid = int(raw["candidate_group_id"])
        group = cand_groups.get(gid)

        if group is None:
            raise RuntimeError(f"Missing candidate group: {gid}")

        shielded, mode = pick_shielded_action(group, raw)

        pair_rows.append({
            "controller": raw["controller"],
            "candidate_group_id": gid,

            "raw_candidate_id": int(raw["raw_candidate_id"]),
            "shield_candidate_id": int(shielded["candidate_id"]),
            "shield_mode": mode,

            "raw_embb_slice_prb": float(raw["raw_embb_slice_prb"]),
            "raw_urllc_slice_prb": float(raw["raw_urllc_slice_prb"]),
            "shield_embb_slice_prb": float(shielded["candidate_embb_slice_prb"]),
            "shield_urllc_slice_prb": float(shielded["candidate_urllc_slice_prb"]),

            "raw_true_V_embb": float(raw["raw_true_V_embb"]),
            "raw_true_V_urllc": float(raw["raw_true_V_urllc"]),
            "raw_true_V_total": float(raw["raw_true_V_total"]),
            "raw_true_management_utility": float(raw["raw_true_management_utility"]),

            "shield_true_V_embb": float(shielded["true_V_embb"]),
            "shield_true_V_urllc": float(shielded["true_V_urllc"]),
            "shield_true_V_total": float(shielded["true_V_total"]),
            "shield_true_management_utility": float(shielded["true_management_utility"]),

            "raw_true_joint_safe": int(raw["raw_true_joint_safe"]),
            "shield_true_joint_safe": int(shielded["true_joint_safe"]),

            "raw_cert_joint_safe": int(raw["raw_cert_joint_safe"]),
            "shield_cert_joint_safe": int(shielded["cert_joint_safe"]),

            "raw_upper_V_embb_q90": float(raw["raw_upper_V_embb_q90"]),
            "raw_upper_V_urllc_q90": float(raw["raw_upper_V_urllc_q90"]),
            "shield_upper_V_embb_q90": float(shielded["upper_V_embb_q90"]),
            "shield_upper_V_urllc_q90": float(shielded["upper_V_urllc_q90"]),
        })

    pair_df = pd.DataFrame(pair_rows)
    summary_df = evaluate_controller(pair_df)

    pair_df.to_csv(OUTPUT_PAIRWISE, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    report = {
        "candidate_file": str(CANDIDATES_FILE),
        "raw_action_file": str(RAW_ACTIONS_FILE),
        "num_candidate_rows": int(len(candidates)),
        "num_candidate_groups": int(candidates["candidate_group_id"].nunique()),
        "num_raw_action_rows": int(len(raw_actions)),
        "controllers": sorted(raw_actions["controller"].unique().tolist()),
        "q_level": Q_LEVEL,
        "lambda_distance": LAMBDA_DISTANCE,
        "lambda_upper_risk": LAMBDA_UPPER_RISK,
        "outputs": {
            "pairwise": str(OUTPUT_PAIRWISE),
            "summary": str(OUTPUT_SUMMARY),
        },
        "important_note": (
            "Step 2.3 applies the same CertiTwin shield to PF, Greedy-SLA, and DT-Top1 raw actions. "
            "This is the first formal raw-vs-shielded paired controller comparison."
        ),
    }

    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.3 Classical Controller Shielding Report\n\n")

        f.write("## Setup\n\n")
        f.write(f"- candidate rows: {len(candidates)}\n")
        f.write(f"- candidate groups: {candidates['candidate_group_id'].nunique()}\n")
        f.write(f"- raw action rows: {len(raw_actions)}\n")
        f.write(f"- q level: {Q_LEVEL}\n")
        f.write(f"- lambda distance: {LAMBDA_DISTANCE}\n")
        f.write(f"- lambda upper risk: {LAMBDA_UPPER_RISK}\n\n")

        f.write("## Controllers\n\n")
        for c in sorted(raw_actions["controller"].unique().tolist()):
            f.write(f"- {c} vs CertiTwin-{c}\n")

        f.write("\n## Summary\n\n")
        for _, r in summary_df.iterrows():
            f.write(f"### {r['controller']} vs {r['shielded_controller']}\n")
            f.write(f"- raw unsafe rate: {r['raw_true_unsafe_rate']}\n")
            f.write(f"- shielded unsafe rate: {r['shielded_true_unsafe_rate']}\n")
            f.write(f"- unsafe reduction relative: {r['unsafe_reduction_rel']}\n")
            f.write(f"- utility retention: {r['utility_retention']}\n")
            f.write(f"- action changed rate: {r['action_changed_rate']}\n")
            f.write(f"- pass-through rate: {r['pass_through_rate']}\n")
            f.write(f"- fallback rate: {r['fallback_rate']}\n\n")

        f.write("## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This is the first formal paired raw-controller comparison.\n")
        f.write("- The raw controller and shielded controller share the same candidate group and raw action.\n")
        f.write("- CertiTwin only modifies the raw action when it is not certified safe.\n")

    print("Step 2.3 completed.")
    print(json.dumps(report, indent=2))
    print("\nClassical shielding summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()