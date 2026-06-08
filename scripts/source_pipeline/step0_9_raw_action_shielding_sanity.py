import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "calibration"
SAFE_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "safe_set"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "shielding"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_9"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SAFE_LABELS = SAFE_DIR / "step0_8_test_predictions_safe_labels.csv"

Q_LEVEL = 90
GROUP_SIZE = 20
MAX_GROUPS = 20000
SEED = 20260604

EPS_EMBB = 0.3178807983617946
EPS_URLLC = 0.10566940977312222


def build_candidate_groups(df, group_size=20, max_groups=20000, seed=20260604):
    rng = np.random.default_rng(seed)
    n = len(df)
    usable = (n // group_size) * group_size
    if usable <= 0:
        raise RuntimeError("Not enough rows to build candidate groups.")

    # Shuffle rows so that each group mimics a candidate set.
    idx = np.arange(n)
    rng.shuffle(idx)
    idx = idx[:usable]

    if usable // group_size > max_groups:
        idx = idx[:max_groups * group_size]
        usable = len(idx)

    grouped = df.iloc[idx].copy().reset_index(drop=True)
    grouped["group_id"] = np.repeat(np.arange(usable // group_size), group_size)
    grouped["candidate_id"] = grouped.groupby("group_id").cumcount()

    return grouped


def true_safe(row):
    return (row["true_V_embb"] <= EPS_EMBB) and (row["true_V_urllc"] <= EPS_URLLC)


def calibrated_safe(row):
    return (
        row[f"upper_V_embb_q{Q_LEVEL}"] <= EPS_EMBB
        and row[f"upper_V_urllc_q{Q_LEVEL}"] <= EPS_URLLC
    )


def raw_safe_pred(row):
    return (
        row["pred_V_embb"] <= EPS_EMBB
        and row["pred_V_urllc"] <= EPS_URLLC
    )


def score_utility(row):
    return row["pred_management_utility"]


def score_low_risk(row):
    return -0.5 * row["pred_V_embb"] - 0.5 * row["pred_V_urllc"]


def score_balanced(row):
    return (
        row["pred_management_utility"]
        - 0.5 * row["pred_V_embb"]
        - 0.5 * row["pred_V_urllc"]
    )


def score_shield(row):
    # Sanity shield score. Final version will include action distance and risk debt.
    return (
        row["pred_management_utility"]
        - 0.5 * row[f"upper_V_embb_q{Q_LEVEL}"]
        - 0.5 * row[f"upper_V_urllc_q{Q_LEVEL}"]
    )


RAW_POLICIES = {
    "Raw-Utility": score_utility,
    "Raw-LowRisk": score_low_risk,
    "Raw-Balanced": score_balanced,
}


def pick_by_score(g, score_func):
    scores = g.apply(score_func, axis=1)
    return g.loc[scores.idxmax()]


def pick_random(g, rng):
    return g.iloc[int(rng.integers(0, len(g)))]


def shield_action(g, raw_row):
    # If raw action is calibrated safe, pass through.
    if calibrated_safe(raw_row):
        return raw_row, "pass"

    safe_candidates = g[g["calibrated_safe"] == 1]
    if len(safe_candidates) > 0:
        safe_scores = safe_candidates.apply(score_shield, axis=1)
        return safe_candidates.loc[safe_scores.idxmax()], "shield_to_certified_safe"

    # Fallback: no certified safe candidate. Choose least upper-risk candidate.
    fallback_scores = -(
        g[f"upper_V_embb_q{Q_LEVEL}"]
        + g[f"upper_V_urllc_q{Q_LEVEL}"]
    )
    return g.loc[fallback_scores.idxmax()], "fallback_no_certified_safe"


def evaluate_pair(policy_name, raw_rows, shield_rows, shield_modes):
    raw_true_safe = raw_rows["true_safe"].to_numpy()
    shield_true_safe = shield_rows["true_safe"].to_numpy()

    raw_cal_safe = raw_rows["calibrated_safe"].to_numpy()
    shield_cal_safe = shield_rows["calibrated_safe"].to_numpy()

    raw_u = raw_rows["true_management_utility"].to_numpy()
    shield_u = shield_rows["true_management_utility"].to_numpy()

    utility_retention = np.mean(shield_u / (raw_u + 1e-9))

    changed = (
        raw_rows["global_row_id"].to_numpy()
        != shield_rows["global_row_id"].to_numpy()
    )

    result = {
        "policy": policy_name,
        "num_groups": int(len(raw_rows)),
        "raw_true_safe_rate": float(raw_true_safe.mean()),
        "shielded_true_safe_rate": float(shield_true_safe.mean()),
        "raw_true_unsafe_rate": float(1 - raw_true_safe.mean()),
        "shielded_true_unsafe_rate": float(1 - shield_true_safe.mean()),
        "raw_certified_safe_rate": float(raw_cal_safe.mean()),
        "shielded_certified_safe_rate": float(shield_cal_safe.mean()),
        "utility_raw_mean": float(raw_u.mean()),
        "utility_shielded_mean": float(shield_u.mean()),
        "utility_retention": float(utility_retention),
        "action_changed_rate": float(changed.mean()),
        "pass_through_rate": float(np.mean(np.array(shield_modes) == "pass")),
        "fallback_rate": float(np.mean(np.array(shield_modes) == "fallback_no_certified_safe")),
        "shield_to_safe_rate": float(np.mean(np.array(shield_modes) == "shield_to_certified_safe")),
    }

    return result


def main():
    print(f"Reading safe labels: {TEST_SAFE_LABELS}")
    df = pd.read_csv(TEST_SAFE_LABELS)

    df = df.copy().reset_index(drop=True)
    df["global_row_id"] = np.arange(len(df))

    df["true_safe"] = df.apply(lambda r: int(true_safe(r)), axis=1)
    df["calibrated_safe"] = df.apply(lambda r: int(calibrated_safe(r)), axis=1)
    df["raw_pred_safe"] = df.apply(lambda r: int(raw_safe_pred(r)), axis=1)

    grouped = build_candidate_groups(df, GROUP_SIZE, MAX_GROUPS, SEED)

    rng = np.random.default_rng(SEED)

    results = []
    selection_rows = []

    all_policies = list(RAW_POLICIES.keys()) + ["Raw-Random"]

    for policy_name in all_policies:
        raw_selected = []
        shield_selected = []
        shield_modes = []

        for gid, g in grouped.groupby("group_id"):
            if policy_name == "Raw-Random":
                raw = pick_random(g, rng)
            else:
                raw = pick_by_score(g, RAW_POLICIES[policy_name])

            shielded, mode = shield_action(g, raw)

            raw_selected.append(raw)
            shield_selected.append(shielded)
            shield_modes.append(mode)

            selection_rows.append({
                "policy": policy_name,
                "group_id": int(gid),
                "raw_global_row_id": int(raw["global_row_id"]),
                "shield_global_row_id": int(shielded["global_row_id"]),
                "shield_mode": mode,
                "raw_true_safe": int(raw["true_safe"]),
                "shield_true_safe": int(shielded["true_safe"]),
                "raw_certified_safe": int(raw["calibrated_safe"]),
                "shield_certified_safe": int(shielded["calibrated_safe"]),
                "raw_true_utility": float(raw["true_management_utility"]),
                "shield_true_utility": float(shielded["true_management_utility"]),
                "raw_true_V_embb": float(raw["true_V_embb"]),
                "raw_true_V_urllc": float(raw["true_V_urllc"]),
                "shield_true_V_embb": float(shielded["true_V_embb"]),
                "shield_true_V_urllc": float(shielded["true_V_urllc"]),
            })

        raw_df = pd.DataFrame(raw_selected)
        shield_df = pd.DataFrame(shield_selected)

        result = evaluate_pair(policy_name, raw_df, shield_df, shield_modes)
        results.append(result)

    result_df = pd.DataFrame(results)
    selection_df = pd.DataFrame(selection_rows)

    result_path = OUT_DIR / "step0_9_raw_shielding_summary.csv"
    selection_path = OUT_DIR / "step0_9_raw_shielding_selections.csv"

    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    selection_df.to_csv(selection_path, index=False, encoding="utf-8-sig")

    summary = {
        "q_level": Q_LEVEL,
        "group_size": GROUP_SIZE,
        "max_groups": MAX_GROUPS,
        "num_groups_used": int(result_df["num_groups"].iloc[0]),
        "eps_embb": EPS_EMBB,
        "eps_urllc": EPS_URLLC,
        "outputs": {
            "raw_shielding_summary": str(result_path),
            "raw_shielding_selections": str(selection_path),
        },
        "important_note": (
            "This is a raw-action shielding sanity check over pseudo candidate groups. "
            "The final experiment will use real controller-generated actions and explicit candidate pools."
        )
    }

    summary_path = OUT_DIR / "step0_9_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step0_9_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.9 Raw-Action Shielding Sanity Report\n\n")
        f.write("## Setup\n\n")
        f.write(f"- q level: {Q_LEVEL}\n")
        f.write(f"- group size: {GROUP_SIZE}\n")
        f.write(f"- num groups used: {summary['num_groups_used']}\n")
        f.write(f"- eps_embb: {EPS_EMBB}\n")
        f.write(f"- eps_urllc: {EPS_URLLC}\n\n")

        f.write("## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- If the raw action is certified safe, the shield passes it through.\n")
        f.write("- If the raw action is unsafe, the shield selects a certified-safe candidate when available.\n")
        f.write("- If no certified-safe candidate exists, the shield falls back to the least upper-risk candidate.\n")
        f.write("- This step uses pseudo candidate groups and is not yet the final PPO/SAC shielding experiment.\n")

    print("Step 0.9 completed.")
    print(json.dumps(summary, indent=2))
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()