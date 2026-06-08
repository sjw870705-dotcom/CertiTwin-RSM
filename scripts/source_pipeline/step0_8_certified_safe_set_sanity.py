import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "calibration"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "safe_set"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_8"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TEST_UPPER = CALIB_DIR / "step0_7_test_predictions_with_upper.csv"

# Use q90 upper bounds from Step 0.7.
Q_LEVEL = 90

# Sanity safe thresholds.
# These are not final SLA thresholds; they are sanity thresholds based on test statistics.
# Later, shield construction will use explicit epsilon_s from the SLA design.
TARGETS = ["V_embb", "V_urllc", "V_total"]


def safe_threshold(true_values, mode="mean_plus_half_std"):
    vals = pd.to_numeric(true_values, errors="coerce").dropna().to_numpy()
    if mode == "mean_plus_half_std":
        return float(vals.mean() + 0.5 * vals.std())
    if mode == "p80":
        return float(np.quantile(vals, 0.80))
    if mode == "p90":
        return float(np.quantile(vals, 0.90))
    raise ValueError(f"Unknown threshold mode: {mode}")


def eval_safe_rule(df, target, threshold, use_upper):
    true = df[f"true_{target}"].to_numpy()
    pred = df[f"pred_{target}"].to_numpy()

    if use_upper:
        score = df[f"upper_{target}_q{Q_LEVEL}"].to_numpy()
        rule_name = f"calibrated_upper_q{Q_LEVEL}"
    else:
        score = pred
        rule_name = "raw_prediction"

    pred_safe = score <= threshold
    true_safe = true <= threshold

    safe_ratio = float(pred_safe.mean())

    if pred_safe.sum() > 0:
        false_safe_rate = float((pred_safe & (~true_safe)).sum() / pred_safe.sum())
        true_safe_precision = float((pred_safe & true_safe).sum() / pred_safe.sum())
    else:
        false_safe_rate = 0.0
        true_safe_precision = 0.0

    unsafe = ~true_safe
    if unsafe.sum() > 0:
        unsafe_interception_rate = float(((~pred_safe) & unsafe).sum() / unsafe.sum())
    else:
        unsafe_interception_rate = 0.0

    # How many truly safe actions are lost due to conservatism?
    if true_safe.sum() > 0:
        safe_rejection_rate = float(((~pred_safe) & true_safe).sum() / true_safe.sum())
    else:
        safe_rejection_rate = 0.0

    return {
        "target": target,
        "rule": rule_name,
        "threshold": float(threshold),
        "safe_ratio": safe_ratio,
        "false_safe_rate": false_safe_rate,
        "true_safe_precision": true_safe_precision,
        "unsafe_interception_rate": unsafe_interception_rate,
        "safe_rejection_rate": safe_rejection_rate,
    }


def eval_joint_safe_rule(df, thresholds, use_upper):
    true_embb = df["true_V_embb"].to_numpy()
    true_urllc = df["true_V_urllc"].to_numpy()

    if use_upper:
        score_embb = df[f"upper_V_embb_q{Q_LEVEL}"].to_numpy()
        score_urllc = df[f"upper_V_urllc_q{Q_LEVEL}"].to_numpy()
        rule_name = f"joint_calibrated_upper_q{Q_LEVEL}"
    else:
        score_embb = df["pred_V_embb"].to_numpy()
        score_urllc = df["pred_V_urllc"].to_numpy()
        rule_name = "joint_raw_prediction"

    pred_safe = (score_embb <= thresholds["V_embb"]) & (score_urllc <= thresholds["V_urllc"])
    true_safe = (true_embb <= thresholds["V_embb"]) & (true_urllc <= thresholds["V_urllc"])

    safe_ratio = float(pred_safe.mean())

    if pred_safe.sum() > 0:
        false_safe_rate = float((pred_safe & (~true_safe)).sum() / pred_safe.sum())
        true_safe_precision = float((pred_safe & true_safe).sum() / pred_safe.sum())
    else:
        false_safe_rate = 0.0
        true_safe_precision = 0.0

    unsafe = ~true_safe
    if unsafe.sum() > 0:
        unsafe_interception_rate = float(((~pred_safe) & unsafe).sum() / unsafe.sum())
    else:
        unsafe_interception_rate = 0.0

    if true_safe.sum() > 0:
        safe_rejection_rate = float(((~pred_safe) & true_safe).sum() / true_safe.sum())
    else:
        safe_rejection_rate = 0.0

    return {
        "target": "joint_V_embb_V_urllc",
        "rule": rule_name,
        "threshold_V_embb": float(thresholds["V_embb"]),
        "threshold_V_urllc": float(thresholds["V_urllc"]),
        "safe_ratio": safe_ratio,
        "false_safe_rate": false_safe_rate,
        "true_safe_precision": true_safe_precision,
        "unsafe_interception_rate": unsafe_interception_rate,
        "safe_rejection_rate": safe_rejection_rate,
    }


def main():
    print(f"Reading test upper-bound predictions: {TEST_UPPER}")
    df = pd.read_csv(TEST_UPPER)

    thresholds = {}
    for target in TARGETS:
        thresholds[target] = safe_threshold(df[f"true_{target}"], mode="mean_plus_half_std")

    rows = []

    for target in TARGETS:
        rows.append(eval_safe_rule(df, target, thresholds[target], use_upper=False))
        rows.append(eval_safe_rule(df, target, thresholds[target], use_upper=True))

    rows.append(eval_joint_safe_rule(df, thresholds, use_upper=False))
    rows.append(eval_joint_safe_rule(df, thresholds, use_upper=True))

    result_df = pd.DataFrame(rows)
    result_path = OUT_DIR / "step0_8_safe_set_sanity_table.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    # Add labels to a capped prediction file for inspection.
    inspect = df.copy()
    inspect["threshold_V_embb"] = thresholds["V_embb"]
    inspect["threshold_V_urllc"] = thresholds["V_urllc"]

    inspect["true_joint_safe"] = (
        (inspect["true_V_embb"] <= thresholds["V_embb"])
        & (inspect["true_V_urllc"] <= thresholds["V_urllc"])
    ).astype(int)

    inspect["raw_pred_joint_safe"] = (
        (inspect["pred_V_embb"] <= thresholds["V_embb"])
        & (inspect["pred_V_urllc"] <= thresholds["V_urllc"])
    ).astype(int)

    inspect["calibrated_joint_safe"] = (
        (inspect[f"upper_V_embb_q{Q_LEVEL}"] <= thresholds["V_embb"])
        & (inspect[f"upper_V_urllc_q{Q_LEVEL}"] <= thresholds["V_urllc"])
    ).astype(int)

    inspect_path = OUT_DIR / "step0_8_test_predictions_safe_labels.csv"
    inspect.to_csv(inspect_path, index=False, encoding="utf-8-sig")

    summary = {
        "q_level": Q_LEVEL,
        "threshold_mode": "mean_plus_half_std",
        "thresholds": thresholds,
        "num_rows": int(len(df)),
        "outputs": {
            "safe_set_sanity_table": str(result_path),
            "test_predictions_safe_labels": str(inspect_path),
        },
        "interpretation": {
            "safe_ratio": "Fraction of candidates certified safe by the rule.",
            "false_safe_rate": "Among predicted-safe candidates, fraction that is truly unsafe.",
            "unsafe_interception_rate": "Among truly unsafe candidates, fraction rejected by the rule.",
            "safe_rejection_rate": "Among truly safe candidates, fraction rejected by conservatism."
        }
    }

    summary_path = OUT_DIR / "step0_8_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step0_8_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.8 Certified Safe Set Sanity Report\n\n")
        f.write("## Thresholds\n\n")
        for k, v in thresholds.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This is a sanity check for certified safe set construction.\n")
        f.write("- It compares raw prediction safe rules against calibrated-upper-bound safe rules.\n")
        f.write("- The final CertiTwin-RSM shield will use explicit SLA thresholds and candidate actions.\n")

    print("Step 0.8 completed.")
    print(json.dumps(summary, indent=2))
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()