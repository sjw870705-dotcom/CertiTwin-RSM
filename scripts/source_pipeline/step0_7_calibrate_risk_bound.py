import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "digital_twin_sanity"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_7"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "calibration"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CALIBRATION = MODEL_DIR / "pred_validation.csv"   # Step 0.6 used validation prediction
PRED_TEST = MODEL_DIR / "pred_test.csv"

# We use validation prediction as calibration sanity set for now.
# Later, we will generate predictions on the actual calibration split.
DELTA_LIST = [0.20, 0.10, 0.05]

RISK_TARGETS = ["V_embb", "V_urllc", "V_total"]


def positive_residual(df, target):
    true_col = f"true_{target}"
    pred_col = f"pred_{target}"
    return np.maximum(df[true_col].to_numpy() - df[pred_col].to_numpy(), 0.0)


def compute_quantiles(calib_df, delta_list):
    q_table = {}
    for target in RISK_TARGETS:
        res = positive_residual(calib_df, target)
        q_table[target] = {}
        for delta in delta_list:
            q = float(np.quantile(res, 1.0 - delta))
            q_table[target][f"q_{int((1-delta)*100)}"] = q
    return q_table


def evaluate_coverage(df, q_table, delta_list, prefix):
    rows = []

    for target in RISK_TARGETS:
        true = df[f"true_{target}"].to_numpy()
        pred = df[f"pred_{target}"].to_numpy()

        for delta in delta_list:
            level = int((1 - delta) * 100)
            q = q_table[target][f"q_{level}"]
            upper = pred + q

            coverage = float(np.mean(true <= upper))
            mean_upper_gap = float(np.mean(upper - true))
            p95_upper_gap = float(np.quantile(upper - true, 0.95))

            # Safe-set proxy threshold: use true target median + 0.5 std as a provisional risk threshold.
            # This is only for sanity checking false-safe behavior.
            threshold = float(np.mean(true) + 0.5 * np.std(true))

            pred_safe = upper <= threshold
            true_safe = true <= threshold

            if pred_safe.sum() > 0:
                false_safe_rate = float(np.mean(pred_safe & (~true_safe)) / np.mean(pred_safe))
            else:
                false_safe_rate = 0.0

            safe_candidate_ratio = float(np.mean(pred_safe))

            rows.append({
                "split": prefix,
                "target": target,
                "delta": delta,
                "target_coverage": 1.0 - delta,
                "q": q,
                "empirical_coverage": coverage,
                "mean_upper_gap": mean_upper_gap,
                "p95_upper_gap": p95_upper_gap,
                "risk_threshold_sanity": threshold,
                "safe_candidate_ratio_sanity": safe_candidate_ratio,
                "false_safe_rate_sanity": false_safe_rate,
            })

    return pd.DataFrame(rows)


def add_upper_bounds(df, q_table, delta=0.10):
    out = df.copy()
    level = int((1 - delta) * 100)

    for target in RISK_TARGETS:
        q = q_table[target][f"q_{level}"]
        out[f"upper_{target}_q{level}"] = out[f"pred_{target}"] + q
        out[f"pos_residual_{target}"] = np.maximum(out[f"true_{target}"] - out[f"pred_{target}"], 0.0)

    return out


def main():
    print(f"Reading calibration predictions: {PRED_CALIBRATION}")
    print(f"Reading test predictions: {PRED_TEST}")

    calib_df = pd.read_csv(PRED_CALIBRATION)
    test_df = pd.read_csv(PRED_TEST)

    q_table = compute_quantiles(calib_df, DELTA_LIST)

    calib_cov = evaluate_coverage(calib_df, q_table, DELTA_LIST, prefix="calibration_sanity")
    test_cov = evaluate_coverage(test_df, q_table, DELTA_LIST, prefix="test")

    coverage_df = pd.concat([calib_cov, test_cov], ignore_index=True)

    q_path = OUT_DIR / "step0_7_calibration_quantiles.json"
    coverage_path = OUT_DIR / "step0_7_coverage_table.csv"

    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(q_table, f, indent=2)

    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    calib_upper = add_upper_bounds(calib_df, q_table, delta=0.10)
    test_upper = add_upper_bounds(test_df, q_table, delta=0.10)

    calib_upper_path = OUT_DIR / "step0_7_calibration_predictions_with_upper.csv"
    test_upper_path = OUT_DIR / "step0_7_test_predictions_with_upper.csv"

    calib_upper.to_csv(calib_upper_path, index=False, encoding="utf-8-sig")
    test_upper.to_csv(test_upper_path, index=False, encoding="utf-8-sig")

    summary = {
        "calibration_prediction_file": str(PRED_CALIBRATION),
        "test_prediction_file": str(PRED_TEST),
        "delta_list": DELTA_LIST,
        "risk_targets": RISK_TARGETS,
        "quantiles": q_table,
        "outputs": {
            "quantiles": str(q_path),
            "coverage_table": str(coverage_path),
            "calibration_predictions_with_upper": str(calib_upper_path),
            "test_predictions_with_upper": str(test_upper_path),
        },
        "important_note": (
            "Step 0.7 is a residual-based calibration sanity check. "
            "The final paper model should use ensemble/dropout uncertainty sigma and calibration residuals."
        )
    }

    summary_path = OUT_DIR / "step0_7_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step0_7_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.7 Calibrated Risk Bound Sanity Report\n\n")
        f.write("## Calibration quantiles\n\n")
        for target, qs in q_table.items():
            f.write(f"### {target}\n")
            for k, v in qs.items():
                f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This step uses positive residual quantiles as a sanity version of risk calibration.\n")
        f.write("- It verifies whether prediction residuals can form useful risk upper bounds.\n")
        f.write("- The final CertiTwin-RSM version should use ensemble uncertainty sigma.\n")
        f.write("- Coverage and false-safe behavior should be checked before moving to shield construction.\n")

    print("Step 0.7 completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()