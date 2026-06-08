import json
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = PROJECT_ROOT / "results" / "step1_7"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
STEP1_1B_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
    / "step1_1b_report.json"
)

STEP1_2_METRICS = (
    PROJECT_ROOT
    / "models"
    / "digital_twin_candidate_baseline"
    / "step1_2_metrics.json"
)

STEP1_3_METRICS = (
    PROJECT_ROOT
    / "models"
    / "digital_twin_candidate_ensemble"
    / "step1_3_ensemble_metrics.json"
)

STEP1_4_COVERAGE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_calibration"
    / "step1_4_coverage_table.csv"
)

STEP1_5_SAFE_SET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
    / "step1_5_safe_set_summary.csv"
)

STEP1_5_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
    / "step1_5_summary.json"
)

STEP1_6_SHIELDING = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_shielding"
    / "step1_6_raw_shielding_summary.csv"
)

STEP1_6_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_shielding"
    / "step1_6_summary.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def build_table_a_prediction():
    baseline = load_json(STEP1_2_METRICS)
    ensemble = load_json(STEP1_3_METRICS)

    rows = []

    for model_name, metrics_obj in [
        ("Single-MLP", baseline),
        ("Ensemble-MLP", ensemble),
    ]:
        for split_name in ["validation_metrics", "test_metrics"]:
            split = split_name.replace("_metrics", "")

            for target, m in metrics_obj[split_name].items():
                row = {
                    "model": model_name,
                    "split": split,
                    "target": target,
                    "mae": m.get("mae"),
                    "rmse": m.get("rmse"),
                    "r2": m.get("r2"),
                    "bias": m.get("bias"),
                    "true_mean": m.get("true_mean"),
                    "pred_mean": m.get("pred_mean"),
                    "true_std": m.get("true_std"),
                    "pred_std": m.get("pred_std"),
                    "ensemble_sigma_mean": m.get("ensemble_sigma_mean", np.nan),
                    "ensemble_sigma_p95": m.get("ensemble_sigma_p95", np.nan),
                    "uncertainty_error_corr": m.get("uncertainty_error_corr", np.nan),
                }
                rows.append(row)

    table = pd.DataFrame(rows)
    return table


def build_table_b_coverage():
    coverage = pd.read_csv(STEP1_4_COVERAGE)

    # Keep q90 mainly, but preserve q80/q95 for appendix.
    cols = [
        "split",
        "target",
        "delta",
        "target_coverage",
        "q_level",
        "q",
        "empirical_coverage",
        "mean_upper_gap",
        "p05_upper_gap",
        "p95_upper_gap",
        "safe_ratio_sanity",
        "false_safe_rate_sanity",
        "unsafe_interception_rate_sanity",
    ]

    cols = [c for c in cols if c in coverage.columns]
    return coverage[cols].copy()


def build_table_c_safe_set():
    safe = pd.read_csv(STEP1_5_SAFE_SET)

    rename = {
        "rule": "safe_rule",
        "safe_ratio": "safe_ratio",
        "false_safe_rate": "false_safe_rate",
        "safe_precision": "safe_precision",
        "unsafe_interception_rate": "unsafe_interception_rate",
        "safe_rejection_rate": "safe_rejection_rate",
    }

    safe = safe.rename(columns=rename)

    cols = [
        "safe_rule",
        "safe_ratio",
        "false_safe_rate",
        "safe_precision",
        "unsafe_interception_rate",
        "safe_rejection_rate",
    ]

    cols = [c for c in cols if c in safe.columns]
    return safe[cols].copy()


def build_table_d_shielding():
    shielding = pd.read_csv(STEP1_6_SHIELDING)

    cols = [
        "policy",
        "num_groups",
        "raw_true_safe_rate",
        "shielded_true_safe_rate",
        "raw_true_unsafe_rate",
        "shielded_true_unsafe_rate",
        "unsafe_reduction_abs",
        "unsafe_reduction_rel",
        "utility_raw_mean",
        "utility_shielded_mean",
        "utility_retention",
        "action_changed_rate",
        "mean_action_distance",
        "p95_action_distance",
        "pass_through_rate",
        "shield_to_safe_rate",
        "fallback_rate",
    ]

    cols = [c for c in cols if c in shielding.columns]
    return shielding[cols].copy()


def evaluate_pass_fail(table_a, table_b, table_c, table_d):
    checks = {}

    # Digital twin prediction checks: focus on test split risk targets.
    test_ensemble = table_a[
        (table_a["model"] == "Ensemble-MLP")
        & (table_a["split"] == "test")
    ]

    risk_targets = ["V_embb", "V_urllc", "V_total"]

    checks["digital_twin_risk_mae_ok"] = bool(
        all(
            test_ensemble[test_ensemble["target"] == t]["mae"].iloc[0] < 0.02
            for t in risk_targets
            if len(test_ensemble[test_ensemble["target"] == t]) > 0
        )
    )

    checks["ensemble_uncertainty_corr_ok"] = bool(
        all(
            test_ensemble[test_ensemble["target"] == t]["uncertainty_error_corr"].iloc[0] > 0.2
            for t in risk_targets
            if len(test_ensemble[test_ensemble["target"] == t]) > 0
        )
    )

    # Coverage q90 checks.
    q90_test = table_b[
        (table_b["split"] == "test")
        & (table_b["q_level"] == 90)
        & (table_b["target"].isin(risk_targets))
    ]

    checks["q90_test_coverage_ok"] = bool(
        (q90_test["empirical_coverage"] >= 0.88).all()
    )

    # Certified safe set checks.
    if "safe_rule" in table_c.columns:
        raw_row = table_c[table_c["safe_rule"] == "raw_pred_joint_safe"]
        cert_row = table_c[table_c["safe_rule"] == "cert_joint_safe"]

        if len(raw_row) > 0 and len(cert_row) > 0:
            raw_false = float(raw_row["false_safe_rate"].iloc[0])
            cert_false = float(cert_row["false_safe_rate"].iloc[0])
            checks["certified_set_false_safe_reduction_ok"] = bool(cert_false < raw_false)

            cert_safe_ratio = float(cert_row["safe_ratio"].iloc[0])
            checks["certified_set_safe_ratio_ok"] = bool(cert_safe_ratio > 0.2)
        else:
            checks["certified_set_false_safe_reduction_ok"] = False
            checks["certified_set_safe_ratio_ok"] = False

    # Shielding checks.
    checks["shielding_fallback_ok"] = bool((table_d["fallback_rate"] <= 0.05).all())
    checks["shielding_utility_retention_ok"] = bool((table_d["utility_retention"] >= 0.85).all())

    # For policies with raw unsafe > 0, shielded unsafe should be lower.
    subset = table_d[table_d["raw_true_unsafe_rate"] > 1e-9]
    if len(subset) > 0:
        checks["shielding_unsafe_reduction_ok"] = bool(
            (subset["shielded_true_unsafe_rate"] <= subset["raw_true_unsafe_rate"]).all()
        )
    else:
        checks["shielding_unsafe_reduction_ok"] = True

    checks["overall_step1_ready_for_step2"] = bool(all(checks.values()))

    return checks


def format_percent(x):
    try:
        return f"{100 * float(x):.4f}%"
    except Exception:
        return str(x)


def main():
    for p in [
        STEP1_1B_REPORT,
        STEP1_2_METRICS,
        STEP1_3_METRICS,
        STEP1_4_COVERAGE,
        STEP1_5_SAFE_SET,
        STEP1_5_SUMMARY,
        STEP1_6_SHIELDING,
        STEP1_6_SUMMARY,
    ]:
        assert_exists(p)

    step1_1b = load_json(STEP1_1B_REPORT)
    step1_5 = load_json(STEP1_5_SUMMARY)
    step1_6 = load_json(STEP1_6_SUMMARY)

    table_a = build_table_a_prediction()
    table_b = build_table_b_coverage()
    table_c = build_table_c_safe_set()
    table_d = build_table_d_shielding()

    path_a = RESULT_DIR / "table_step1_A_digital_twin_prediction.csv"
    path_b = RESULT_DIR / "table_step1_B_calibrated_coverage.csv"
    path_c = RESULT_DIR / "table_step1_C_certified_safe_set.csv"
    path_d = RESULT_DIR / "table_step1_D_candidate_shielding.csv"

    table_a.to_csv(path_a, index=False, encoding="utf-8-sig")
    table_b.to_csv(path_b, index=False, encoding="utf-8-sig")
    table_c.to_csv(path_c, index=False, encoding="utf-8-sig")
    table_d.to_csv(path_d, index=False, encoding="utf-8-sig")

    checks = evaluate_pass_fail(table_a, table_b, table_c, table_d)

    # Extract headline numbers.
    headline = {}

    test_ensemble = table_a[
        (table_a["model"] == "Ensemble-MLP")
        & (table_a["split"] == "test")
    ]

    for target in ["V_embb", "V_urllc", "V_total", "management_utility"]:
        row = test_ensemble[test_ensemble["target"] == target]
        if len(row) > 0:
            headline[f"test_mae_{target}"] = float(row["mae"].iloc[0])
            headline[f"test_r2_{target}"] = float(row["r2"].iloc[0])

    q90_test = table_b[
        (table_b["split"] == "test")
        & (table_b["q_level"] == 90)
    ]

    for target in ["V_embb", "V_urllc", "V_total"]:
        row = q90_test[q90_test["target"] == target]
        if len(row) > 0:
            headline[f"q90_test_coverage_{target}"] = float(row["empirical_coverage"].iloc[0])

    raw_joint = table_c[table_c["safe_rule"] == "raw_pred_joint_safe"]
    cert_joint = table_c[table_c["safe_rule"] == "cert_joint_safe"]

    if len(raw_joint) > 0 and len(cert_joint) > 0:
        headline["raw_joint_false_safe_rate"] = float(raw_joint["false_safe_rate"].iloc[0])
        headline["cert_joint_false_safe_rate"] = float(cert_joint["false_safe_rate"].iloc[0])
        headline["raw_joint_safe_ratio"] = float(raw_joint["safe_ratio"].iloc[0])
        headline["cert_joint_safe_ratio"] = float(cert_joint["safe_ratio"].iloc[0])

    # Best/worst shielding summary.
    if len(table_d) > 0:
        headline["max_raw_unsafe_rate"] = float(table_d["raw_true_unsafe_rate"].max())
        headline["max_shielded_unsafe_rate"] = float(table_d["shielded_true_unsafe_rate"].max())
        headline["min_utility_retention"] = float(table_d["utility_retention"].min())
        headline["max_fallback_rate"] = float(table_d["fallback_rate"].max())

    overall = {
        "step1_1b_dataset": {
            "base_rows_used": step1_1b.get("base_rows_used"),
            "num_expanded_rows": step1_1b.get("num_expanded_rows"),
            "observed_action_pair_count": step1_1b.get("observed_action_pair_count"),
            "candidate_actions_per_state": step1_1b.get("candidate_actions_per_state"),
            "feature_count": step1_1b.get("feature_count"),
            "target_count": step1_1b.get("target_count"),
        },
        "step1_5_safe_set": {
            "eps_embb": step1_5.get("eps_embb"),
            "eps_urllc": step1_5.get("eps_urllc"),
            "num_candidates": step1_5.get("num_candidates"),
            "group_metrics": step1_5.get("group_metrics"),
        },
        "step1_6_shielding": {
            "num_candidate_rows": step1_6.get("num_candidate_rows"),
            "num_candidate_groups": step1_6.get("num_candidate_groups"),
            "policies": step1_6.get("policies"),
        },
        "headline_metrics": headline,
        "pass_fail_checks": checks,
        "outputs": {
            "table_A_digital_twin_prediction": str(path_a),
            "table_B_calibrated_coverage": str(path_b),
            "table_C_certified_safe_set": str(path_c),
            "table_D_candidate_shielding": str(path_d),
        },
        "conclusion": (
            "Step 1 candidate-aware diagnostic pipeline is ready for Step 2 raw-controller paired experiments."
            if checks.get("overall_step1_ready_for_step2")
            else "Step 1 has warnings; inspect pass_fail_checks before Step 2."
        )
    }

    summary_path = RESULT_DIR / "step1_7_overall_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    md_path = RESULT_DIR / "step1_7_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Step 1.7 Formal Diagnostics Summary for Step 1.1–1.6\n\n")

        f.write("## Dataset summary\n\n")
        f.write(f"- Expanded candidate samples: {step1_1b.get('num_expanded_rows')}\n")
        f.write(f"- Observed action pairs: {step1_1b.get('observed_action_pair_count')}\n")
        f.write(f"- Candidate actions per state: {step1_1b.get('candidate_actions_per_state')}\n")
        f.write(f"- Feature count: {step1_1b.get('feature_count')}\n")
        f.write(f"- Target count: {step1_1b.get('target_count')}\n\n")

        f.write("## Table A. Candidate-aware digital twin prediction\n\n")
        f.write("See `table_step1_A_digital_twin_prediction.csv`.\n\n")
        for target in ["V_embb", "V_urllc", "V_total", "management_utility"]:
            mae_key = f"test_mae_{target}"
            r2_key = f"test_r2_{target}"
            if mae_key in headline:
                f.write(f"- {target}: test MAE = {headline[mae_key]:.6f}, test R2 = {headline[r2_key]:.6f}\n")

        f.write("\n## Table B. Calibrated risk-bound coverage\n\n")
        f.write("See `table_step1_B_calibrated_coverage.csv`.\n\n")
        for target in ["V_embb", "V_urllc", "V_total"]:
            key = f"q90_test_coverage_{target}"
            if key in headline:
                f.write(f"- {target}: q90 test coverage = {headline[key]:.6f}\n")

        f.write("\n## Table C. Certified safe set\n\n")
        f.write("See `table_step1_C_certified_safe_set.csv`.\n\n")
        if "raw_joint_false_safe_rate" in headline:
            f.write(f"- Raw joint false-safe rate: {format_percent(headline['raw_joint_false_safe_rate'])}\n")
            f.write(f"- Certified joint false-safe rate: {format_percent(headline['cert_joint_false_safe_rate'])}\n")
            f.write(f"- Raw joint safe ratio: {format_percent(headline['raw_joint_safe_ratio'])}\n")
            f.write(f"- Certified joint safe ratio: {format_percent(headline['cert_joint_safe_ratio'])}\n")

        f.write("\n## Table D. Candidate-level shielding\n\n")
        f.write("See `table_step1_D_candidate_shielding.csv`.\n\n")
        if "max_raw_unsafe_rate" in headline:
            f.write(f"- Max raw unsafe rate: {format_percent(headline['max_raw_unsafe_rate'])}\n")
            f.write(f"- Max shielded unsafe rate: {format_percent(headline['max_shielded_unsafe_rate'])}\n")
            f.write(f"- Minimum utility retention: {headline['min_utility_retention']:.6f}\n")
            f.write(f"- Maximum fallback rate: {format_percent(headline['max_fallback_rate'])}\n")

        f.write("\n## Pass/fail checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Interpretation\n\n")
        f.write("- Step 1 verifies the candidate-aware digital twin, ensemble uncertainty, calibrated risk bound, certified safe set, and raw-action shield.\n")
        f.write("- These are diagnostic results before the final PF/Greedy/DT/PPO/SAC paired-controller experiments.\n")
        f.write("- If all pass/fail checks are true, proceed to Step 2 raw-controller paired experiments.\n")

    print("Step 1.7 completed.")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()