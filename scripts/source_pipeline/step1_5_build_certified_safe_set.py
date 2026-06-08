import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIB_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_calibration"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step1_5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TEST = CALIB_DIR / "step1_4_test_predictions_with_upper.csv"

Q_LEVEL = 90

# Use sanity thresholds consistent with Step 1.4 for first certified-set analysis.
# Later main experiment will use explicit SLA epsilons.
THRESHOLD_MODE = "mean_plus_half_std"

RISK_TARGETS = ["V_embb", "V_urllc"]


def compute_threshold(vals, mode):
    vals = pd.to_numeric(vals, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        raise RuntimeError("Cannot compute threshold from empty values.")

    if mode == "mean_plus_half_std":
        return float(vals.mean() + 0.5 * vals.std())
    if mode == "p80":
        return float(np.quantile(vals, 0.80))
    if mode == "p90":
        return float(np.quantile(vals, 0.90))

    raise ValueError(f"Unknown threshold mode: {mode}")


def add_safe_labels(df, eps_embb, eps_urllc):
    out = df.copy()

    out["true_safe_embb"] = (out["true_V_embb"] <= eps_embb).astype(int)
    out["true_safe_urllc"] = (out["true_V_urllc"] <= eps_urllc).astype(int)
    out["true_joint_safe"] = (
        (out["true_safe_embb"] == 1)
        & (out["true_safe_urllc"] == 1)
    ).astype(int)

    out["raw_pred_safe_embb"] = (out["pred_mean_V_embb"] <= eps_embb).astype(int)
    out["raw_pred_safe_urllc"] = (out["pred_mean_V_urllc"] <= eps_urllc).astype(int)
    out["raw_pred_joint_safe"] = (
        (out["raw_pred_safe_embb"] == 1)
        & (out["raw_pred_safe_urllc"] == 1)
    ).astype(int)

    out["cert_safe_embb"] = (out[f"upper_V_embb_q{Q_LEVEL}"] <= eps_embb).astype(int)
    out["cert_safe_urllc"] = (out[f"upper_V_urllc_q{Q_LEVEL}"] <= eps_urllc).astype(int)
    out["cert_joint_safe"] = (
        (out["cert_safe_embb"] == 1)
        & (out["cert_safe_urllc"] == 1)
    ).astype(int)

    return out


def safe_set_metrics(df, rule_col, true_col="true_joint_safe"):
    pred_safe = df[rule_col].astype(bool).to_numpy()
    true_safe = df[true_col].astype(bool).to_numpy()

    safe_ratio = float(pred_safe.mean())

    if pred_safe.sum() > 0:
        false_safe_rate = float((pred_safe & (~true_safe)).sum() / pred_safe.sum())
        precision = float((pred_safe & true_safe).sum() / pred_safe.sum())
    else:
        false_safe_rate = 0.0
        precision = 0.0

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
        "rule": rule_col,
        "safe_ratio": safe_ratio,
        "false_safe_rate": false_safe_rate,
        "safe_precision": precision,
        "unsafe_interception_rate": unsafe_interception_rate,
        "safe_rejection_rate": safe_rejection_rate,
    }


def build_pseudo_groups(df, group_size=5):
    # Step 1.1B expanded each sampled state with all 5 observed actions.
    # The saved calibration/test prediction file is capped, but rows should still preserve groups by contiguous blocks.
    out = df.copy().reset_index(drop=True)
    out["candidate_group_id"] = np.arange(len(out)) // group_size
    out["candidate_id"] = np.arange(len(out)) % group_size
    return out


def group_level_summary(df):
    g = (
        df.groupby("candidate_group_id")
        .agg(
            num_candidates=("candidate_id", "count"),
            num_true_safe=("true_joint_safe", "sum"),
            num_raw_pred_safe=("raw_pred_joint_safe", "sum"),
            num_cert_safe=("cert_joint_safe", "sum"),
            mean_true_V_embb=("true_V_embb", "mean"),
            mean_true_V_urllc=("true_V_urllc", "mean"),
            max_true_utility=("true_management_utility", "max"),
        )
        .reset_index()
    )

    g["has_true_safe"] = (g["num_true_safe"] > 0).astype(int)
    g["has_raw_pred_safe"] = (g["num_raw_pred_safe"] > 0).astype(int)
    g["has_cert_safe"] = (g["num_cert_safe"] > 0).astype(int)

    return g


def main():
    print(f"Reading test calibrated predictions: {INPUT_TEST}")
    df = pd.read_csv(INPUT_TEST)

    # Ensure needed utility column exists. It should be saved by Step 1.4.
    if "true_management_utility" not in df.columns:
        raise RuntimeError("Missing true_management_utility in input file.")

    eps_embb = compute_threshold(df["true_V_embb"], THRESHOLD_MODE)
    eps_urllc = compute_threshold(df["true_V_urllc"], THRESHOLD_MODE)

    df = add_safe_labels(df, eps_embb, eps_urllc)
    df = build_pseudo_groups(df, group_size=5)

    metrics_rows = []
    metrics_rows.append(safe_set_metrics(df, "raw_pred_joint_safe"))
    metrics_rows.append(safe_set_metrics(df, "cert_joint_safe"))

    # Also per-slice metrics.
    metrics_rows.append(safe_set_metrics(df, "raw_pred_safe_embb", true_col="true_safe_embb"))
    metrics_rows.append(safe_set_metrics(df, "cert_safe_embb", true_col="true_safe_embb"))
    metrics_rows.append(safe_set_metrics(df, "raw_pred_safe_urllc", true_col="true_safe_urllc"))
    metrics_rows.append(safe_set_metrics(df, "cert_safe_urllc", true_col="true_safe_urllc"))

    metrics_df = pd.DataFrame(metrics_rows)

    group_df = group_level_summary(df)

    group_metrics = {
        "num_groups": int(len(group_df)),
        "mean_candidates_per_group": float(group_df["num_candidates"].mean()),
        "group_has_true_safe_rate": float(group_df["has_true_safe"].mean()),
        "group_has_raw_pred_safe_rate": float(group_df["has_raw_pred_safe"].mean()),
        "group_has_cert_safe_rate": float(group_df["has_cert_safe"].mean()),
        "mean_true_safe_candidates_per_group": float(group_df["num_true_safe"].mean()),
        "mean_raw_pred_safe_candidates_per_group": float(group_df["num_raw_pred_safe"].mean()),
        "mean_cert_safe_candidates_per_group": float(group_df["num_cert_safe"].mean()),
        "no_cert_safe_group_rate": float((group_df["num_cert_safe"] == 0).mean()),
    }

    safe_labeled_path = OUT_DIR / "step1_5_safe_labeled_candidates.csv"
    metrics_path = OUT_DIR / "step1_5_safe_set_summary.csv"
    group_path = OUT_DIR / "step1_5_group_level_summary.csv"

    # Save capped for inspection.
    df.to_csv(safe_labeled_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    group_df.to_csv(group_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_file": str(INPUT_TEST),
        "q_level": Q_LEVEL,
        "threshold_mode": THRESHOLD_MODE,
        "eps_embb": eps_embb,
        "eps_urllc": eps_urllc,
        "num_candidates": int(len(df)),
        "group_metrics": group_metrics,
        "outputs": {
            "safe_labeled_candidates": str(safe_labeled_path),
            "safe_set_summary": str(metrics_path),
            "group_level_summary": str(group_path),
        },
        "important_note": (
            "This step constructs a candidate-level certified safe set using q90 calibrated upper bounds. "
            "Thresholds are sanity thresholds from test prediction distribution; final experiments should use explicit SLA epsilons."
        )
    }

    summary_path = OUT_DIR / "step1_5_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = RESULT_DIR / "step1_5_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 1.5 Certified Safe Candidate Set Report\n\n")

        f.write("## Thresholds\n\n")
        f.write(f"- q level: {Q_LEVEL}\n")
        f.write(f"- threshold mode: {THRESHOLD_MODE}\n")
        f.write(f"- eps_embb: {eps_embb}\n")
        f.write(f"- eps_urllc: {eps_urllc}\n\n")

        f.write("## Group-level metrics\n\n")
        for k, v in group_metrics.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- Certified safe set is based on calibrated upper bounds, not raw predictions.\n")
        f.write("- Compare raw_pred_joint_safe and cert_joint_safe to evaluate false-safe reduction.\n")
        f.write("- The next step will apply raw-action shielding on the candidate groups.\n")

    print("Step 1.5 completed.")
    print(json.dumps(summary, indent=2))
    print("\nSafe-set summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()