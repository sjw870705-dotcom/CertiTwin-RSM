import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_4"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "commercial_twin_replay_timeseries_final.csv"

OUTPUT_LABELS = PROCESSED_DIR / "commercial_twin_supervised_labels.csv"
OUTPUT_THRESHOLDS = PROCESSED_DIR / "commercial_twin_sla_thresholds.json"
OUTPUT_REPORT_JSON = PROCESSED_DIR / "commercial_twin_label_quality_report.json"
OUTPUT_REPORT_MD = RESULT_DIR / "step0_4_report.md"


# Default SLA quantile settings.
EMBB_R_MIN_QUANTILE = 0.25
URLLC_D_MAX_QUANTILE = 0.90

# Risk weights.
W_EMBB_THR = 0.80
W_EMBB_GRANT = 0.20

W_URLLC_DELAY = 0.75
W_URLLC_ERR = 0.15
W_URLLC_GRANT = 0.10

# Management utility weights.
ALPHA_EMBB = 0.45
BETA_URLLC_SAFETY = 0.25
GAMMA_FAIRNESS = 0.20
ETA_RECONFIG = 0.10


ID_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin", "split"]

FEATURE_COLS = [
    "embb_buffer_bytes",
    "urllc_buffer_bytes",
    "embb_cqi",
    "urllc_cqi",
    "embb_slice_prb",
    "urllc_slice_prb",
    "embb_granted_prbs",
    "urllc_granted_prbs",
    "embb_requested_prbs",
    "urllc_requested_prbs",
    "embb_throughput_mbps",
    "urllc_throughput_mbps",
    "embb_tx_error_pct",
    "urllc_tx_error_pct",
    "embb_ul_sinr",
    "urllc_ul_sinr",
    "embb_prb_grant_ratio",
    "urllc_prb_grant_ratio",
    "urllc_delay_proxy_s_raw",
    "urllc_delay_proxy_s_final",
    "urllc_delay_is_observed",
]

MISSING_FLAG_COLS = [
    "embb_throughput_mbps_missing",
    "urllc_throughput_mbps_missing",
    "embb_buffer_bytes_missing",
    "urllc_buffer_bytes_missing",
    "embb_cqi_missing",
    "urllc_cqi_missing",
    "embb_slice_prb_missing",
    "urllc_slice_prb_missing",
    "embb_requested_prbs_missing",
    "urllc_requested_prbs_missing",
    "embb_granted_prbs_missing",
    "urllc_granted_prbs_missing",
]


def safe_quantile(series, q, fallback):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return fallback
    val = float(vals.quantile(q))
    if not np.isfinite(val):
        return fallback
    return val


def positive_part(x):
    return np.maximum(x, 0.0)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def jain_fairness(x, y, eps=1e-9):
    numerator = (x + y) ** 2
    denominator = 2.0 * (x ** 2 + y ** 2 + eps)
    return numerator / denominator


def summarize_by_split(df, cols):
    records = []
    for split, g in df.groupby("split"):
        rec = {"split": split, "num_rows": int(len(g))}
        for c in cols:
            if c in g.columns:
                vals = pd.to_numeric(g[c], errors="coerce")
                rec[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else None
                rec[f"{c}_median"] = float(vals.median()) if vals.notna().any() else None
                rec[f"{c}_p95"] = float(vals.quantile(0.95)) if vals.notna().any() else None
                rec[f"{c}_non_null"] = int(vals.notna().sum())
        records.append(rec)
    return records


def main():
    print(f"Reading final replay: {INPUT_CSV}")

    # Read all columns because Step 0.3C may have generated useful missing flags.
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded rows: {len(df)}")

    if "split" not in df.columns:
        raise RuntimeError("Column `split` is missing. Please rerun Step 0.3C.")

    train = df[df["split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("Train split is empty.")

    # Convert important fields to numeric.
    numeric_needed = [
        "embb_throughput_mbps",
        "urllc_delay_proxy_s_final",
        "embb_prb_grant_ratio",
        "urllc_prb_grant_ratio",
        "urllc_tx_error_pct",
        "embb_slice_prb",
        "urllc_slice_prb",
        "urllc_throughput_mbps",
    ]

    for c in numeric_needed:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) Define SLA thresholds using train split only.
    embb_R_min = safe_quantile(
        train["embb_throughput_mbps"],
        EMBB_R_MIN_QUANTILE,
        fallback=1.0
    )

    urllc_D_max = safe_quantile(
        train["urllc_delay_proxy_s_final"],
        URLLC_D_MAX_QUANTILE,
        fallback=0.1
    )

    # Error threshold used for normalized reliability risk.
    urllc_err_ref = safe_quantile(
        train["urllc_tx_error_pct"],
        0.95,
        fallback=1.0
    )
    urllc_err_ref = max(urllc_err_ref, 1e-6)

    # Utility normalization caps from train split.
    embb_thr_cap = safe_quantile(train["embb_throughput_mbps"], 0.95, fallback=max(embb_R_min, 1.0))
    urllc_thr_cap = safe_quantile(train["urllc_throughput_mbps"], 0.95, fallback=0.02)

    # 2) eMBB risk.
    embb_thr = df["embb_throughput_mbps"].astype(float)
    embb_grant = df["embb_prb_grant_ratio"].astype(float).clip(0, 2)

    df["embb_throughput_shortfall"] = positive_part((embb_R_min - embb_thr) / (embb_R_min + 1e-9))
    df["embb_grant_shortfall"] = positive_part(1.0 - embb_grant)

    df["V_embb"] = (
        W_EMBB_THR * df["embb_throughput_shortfall"]
        + W_EMBB_GRANT * df["embb_grant_shortfall"]
    )

    df["embb_violation"] = (df["embb_throughput_shortfall"] > 0).astype(int)

    # 3) URLLC risk.
    urllc_delay = df["urllc_delay_proxy_s_final"].astype(float)
    urllc_err = df["urllc_tx_error_pct"].astype(float)
    urllc_grant = df["urllc_prb_grant_ratio"].astype(float).clip(0, 2)

    df["urllc_delay_excess"] = positive_part((urllc_delay - urllc_D_max) / (urllc_D_max + 1e-9))
    df["urllc_error_risk"] = (urllc_err / (urllc_err_ref + 1e-9)).clip(0, 1)
    df["urllc_grant_shortfall"] = positive_part(1.0 - urllc_grant)

    df["V_urllc"] = (
        W_URLLC_DELAY * df["urllc_delay_excess"]
        + W_URLLC_ERR * df["urllc_error_risk"]
        + W_URLLC_GRANT * df["urllc_grant_shortfall"]
    )

    df["urllc_violation"] = (df["urllc_delay_excess"] > 0).astype(int)

    # 4) Total risk.
    df["V_total"] = 0.5 * df["V_embb"] + 0.5 * df["V_urllc"]

    # 5) Reconfiguration cost.
    df = df.sort_values(["cluster", "slicing", "scheduling", "reservation", "time_bin"]).copy()

    group_cols = ["cluster", "slicing", "scheduling", "reservation"]

    df["embb_slice_prb_prev"] = df.groupby(group_cols)["embb_slice_prb"].shift(1)
    df["urllc_slice_prb_prev"] = df.groupby(group_cols)["urllc_slice_prb"].shift(1)

    df["reconfig_cost"] = (
        (df["embb_slice_prb"] - df["embb_slice_prb_prev"]).abs().fillna(0)
        + (df["urllc_slice_prb"] - df["urllc_slice_prb_prev"]).abs().fillna(0)
    ) / 50.0

    df["reconfig_cost"] = df["reconfig_cost"].clip(0, 2)

    # 6) Management utility.
    df["embb_utility_norm"] = (df["embb_throughput_mbps"] / (embb_thr_cap + 1e-9)).clip(0, 1)
    df["urllc_safety_norm"] = (1.0 - df["V_urllc"].clip(0, 1)).clip(0, 1)

    urllc_thr_norm = (df["urllc_throughput_mbps"] / (urllc_thr_cap + 1e-9)).clip(0, 1)
    df["slice_fairness"] = jain_fairness(df["embb_utility_norm"], urllc_thr_norm).clip(0, 1)

    df["management_utility"] = (
        ALPHA_EMBB * df["embb_utility_norm"]
        + BETA_URLLC_SAFETY * df["urllc_safety_norm"]
        + GAMMA_FAIRNESS * df["slice_fairness"]
        - ETA_RECONFIG * df["reconfig_cost"]
    )

    # 7) Keep useful columns only.
    available_feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    available_missing_cols = [c for c in MISSING_FLAG_COLS if c in df.columns]

    label_cols = [
        "embb_throughput_shortfall",
        "embb_grant_shortfall",
        "V_embb",
        "embb_violation",
        "urllc_delay_excess",
        "urllc_error_risk",
        "urllc_grant_shortfall",
        "V_urllc",
        "urllc_violation",
        "V_total",
        "reconfig_cost",
        "embb_utility_norm",
        "urllc_safety_norm",
        "slice_fairness",
        "management_utility",
    ]

    out_cols = ID_COLS + available_feature_cols + available_missing_cols + label_cols
    out_cols = [c for c in out_cols if c in df.columns]

    out = df[out_cols].copy()
    out.to_csv(OUTPUT_LABELS, index=False, encoding="utf-8-sig")

    thresholds = {
        "embb_R_min_mbps": embb_R_min,
        "urllc_D_max_s": urllc_D_max,
        "urllc_err_ref_pct": urllc_err_ref,
        "embb_thr_cap_p95": embb_thr_cap,
        "urllc_thr_cap_p95": urllc_thr_cap,
        "threshold_source": "train split only",
        "embb_R_min_quantile": EMBB_R_MIN_QUANTILE,
        "urllc_D_max_quantile": URLLC_D_MAX_QUANTILE,
        "risk_weights": {
            "embb": {
                "throughput_shortfall": W_EMBB_THR,
                "grant_shortfall": W_EMBB_GRANT,
            },
            "urllc": {
                "delay_excess": W_URLLC_DELAY,
                "error_risk": W_URLLC_ERR,
                "grant_shortfall": W_URLLC_GRANT,
            }
        },
        "utility_weights": {
            "embb_utility": ALPHA_EMBB,
            "urllc_safety": BETA_URLLC_SAFETY,
            "fairness": GAMMA_FAIRNESS,
            "reconfiguration": ETA_RECONFIG,
        }
    }

    with open(OUTPUT_THRESHOLDS, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    quality_cols = [
        "V_embb",
        "V_urllc",
        "V_total",
        "embb_violation",
        "urllc_violation",
        "management_utility",
        "reconfig_cost",
        "embb_utility_norm",
        "urllc_safety_norm",
        "slice_fairness",
    ]

    report = {
        "num_rows": int(len(out)),
        "split_counts": {str(k): int(v) for k, v in out["split"].value_counts().to_dict().items()},
        "thresholds": thresholds,
        "label_summary_by_split": summarize_by_split(out, quality_cols),
        "outputs": {
            "labels": str(OUTPUT_LABELS),
            "thresholds": str(OUTPUT_THRESHOLDS),
            "report_json": str(OUTPUT_REPORT_JSON),
        }
    }

    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 0.4 SLA Label Construction Report\n\n")
        f.write("## Thresholds\n\n")
        f.write(f"- eMBB R_min: {embb_R_min:.6f} Mbps\n")
        f.write(f"- URLLC D_max: {urllc_D_max:.6f} s\n")
        f.write(f"- URLLC error reference: {urllc_err_ref:.6f} %\n")
        f.write(f"- Threshold source: train split only\n\n")

        f.write("## Split counts\n\n")
        for k, v in report["split_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- eMBB risk is based on throughput shortfall and PRB grant shortfall.\n")
        f.write("- URLLC risk is based on delay excess, error risk, and PRB grant shortfall.\n")
        f.write("- URLLC delay is a proxy derived from sparse APP delay alignment and group-level filling.\n")
        f.write("- The SLA thresholds are derived only from the train split to avoid test leakage.\n")
        f.write("- This file is used for Step 0.5 digital-twin dataset construction.\n")

    print("Step 0.4 completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()