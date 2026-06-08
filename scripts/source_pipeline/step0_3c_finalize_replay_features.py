import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_3c"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_REPLAY = PROCESSED_DIR / "commercial_twin_replay_timeseries_clean.csv"

KEY_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin"]


def safe_grant_ratio(granted, requested):
    granted = pd.to_numeric(granted, errors="coerce")
    requested = pd.to_numeric(requested, errors="coerce")

    ratio = pd.Series(np.nan, index=granted.index, dtype="float64")

    valid_request = requested > 1.0
    ratio.loc[valid_request] = granted.loc[valid_request] / requested.loc[valid_request]

    no_request = ~valid_request
    ratio.loc[no_request & (granted > 0)] = 1.0
    ratio.loc[no_request & ((granted <= 0) | granted.isna())] = 0.0

    return ratio.clip(lower=0.0, upper=2.0)


def summarize_numeric(df, cols):
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "count": int(vals.notna().sum()),
            "missing_rate": float(vals.isna().mean()),
            "mean": float(vals.mean()) if vals.notna().any() else None,
            "median": float(vals.median()) if vals.notna().any() else None,
            "p05": float(vals.quantile(0.05)) if vals.notna().any() else None,
            "p95": float(vals.quantile(0.95)) if vals.notna().any() else None,
            "max": float(vals.max()) if vals.notna().any() else None,
        }
    return out


def assign_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(KEY_COLS).copy()

    out["row_order"] = out.groupby(
        ["cluster", "slicing", "scheduling", "reservation"]
    ).cumcount()

    out["row_count"] = out.groupby(
        ["cluster", "slicing", "scheduling", "reservation"]
    )["row_order"].transform("max") + 1

    out["row_frac"] = out["row_order"] / out["row_count"].clip(lower=1)

    conditions = [
        out["row_frac"] < 0.50,
        out["row_frac"] < 0.70,
        out["row_frac"] < 0.80,
    ]
    choices = ["train", "calibration", "validation"]
    out["split"] = np.select(conditions, choices, default="test")

    return out


def main():
    print(f"Reading clean replay: {CLEAN_REPLAY}")
    df = pd.read_csv(CLEAN_REPLAY)

    original_rows = len(df)

    # 1) Fix PRB grant ratios.
    if "embb_granted_prbs" in df.columns and "embb_requested_prbs" in df.columns:
        df["embb_prb_grant_ratio_raw"] = df["embb_granted_prbs"] / (df["embb_requested_prbs"] + 1e-9)
        df["embb_prb_grant_ratio"] = safe_grant_ratio(df["embb_granted_prbs"], df["embb_requested_prbs"])

    if "urllc_granted_prbs" in df.columns and "urllc_requested_prbs" in df.columns:
        df["urllc_prb_grant_ratio_raw"] = df["urllc_granted_prbs"] / (df["urllc_requested_prbs"] + 1e-9)
        df["urllc_prb_grant_ratio"] = safe_grant_ratio(df["urllc_granted_prbs"], df["urllc_requested_prbs"])

    # 2) Fix delay proxy.
    # Preserve exact observed delay.
    if "urllc_delay_proxy_s" in df.columns:
        df["urllc_delay_proxy_s_raw"] = pd.to_numeric(df["urllc_delay_proxy_s"], errors="coerce")
    elif "app_delay_p95_s" in df.columns:
        df["urllc_delay_proxy_s_raw"] = pd.to_numeric(df["app_delay_p95_s"], errors="coerce")
    else:
        df["urllc_delay_proxy_s_raw"] = np.nan

    df["urllc_delay_is_observed"] = df["urllc_delay_proxy_s_raw"].notna().astype(int)

    # Build group-level delay fill values.
    # Use median of observed p95 delay within cluster/slicing/scheduling.
    group_cols_1 = ["cluster", "slicing", "scheduling"]
    group_delay = (
        df.groupby(group_cols_1, dropna=False)["urllc_delay_proxy_s_raw"]
        .median()
        .reset_index()
        .rename(columns={"urllc_delay_proxy_s_raw": "delay_fill_group_median_s"})
    )

    df = df.merge(group_delay, on=group_cols_1, how="left")

    global_delay_median = df["urllc_delay_proxy_s_raw"].median()
    if pd.isna(global_delay_median):
        global_delay_median = 0.05  # conservative fallback: 50 ms

    df["urllc_delay_proxy_s_filled"] = df["urllc_delay_proxy_s_raw"]
    df["urllc_delay_proxy_s_filled"] = df["urllc_delay_proxy_s_filled"].fillna(df["delay_fill_group_median_s"])
    df["urllc_delay_proxy_s_filled"] = df["urllc_delay_proxy_s_filled"].fillna(global_delay_median)

    # Clip extremely large delay outliers for stable downstream label construction.
    # Keep raw delay separately; use clipped filled delay for risk labeling.
    delay_p99 = df["urllc_delay_proxy_s_filled"].quantile(0.99)
    if pd.isna(delay_p99) or delay_p99 <= 0:
        delay_p99 = 5.0
    df["urllc_delay_proxy_s_final"] = df["urllc_delay_proxy_s_filled"].clip(lower=0, upper=delay_p99)

    # 3) Rename final proxies used later.
    df["urllc_violation_proxy"] = df["urllc_delay_proxy_s_final"]
    df["embb_satisfaction_proxy"] = pd.to_numeric(df["embb_throughput_mbps"], errors="coerce")

    # 4) Optional: missing flags for ML models.
    important_cols = [
        "embb_throughput_mbps",
        "urllc_throughput_mbps",
        "embb_buffer_bytes",
        "urllc_buffer_bytes",
        "embb_cqi",
        "urllc_cqi",
        "embb_slice_prb",
        "urllc_slice_prb",
        "embb_requested_prbs",
        "urllc_requested_prbs",
        "embb_granted_prbs",
        "urllc_granted_prbs",
    ]

    for c in important_cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)

    # Fill numeric missing values with group medians, then global medians.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_fill_cols = {
        "time_bin",
        "row_order",
        "row_count",
        "row_frac",
        "split",
    }

    fill_numeric_cols = [c for c in numeric_cols if c not in non_fill_cols]

    group_cols = ["cluster", "slicing", "scheduling"]
    for c in fill_numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df.groupby(group_cols, dropna=False)[c].transform("median"))
            df[c] = df[c].fillna(df[c].median())

    # 5) Reassign split after final cleaning.
    df = assign_chronological_split(df)

    final_path = PROCESSED_DIR / "commercial_twin_replay_timeseries_final.csv"
    df.to_csv(final_path, index=False, encoding="utf-8-sig")

    split_path = PROCESSED_DIR / "commercial_twin_split_index_final.csv"
    df[KEY_COLS + ["split"]].to_csv(split_path, index=False, encoding="utf-8-sig")

    check_cols = [
        "embb_throughput_mbps",
        "urllc_throughput_mbps",
        "embb_prb_grant_ratio",
        "urllc_prb_grant_ratio",
        "urllc_delay_proxy_s_raw",
        "urllc_delay_proxy_s_final",
        "embb_satisfaction_proxy",
        "urllc_violation_proxy",
    ]

    split_counts = df["split"].value_counts().to_dict()

    report = {
        "original_clean_rows": int(original_rows),
        "final_rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "global_delay_median_raw_observed": float(global_delay_median),
        "delay_clip_p99": float(delay_p99),
        "observed_delay_rows": int(df["urllc_delay_is_observed"].sum()),
        "observed_delay_rate": float(df["urllc_delay_is_observed"].mean()),
        "distribution_summary": summarize_numeric(df, check_cols),
        "outputs": {
            "final_replay": str(final_path),
            "final_split_index": str(split_path),
        }
    }

    report_path = PROCESSED_DIR / "commercial_twin_replay_final_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md = RESULT_DIR / "step0_3c_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Step 0.3C Final Replay Feature Quality Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Final rows: {len(df)}\n")
        f.write(f"- Observed delay rows: {report['observed_delay_rows']}\n")
        f.write(f"- Observed delay rate: {report['observed_delay_rate']:.6f}\n")
        f.write(f"- Global observed delay median: {global_delay_median}\n")
        f.write(f"- Delay clip p99: {delay_p99}\n\n")

        f.write("## Split counts\n\n")
        for k, v in split_counts.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- `urllc_delay_proxy_s_raw` is the delay value directly matched from APP logs.\n")
        f.write("- `urllc_delay_proxy_s_final` fills missing delay by cluster/slicing/scheduling median.\n")
        f.write("- `urllc_delay_is_observed` marks whether the delay was directly observed.\n")
        f.write("- PRB grant ratios are safely recomputed and clipped to [0, 2].\n")
        f.write("- Use `commercial_twin_replay_timeseries_final.csv` for Step 0.4.\n")

    print("Step 0.3C completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()