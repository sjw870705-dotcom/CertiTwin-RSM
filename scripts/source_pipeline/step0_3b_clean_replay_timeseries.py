import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_3b"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

REPLAY_PATH = PROCESSED_DIR / "commercial_twin_replay_timeseries.csv"

KEY_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin"]

NUMERIC_COLS = [
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
    "app_delay_mean_s",
    "app_delay_median_s",
    "app_delay_p95_s",
    "app_payload_size_mean",
    "embb_prb_grant_ratio",
    "urllc_prb_grant_ratio",
    "urllc_delay_proxy_s",
    "embb_satisfaction_proxy",
    "urllc_violation_proxy",
]


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
    print(f"Reading replay table: {REPLAY_PATH}")
    df = pd.read_csv(REPLAY_PATH)

    original_rows = len(df)
    original_unique_keys = df[KEY_COLS].drop_duplicates().shape[0]
    duplicate_rows = original_rows - original_unique_keys

    print(f"Original rows: {original_rows}")
    print(f"Unique key rows: {original_unique_keys}")
    print(f"Duplicate rows by key: {duplicate_rows}")

    # Ensure numeric columns are numeric.
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Collapse duplicated time bins.
    agg_dict = {}
    for c in NUMERIC_COLS:
        if c in df.columns:
            if c in {"app_delay_p95_s", "urllc_delay_proxy_s", "urllc_violation_proxy"}:
                # Conservative for URLLC: keep high percentile / max over duplicated app entries.
                agg_dict[c] = "max"
            else:
                agg_dict[c] = "mean"

    clean = (
        df.groupby(KEY_COLS, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    # Recompute ratios after averaging requested/granted PRBs.
    if "embb_granted_prbs" in clean.columns and "embb_requested_prbs" in clean.columns:
        clean["embb_prb_grant_ratio"] = clean["embb_granted_prbs"] / (clean["embb_requested_prbs"] + 1e-9)

    if "urllc_granted_prbs" in clean.columns and "urllc_requested_prbs" in clean.columns:
        clean["urllc_prb_grant_ratio"] = clean["urllc_granted_prbs"] / (clean["urllc_requested_prbs"] + 1e-9)

    if "embb_throughput_mbps" in clean.columns:
        clean["embb_satisfaction_proxy"] = clean["embb_throughput_mbps"]

    if "urllc_delay_proxy_s" in clean.columns:
        clean["urllc_violation_proxy"] = clean["urllc_delay_proxy_s"]

    # Drop rows where both eMBB and URLLC are mostly missing.
    important_cols = [
        "embb_throughput_mbps",
        "urllc_throughput_mbps",
        "embb_slice_prb",
        "urllc_slice_prb",
    ]
    available_important = [c for c in important_cols if c in clean.columns]
    if available_important:
        before = len(clean)
        clean = clean.dropna(subset=available_important, how="all")
        after = len(clean)
    else:
        before = len(clean)
        after = len(clean)

    clean = assign_chronological_split(clean)

    clean_path = PROCESSED_DIR / "commercial_twin_replay_timeseries_clean.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    split_index_clean = clean[KEY_COLS + ["split"]].copy()
    split_index_path = PROCESSED_DIR / "commercial_twin_split_index_clean.csv"
    split_index_clean.to_csv(split_index_path, index=False, encoding="utf-8-sig")

    split_counts = clean["split"].value_counts().to_dict()

    missing_rates = {}
    for c in NUMERIC_COLS:
        if c in clean.columns:
            missing_rates[c] = float(clean[c].isna().mean())

    # Basic distribution summary for key columns.
    key_dist_cols = [
        "embb_throughput_mbps",
        "urllc_throughput_mbps",
        "embb_buffer_bytes",
        "urllc_buffer_bytes",
        "embb_slice_prb",
        "urllc_slice_prb",
        "urllc_delay_proxy_s",
        "embb_prb_grant_ratio",
        "urllc_prb_grant_ratio",
    ]

    dist_summary = {}
    for c in key_dist_cols:
        if c in clean.columns:
            vals = pd.to_numeric(clean[c], errors="coerce")
            dist_summary[c] = {
                "count": int(vals.notna().sum()),
                "mean": float(vals.mean()) if vals.notna().any() else None,
                "median": float(vals.median()) if vals.notna().any() else None,
                "p05": float(vals.quantile(0.05)) if vals.notna().any() else None,
                "p95": float(vals.quantile(0.95)) if vals.notna().any() else None,
                "missing_rate": float(vals.isna().mean()),
            }

    report = {
        "original_rows": int(original_rows),
        "original_unique_key_rows": int(original_unique_keys),
        "duplicate_rows_by_key": int(duplicate_rows),
        "clean_rows": int(len(clean)),
        "rows_dropped_all_important_missing": int(before - after),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "missing_rates": missing_rates,
        "distribution_summary": dist_summary,
        "outputs": {
            "clean_replay": str(clean_path),
            "clean_split_index": str(split_index_path),
        }
    }

    report_json_path = PROCESSED_DIR / "commercial_twin_replay_quality_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md_path = RESULT_DIR / "step0_3b_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.3B CommercialTwin Replay Quality Report\n\n")
        f.write("## Row quality\n\n")
        f.write(f"- Original rows: {original_rows}\n")
        f.write(f"- Unique key rows: {original_unique_keys}\n")
        f.write(f"- Duplicate rows by key: {duplicate_rows}\n")
        f.write(f"- Clean rows: {len(clean)}\n")
        f.write(f"- Rows dropped because all important fields were missing: {before - after}\n\n")

        f.write("## Split counts\n\n")
        for k, v in split_counts.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- Duplicated rows with the same cluster/slicing/scheduling/reservation/time_bin were collapsed.\n")
        f.write("- URLLC delay proxy uses conservative max aggregation when duplicated app-delay values exist.\n")
        f.write("- The clean replay table should be used for Step 0.4 and all following experiments.\n")
        f.write("- The main CommercialTwin-calibrated protocol remains eMBB+URLLC only.\n")

    print("Step 0.3B completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()