import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "commercial_twin"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_3"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BS_FILES = INTERIM_DIR / "step0_2_bs_metrics_files.csv"
APP_FILES = INTERIM_DIR / "step0_2_app_log_files.csv"

# Provisional mapping from Step 0.2B.
SLICE_MAP = {
    0: "embb",
    1: "urllc",
}

BS_USECOLS = [
    "Timestamp",
    "slice_id",
    "slice_prb",
    "scheduling_policy",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_errors downlink (%)",
    "dl_cqi",
    "ul_sinr",
    "sum_requested_prbs",
    "sum_granted_prbs",
]

APP_USECOLS = [
    "Received Time",
    "Sent Time",
    "Timestamp_recv",
    "Timestamp_send",
    "Protocol Payload Size",
]


def safe_read_csv(path, usecols=None):
    try:
        if usecols is None:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=lambda c: c in usecols)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


def add_path_tags(df, row):
    df["cluster"] = row.get("cluster", "")
    df["slicing"] = row.get("slicing", "")
    df["scheduling"] = row.get("scheduling", "")
    df["reservation"] = row.get("reservation", "")
    return df


def build_time_bin(series, decimals=0):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    # Relative time in seconds, rounded to integer seconds by default.
    rel = vals - vals.min()
    return rel.round(decimals).astype("Int64")


def aggregate_bs_metrics(max_files=None):
    bs_files = pd.read_csv(BS_FILES)

    if max_files is not None:
        bs_files = bs_files.head(max_files)

    parts = []

    for _, row in tqdm(bs_files.iterrows(), total=len(bs_files), desc="Aggregating BS metrics"):
        path = Path(row["path"])
        df = safe_read_csv(path, usecols=BS_USECOLS)
        if df is None or "slice_id" not in df.columns or "Timestamp" not in df.columns:
            continue

        df = add_path_tags(df, row)

        df["slice_id"] = pd.to_numeric(df["slice_id"], errors="coerce")
        df["slice_name"] = df["slice_id"].map(SLICE_MAP)

        # Keep only eMBB/URLLC mapped rows.
        df = df[df["slice_name"].isin(["embb", "urllc"])].copy()
        if df.empty:
            continue

        df["time_bin"] = build_time_bin(df["Timestamp"], decimals=0)

        for col in [
            "slice_prb",
            "dl_buffer [bytes]",
            "tx_brate downlink [Mbps]",
            "tx_errors downlink (%)",
            "dl_cqi",
            "ul_sinr",
            "sum_requested_prbs",
            "sum_granted_prbs",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        group_cols = ["cluster", "slicing", "scheduling", "reservation", "time_bin", "slice_name"]

        agg = (
            df.groupby(group_cols, dropna=False)
            .agg({
                "slice_prb": "mean",
                "dl_buffer [bytes]": "mean",
                "tx_brate downlink [Mbps]": "mean",
                "tx_errors downlink (%)": "mean",
                "dl_cqi": "mean",
                "ul_sinr": "mean",
                "sum_requested_prbs": "mean",
                "sum_granted_prbs": "mean",
            })
            .reset_index()
        )

        parts.append(agg)

    if not parts:
        raise RuntimeError("No BS metrics aggregated.")

    bs_long = pd.concat(parts, ignore_index=True)

    # Pivot eMBB/URLLC into wide replay table.
    value_cols = [
        "slice_prb",
        "dl_buffer [bytes]",
        "tx_brate downlink [Mbps]",
        "tx_errors downlink (%)",
        "dl_cqi",
        "ul_sinr",
        "sum_requested_prbs",
        "sum_granted_prbs",
    ]

    wide = bs_long.pivot_table(
        index=["cluster", "slicing", "scheduling", "reservation", "time_bin"],
        columns="slice_name",
        values=value_cols,
        aggfunc="mean"
    )

    wide.columns = [f"{slice_name}_{field}" for field, slice_name in wide.columns]
    wide = wide.reset_index()

    rename_map = {
        "embb_tx_brate downlink [Mbps]": "embb_throughput_mbps",
        "embb_dl_buffer [bytes]": "embb_buffer_bytes",
        "embb_slice_prb": "embb_slice_prb",
        "embb_tx_errors downlink (%)": "embb_tx_error_pct",
        "embb_dl_cqi": "embb_cqi",
        "embb_ul_sinr": "embb_ul_sinr",
        "embb_sum_requested_prbs": "embb_requested_prbs",
        "embb_sum_granted_prbs": "embb_granted_prbs",

        "urllc_tx_brate downlink [Mbps]": "urllc_throughput_mbps",
        "urllc_dl_buffer [bytes]": "urllc_buffer_bytes",
        "urllc_slice_prb": "urllc_slice_prb",
        "urllc_tx_errors downlink (%)": "urllc_tx_error_pct",
        "urllc_dl_cqi": "urllc_cqi",
        "urllc_ul_sinr": "urllc_ul_sinr",
        "urllc_sum_requested_prbs": "urllc_requested_prbs",
        "urllc_sum_granted_prbs": "urllc_granted_prbs",
    }

    wide = wide.rename(columns=rename_map)

    return wide


def aggregate_app_delay(max_files=None):
    app_files = pd.read_csv(APP_FILES)

    if max_files is not None:
        app_files = app_files.head(max_files)

    parts = []

    for _, row in tqdm(app_files.iterrows(), total=len(app_files), desc="Aggregating APP delay"):
        path = Path(row["path"])
        df = safe_read_csv(path, usecols=APP_USECOLS)
        if df is None:
            continue

        df = add_path_tags(df, row)

        delay = None
        if "Received Time" in df.columns and "Sent Time" in df.columns:
            recv = pd.to_numeric(df["Received Time"], errors="coerce")
            send = pd.to_numeric(df["Sent Time"], errors="coerce")
            delay = recv - send
            time_source = recv
        elif "Timestamp_recv" in df.columns and "Timestamp_send" in df.columns:
            recv = pd.to_numeric(df["Timestamp_recv"], errors="coerce")
            send = pd.to_numeric(df["Timestamp_send"], errors="coerce")
            delay = recv - send
            time_source = recv
        else:
            continue

        df["app_delay_s"] = delay
        df["time_bin"] = build_time_bin(time_source, decimals=0)

        if "Protocol Payload Size" in df.columns:
            df["Protocol Payload Size"] = pd.to_numeric(df["Protocol Payload Size"], errors="coerce")
        else:
            df["Protocol Payload Size"] = np.nan

        agg = (
            df.groupby(["cluster", "slicing", "scheduling", "reservation", "time_bin"], dropna=False)
            .agg({
                "app_delay_s": ["mean", "median", lambda x: x.quantile(0.95)],
                "Protocol Payload Size": "mean",
            })
            .reset_index()
        )

        agg.columns = [
            "cluster", "slicing", "scheduling", "reservation", "time_bin",
            "app_delay_mean_s", "app_delay_median_s", "app_delay_p95_s",
            "app_payload_size_mean"
        ]

        parts.append(agg)

    if not parts:
        print("[WARN] No APP delay aggregated.")
        return None

    return pd.concat(parts, ignore_index=True)


def add_proxy_metrics(df):
    out = df.copy()

    # eMBB satisfaction proxy: normalized throughput by slicing-level p50/p75 later.
    # For now store raw fields and simple pressure ratios.
    out["embb_prb_grant_ratio"] = out["embb_granted_prbs"] / (out["embb_requested_prbs"] + 1e-9)
    out["urllc_prb_grant_ratio"] = out["urllc_granted_prbs"] / (out["urllc_requested_prbs"] + 1e-9)

    # Delay proxy is app-level because app log may not include slice_id.
    # Conservative assignment: use it as URLLC delay proxy in the two-slice main protocol.
    if "app_delay_p95_s" in out.columns:
        out["urllc_delay_proxy_s"] = out["app_delay_p95_s"]
    else:
        out["urllc_delay_proxy_s"] = np.nan

    # Simple placeholders for next-step thresholding.
    out["embb_satisfaction_proxy"] = out["embb_throughput_mbps"]
    out["urllc_violation_proxy"] = out["urllc_delay_proxy_s"]

    return out


def main():
    # Use full files by default. If it is too slow, set max_files for debug.
    bs_wide = aggregate_bs_metrics(max_files=None)
    bs_path = PROCESSED_DIR / "commercial_twin_bs_timeseries_wide.csv"
    bs_wide.to_csv(bs_path, index=False, encoding="utf-8-sig")

    app_delay = aggregate_app_delay(max_files=None)
    if app_delay is not None:
        app_path = PROCESSED_DIR / "commercial_twin_app_delay_timeseries.csv"
        app_delay.to_csv(app_path, index=False, encoding="utf-8-sig")

        replay = bs_wide.merge(
            app_delay,
            on=["cluster", "slicing", "scheduling", "reservation", "time_bin"],
            how="left"
        )
    else:
        app_path = None
        replay = bs_wide

    replay = add_proxy_metrics(replay)

    replay_path = PROCESSED_DIR / "commercial_twin_replay_timeseries.csv"
    replay.to_csv(replay_path, index=False, encoding="utf-8-sig")

    # Chronological split per reservation.
    replay_sorted = replay.sort_values(["cluster", "slicing", "scheduling", "reservation", "time_bin"]).copy()
    replay_sorted["row_order"] = replay_sorted.groupby(
        ["cluster", "slicing", "scheduling", "reservation"]
    ).cumcount()
    replay_sorted["row_count"] = replay_sorted.groupby(
        ["cluster", "slicing", "scheduling", "reservation"]
    )["row_order"].transform("max") + 1
    replay_sorted["row_frac"] = replay_sorted["row_order"] / replay_sorted["row_count"].clip(lower=1)

    conditions = [
        replay_sorted["row_frac"] < 0.50,
        replay_sorted["row_frac"] < 0.70,
        replay_sorted["row_frac"] < 0.80,
    ]
    choices = ["train", "calibration", "validation"]
    replay_sorted["split"] = np.select(conditions, choices, default="test")

    split_index = replay_sorted[
        ["cluster", "slicing", "scheduling", "reservation", "time_bin", "split"]
    ].copy()
    split_path = PROCESSED_DIR / "commercial_twin_split_index.csv"
    split_index.to_csv(split_path, index=False, encoding="utf-8-sig")

    # Basic summary.
    summary = {
        "num_replay_rows": int(len(replay)),
        "columns": list(replay.columns),
        "slice_mapping": {
            "0": "eMBB",
            "1": "URLLC"
        },
        "main_protocol": "CommercialTwin-calibrated ns-O-RAN, eMBB+URLLC only",
        "delay_proxy": "urllc_delay_proxy_s = app_delay_p95_s from APP logs",
        "outputs": {
            "bs_wide": str(bs_path),
            "app_delay": str(app_path) if app_path else None,
            "replay_timeseries": str(replay_path),
            "split_index": str(split_path),
        }
    }

    with open(PROCESSED_DIR / "commercial_twin_mapping_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    split_counts = replay_sorted["split"].value_counts().to_dict()

    with open(PROCESSED_DIR / "commercial_twin_step0_3_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 0.3 CommercialTwin-to-ns-O-RAN Mapping Report\n\n")
        f.write("## Slice mapping\n\n")
        f.write("- slice_id 0 -> eMBB, based on high throughput, high buffer, and high PRB request statistics.\n")
        f.write("- slice_id 1 -> URLLC, based on low-rate traffic and APP-level delay proxy.\n\n")
        f.write("## Split protocol\n\n")
        f.write("- train: first 50% of each reservation timeline\n")
        f.write("- calibration: next 20%\n")
        f.write("- validation: next 10%\n")
        f.write("- test: final 20%\n\n")
        f.write("## Split counts\n\n")
        for k, v in split_counts.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Important caution\n\n")
        f.write("- This main CommercialTwin-calibrated protocol is eMBB+URLLC only.\n")
        f.write("- MTC/mMTC is not claimed for CommercialTwin and will be evaluated in ColO-RAN logged validation.\n")

    print("Step 0.3 completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()