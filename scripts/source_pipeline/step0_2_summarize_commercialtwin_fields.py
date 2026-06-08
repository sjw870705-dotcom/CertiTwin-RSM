import re
import json
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FILE_INV = INTERIM_DIR / "file_inventory.csv"


KEY_BS_FIELDS = [
    "Timestamp",
    "num_ues",
    "IMSI",
    "RNTI",
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "scheduling_policy",
    "dl_mcs",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "tx_errors downlink (%)",
    "dl_cqi",
    "ul_mcs",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "rx_pkts uplink",
    "rx_errors uplink (%)",
    "ul_rssi",
    "ul_sinr",
    "sum_requested_prbs",
    "sum_granted_prbs",
]

APP_TIME_FIELDS = [
    "Timestamp_recv",
    "Timestamp_send",
    "Received Time",
    "Sent Time",
    "Flow ID",
    "Sequence No.",
    "Protocol Payload Size",
]


def extract_path_tags(path_str: str) -> dict:
    parts = re.split(r"[\\/]+", str(path_str))
    out = {
        "cluster": "",
        "slicing": "",
        "scheduling": "",
        "reservation": "",
        "level": "",
    }
    for p in parts:
        if p.startswith("cluster_"):
            out["cluster"] = p
        elif p.startswith("slicing_"):
            out["slicing"] = p
        elif p.startswith("scheduling_"):
            out["scheduling"] = p
        elif p.startswith("RESERVATION-"):
            out["reservation"] = p
        elif p in {"bs", "ue", "gnb", "user"}:
            out["level"] = p
    return out


def safe_read_csv(path, nrows=5000):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


def main():
    inv = pd.read_csv(FILE_INV)
    inv["columns_str"] = inv["columns"].fillna("").astype(str)

    # Identify BS metrics files.
    bs_metrics = inv[
        inv["columns_str"].str.contains("slice_id", regex=False)
        & inv["columns_str"].str.contains("tx_brate downlink", regex=False)
        & inv["columns_str"].str.contains("slice_prb", regex=False)
    ].copy()

    # Identify APP-level files with send/receive timestamps.
    app_logs = inv[
        inv["columns_str"].str.contains("Timestamp_recv", regex=False)
        | inv["columns_str"].str.contains("Received Time", regex=False)
    ].copy()

    # Identify compact aggregate files.
    aggregate_files = inv[
        inv["columns_str"].str.contains("nof_ue", regex=False)
        & inv["columns_str"].str.contains("dl_brate", regex=False)
    ].copy()

    # Add path tags.
    for df in [bs_metrics, app_logs, aggregate_files]:
        tags = df["path"].apply(extract_path_tags).apply(pd.Series)
        for c in tags.columns:
            df[c] = tags[c]

    bs_metrics.to_csv(INTERIM_DIR / "step0_2_bs_metrics_files.csv", index=False, encoding="utf-8-sig")
    app_logs.to_csv(INTERIM_DIR / "step0_2_app_log_files.csv", index=False, encoding="utf-8-sig")
    aggregate_files.to_csv(INTERIM_DIR / "step0_2_aggregate_files.csv", index=False, encoding="utf-8-sig")

    # Representative files: one per cluster/slicing/scheduling when possible.
    rep_bs = (
        bs_metrics.sort_values(["cluster", "slicing", "scheduling", "reservation", "filename"])
        .groupby(["cluster", "slicing", "scheduling"], dropna=False)
        .head(1)
        .copy()
    )

    rep_app = (
        app_logs.sort_values(["cluster", "slicing", "scheduling", "reservation", "filename"])
        .groupby(["cluster", "slicing", "scheduling"], dropna=False)
        .head(1)
        .copy()
    )

    rep_bs.to_csv(INTERIM_DIR / "step0_2_representative_bs_files.csv", index=False, encoding="utf-8-sig")
    rep_app.to_csv(INTERIM_DIR / "step0_2_representative_app_files.csv", index=False, encoding="utf-8-sig")

    # Read a few representative BS files and sample key fields.
    bs_samples = []
    for _, row in rep_bs.head(20).iterrows():
        path = Path(row["path"])
        df = safe_read_csv(path, nrows=2000)
        if df is None:
            continue
        keep = [c for c in KEY_BS_FIELDS if c in df.columns]
        tmp = df[keep].copy()
        tmp["source_file"] = str(path)
        tmp["cluster"] = row["cluster"]
        tmp["slicing"] = row["slicing"]
        tmp["scheduling"] = row["scheduling"]
        tmp["reservation"] = row["reservation"]
        bs_samples.append(tmp)

    if bs_samples:
        bs_sample_df = pd.concat(bs_samples, ignore_index=True)
        bs_sample_df.to_csv(INTERIM_DIR / "step0_2_bs_metrics_sample.csv",
                            index=False, encoding="utf-8-sig")

        # Slice ID summary.
        slice_summary = []
        if "slice_id" in bs_sample_df.columns:
            for sid, g in bs_sample_df.groupby("slice_id", dropna=False):
                rec = {
                    "slice_id": sid,
                    "num_rows": len(g),
                }
                for field in ["tx_brate downlink [Mbps]", "dl_buffer [bytes]", "slice_prb",
                              "sum_requested_prbs", "sum_granted_prbs", "tx_errors downlink (%)"]:
                    if field in g.columns:
                        vals = pd.to_numeric(g[field], errors="coerce")
                        rec[f"{field}_mean"] = vals.mean()
                        rec[f"{field}_median"] = vals.median()
                slice_summary.append(rec)

            pd.DataFrame(slice_summary).to_csv(
                INTERIM_DIR / "step0_2_slice_id_summary.csv",
                index=False,
                encoding="utf-8-sig"
            )

    # Read representative APP files and try to compute delay.
    app_samples = []
    for _, row in rep_app.head(20).iterrows():
        path = Path(row["path"])
        df = safe_read_csv(path, nrows=2000)
        if df is None:
            continue

        keep = [c for c in APP_TIME_FIELDS if c in df.columns]
        tmp = df[keep].copy()

        # Try delay calculation.
        if "Timestamp_recv" in tmp.columns and "Timestamp_send" in tmp.columns:
            recv = pd.to_numeric(tmp["Timestamp_recv"], errors="coerce")
            send = pd.to_numeric(tmp["Timestamp_send"], errors="coerce")
            tmp["delay_timestamp_recv_minus_send"] = recv - send

        if "Received Time" in tmp.columns and "Sent Time" in tmp.columns:
            recv = pd.to_numeric(tmp["Received Time"], errors="coerce")
            send = pd.to_numeric(tmp["Sent Time"], errors="coerce")
            tmp["delay_received_minus_sent"] = recv - send

        tmp["source_file"] = str(path)
        tmp["cluster"] = row["cluster"]
        tmp["slicing"] = row["slicing"]
        tmp["scheduling"] = row["scheduling"]
        tmp["reservation"] = row["reservation"]
        app_samples.append(tmp)

    if app_samples:
        app_sample_df = pd.concat(app_samples, ignore_index=True)
        app_sample_df.to_csv(INTERIM_DIR / "step0_2_app_delay_sample.csv",
                             index=False, encoding="utf-8-sig")

        delay_cols = [c for c in app_sample_df.columns if c.startswith("delay_")]
        delay_summary = []
        for c in delay_cols:
            vals = pd.to_numeric(app_sample_df[c], errors="coerce")
            delay_summary.append({
                "delay_column": c,
                "count": int(vals.notna().sum()),
                "min": vals.min(),
                "median": vals.median(),
                "mean": vals.mean(),
                "p95": vals.quantile(0.95),
                "max": vals.max(),
            })
        pd.DataFrame(delay_summary).to_csv(
            INTERIM_DIR / "step0_2_delay_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # Schema summary.
    schema_summary = {
        "num_bs_metrics_files": int(len(bs_metrics)),
        "num_app_log_files": int(len(app_logs)),
        "num_aggregate_files": int(len(aggregate_files)),
        "clusters": sorted(bs_metrics["cluster"].dropna().unique().tolist()),
        "slicing_configs": sorted(bs_metrics["slicing"].dropna().unique().tolist()),
        "scheduling_configs": sorted(bs_metrics["scheduling"].dropna().unique().tolist()),
        "outputs": {
            "bs_metrics_files": str(INTERIM_DIR / "step0_2_bs_metrics_files.csv"),
            "app_log_files": str(INTERIM_DIR / "step0_2_app_log_files.csv"),
            "representative_bs_files": str(INTERIM_DIR / "step0_2_representative_bs_files.csv"),
            "representative_app_files": str(INTERIM_DIR / "step0_2_representative_app_files.csv"),
            "bs_metrics_sample": str(INTERIM_DIR / "step0_2_bs_metrics_sample.csv"),
            "app_delay_sample": str(INTERIM_DIR / "step0_2_app_delay_sample.csv"),
            "slice_id_summary": str(INTERIM_DIR / "step0_2_slice_id_summary.csv"),
            "delay_summary": str(INTERIM_DIR / "step0_2_delay_summary.csv"),
        }
    }

    with open(INTERIM_DIR / "step0_2_summary.json", "w", encoding="utf-8") as f:
        json.dump(schema_summary, f, indent=2)

    with open(RESULT_DIR / "step0_2_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 0.2 CommercialTwin Field Summary Report\n\n")
        f.write("## Summary\n\n")
        for k, v in schema_summary.items():
            if k != "outputs":
                f.write(f"- {k}: {v}\n")
        f.write("\n## Interpretation\n\n")
        f.write("- BS metrics files should support slice-level throughput, PRB, scheduling, and radio KPI extraction.\n")
        f.write("- APP logs should support delay proxy calculation from send/receive timestamps.\n")
        f.write("- Main experiments should remain eMBB + URLLC only for CommercialTwin-calibrated ns-O-RAN.\n")
        f.write("- Three-slice eMBB/URLLC/MTC validation should be performed later on ColO-RAN logged data.\n")

    print("Step 0.2 summary completed.")
    print(json.dumps(schema_summary, indent=2))


if __name__ == "__main__":
    main()