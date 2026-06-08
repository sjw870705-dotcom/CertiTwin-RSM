import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "commercial_twin"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_2b"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BS_FILES_CSV = INTERIM_DIR / "step0_2_bs_metrics_files.csv"

KEY_FIELDS = [
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


def safe_read_columns(path, usecols=None):
    try:
        if usecols is None:
            return pd.read_csv(path)
        return pd.read_csv(path, usecols=lambda c: c in usecols)
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}")
        return None


def numeric_summary(df, group_cols, value_cols):
    records = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec["num_rows"] = len(g)
        rec["num_files"] = g["source_file"].nunique() if "source_file" in g.columns else None

        for col in value_cols:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                rec[f"{col}_mean"] = vals.mean()
                rec[f"{col}_median"] = vals.median()
                rec[f"{col}_p95"] = vals.quantile(0.95)
                rec[f"{col}_non_null"] = int(vals.notna().sum())
        records.append(rec)
    return pd.DataFrame(records)


def main():
    bs_files = pd.read_csv(BS_FILES_CSV)
    print(f"Found BS metrics files: {len(bs_files)}")

    all_parts = []
    file_level_records = []

    for _, row in tqdm(bs_files.iterrows(), total=len(bs_files)):
        path = Path(row["path"])
        df = safe_read_columns(path, usecols=KEY_FIELDS)
        if df is None or "slice_id" not in df.columns:
            continue

        keep = [c for c in KEY_FIELDS if c in df.columns]
        tmp = df[keep].copy()
        tmp["source_file"] = str(path)
        tmp["cluster"] = row.get("cluster", "")
        tmp["slicing"] = row.get("slicing", "")
        tmp["scheduling"] = row.get("scheduling", "")
        tmp["reservation"] = row.get("reservation", "")

        all_parts.append(tmp)

        file_level_records.append({
            "source_file": str(path),
            "cluster": row.get("cluster", ""),
            "slicing": row.get("slicing", ""),
            "scheduling": row.get("scheduling", ""),
            "reservation": row.get("reservation", ""),
            "num_rows": len(df),
            "slice_ids": "|".join(map(str, sorted(pd.Series(df["slice_id"]).dropna().unique().tolist()))),
            "num_slice_ids": pd.Series(df["slice_id"]).dropna().nunique(),
        })

    if not all_parts:
        raise RuntimeError("No readable BS metrics with slice_id found.")

    full = pd.concat(all_parts, ignore_index=True)
    full.to_csv(INTERIM_DIR / "step0_2b_all_bs_slice_rows_sampled_columns.csv",
                index=False, encoding="utf-8-sig")

    file_level = pd.DataFrame(file_level_records)
    file_level.to_csv(INTERIM_DIR / "step0_2b_file_level_slice_ids.csv",
                      index=False, encoding="utf-8-sig")

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

    slice_summary = numeric_summary(
        full,
        group_cols=["slice_id"],
        value_cols=value_cols
    )
    slice_summary.to_csv(INTERIM_DIR / "step0_2b_slice_id_global_summary.csv",
                         index=False, encoding="utf-8-sig")

    combo_summary = numeric_summary(
        full,
        group_cols=["cluster", "slicing", "scheduling", "slice_id"],
        value_cols=value_cols
    )
    combo_summary.to_csv(INTERIM_DIR / "step0_2b_slice_id_by_combo_summary.csv",
                         index=False, encoding="utf-8-sig")

    # Count how many files have one or more slice ids.
    file_count_summary = (
        file_level.groupby(["cluster", "slicing", "scheduling", "slice_ids"], dropna=False)
        .size()
        .reset_index(name="num_files")
    )
    file_count_summary.to_csv(INTERIM_DIR / "step0_2b_file_count_by_slice_ids.csv",
                              index=False, encoding="utf-8-sig")

    summary = {
        "num_bs_files_scanned": int(len(bs_files)),
        "num_readable_files_with_slice_id": int(len(file_level)),
        "num_rows_loaded": int(len(full)),
        "global_slice_ids": sorted(map(str, full["slice_id"].dropna().unique().tolist())),
        "num_unique_slice_ids": int(full["slice_id"].dropna().nunique()),
        "outputs": {
            "slice_global_summary": str(INTERIM_DIR / "step0_2b_slice_id_global_summary.csv"),
            "slice_by_combo_summary": str(INTERIM_DIR / "step0_2b_slice_id_by_combo_summary.csv"),
            "file_level_slice_ids": str(INTERIM_DIR / "step0_2b_file_level_slice_ids.csv"),
            "file_count_by_slice_ids": str(INTERIM_DIR / "step0_2b_file_count_by_slice_ids.csv"),
        }
    }

    with open(INTERIM_DIR / "step0_2b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(RESULT_DIR / "step0_2b_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 0.2B CommercialTwin Slice Coverage Report\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            if k != "outputs":
                f.write(f"- {k}: {v}\n")
        f.write("\n## Interpretation guide\n\n")
        f.write("- If global_slice_ids contains both 0 and 1, CommercialTwin supports two-slice eMBB/URLLC analysis.\n")
        f.write("- If only one slice_id appears, check whether the other slice is represented in APP logs or separate UE files.\n")
        f.write("- Use slice_by_combo_summary to inspect whether slicing_1~slicing_5 change PRB allocation.\n")

    print("Step 0.2B completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()