import os
import json
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml


TEXT_EXTS = {".txt", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".md"}
TABLE_EXTS = {".csv", ".tsv", ".json", ".parquet", ".xlsx"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz"}


FIELD_HINTS = {
    "time": ["time", "timestamp", "ts", "datetime", "frame", "slot", "tti"],
    "slice": ["slice", "slice_id", "sliceid", "slice_type", "sst"],
    "ue": ["ue", "ue_id", "rnti", "imsi", "user"],
    "bs": ["bs", "gnb", "enb", "cell", "nodeb"],
    "prb": ["prb", "rb", "rbg", "resource", "allocation"],
    "traffic": ["traffic", "throughput", "tx_brate", "rx_brate", "bitrate", "bytes", "volume", "load"],
    "latency": ["latency", "delay", "rtt", "jitter"],
    "reliability": ["loss", "drop", "error", "bler", "success", "retrans"],
    "scheduler": ["scheduler", "sched", "pf", "rr", "policy"],
    "kpi": ["kpi", "kpm", "metric", "throughput", "delay", "latency", "prb"],
}


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_fields(columns):
    records = []
    for col in columns:
        col_l = str(col).lower()
        matched = []
        for category, hints in FIELD_HINTS.items():
            if any(h in col_l for h in hints):
                matched.append(category)
        records.append({
            "field": col,
            "matched_categories": ";".join(matched) if matched else "",
        })
    return records


def safe_read_table(path: Path, max_rows: int):
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=max_rows)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=max_rows)
        if suffix == ".json":
            try:
                return pd.read_json(path, lines=True, nrows=max_rows)
            except TypeError:
                return pd.read_json(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".xlsx":
            return pd.read_excel(path, nrows=max_rows)
    except Exception as e:
        return f"READ_ERROR: {e}"
    return None


def inspect_regular_file(path: Path, max_rows: int):
    stat = path.stat()
    item = {
        "path": str(path),
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "size_mb": round(stat.st_size / 1024 / 1024, 4),
        "kind": "regular",
        "readable_table": False,
        "n_columns": None,
        "columns": "",
        "error": "",
    }

    if path.suffix.lower() in TABLE_EXTS:
        df = safe_read_table(path, max_rows=max_rows)
        if isinstance(df, pd.DataFrame):
            item["readable_table"] = True
            item["n_columns"] = len(df.columns)
            item["columns"] = "|".join(map(str, df.columns))
            item["n_sample_rows"] = len(df)
        elif isinstance(df, str):
            item["error"] = df

    return item


def list_archive_members(path: Path, max_files_inside: int):
    members = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist()[:max_files_inside]:
                    members.append(name)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as tf:
                for m in tf.getmembers()[:max_files_inside]:
                    members.append(m.name)
    except Exception as e:
        members.append(f"ARCHIVE_READ_ERROR: {e}")
    return members


def main():
    config_path = Path("configs/commercial_twin_paths.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "Cannot find configs/commercial_twin_paths.yaml. "
            "Run this script from the repository root."
        )

    cfg = load_config(config_path)
    raw_dir = Path(cfg["commercial_twin"]["raw_dir"])
    download_dir = Path(cfg["commercial_twin"]["download_dir"])
    interim_dir = Path(cfg["commercial_twin"]["interim_dir"])
    report_dir = Path(cfg["commercial_twin"]["report_dir"])

    max_rows = int(cfg["inspection"]["max_rows_per_file"])
    scan_archives = bool(cfg["inspection"]["scan_archives"])
    max_files_inside_archive = int(cfg["inspection"]["max_files_inside_archive"])

    interim_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scan_roots = [raw_dir, download_dir]
    all_files = []
    for root in scan_roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    all_files.append(p)

    inventory = []
    archive_records = []

    for p in sorted(set(all_files)):
        item = inspect_regular_file(p, max_rows=max_rows)
        inventory.append(item)

        if scan_archives and (
            p.suffix.lower() in {".zip", ".tar", ".tgz", ".gz"} or
            p.name.lower().endswith(".tar.gz")
        ):
            members = list_archive_members(p, max_files_inside=max_files_inside_archive)
            for idx, member in enumerate(members):
                archive_records.append({
                    "archive_path": str(p),
                    "member_index": idx,
                    "member_name": member,
                })

    inv_df = pd.DataFrame(inventory)
    inv_path = interim_dir / "file_inventory.csv"
    inv_df.to_csv(inv_path, index=False, encoding="utf-8-sig")

    if archive_records:
        archive_df = pd.DataFrame(archive_records)
        archive_df.to_csv(interim_dir / "archive_member_inventory.csv",
                          index=False, encoding="utf-8-sig")

    field_rows = []
    for _, row in inv_df.iterrows():
        if row.get("readable_table") is True or str(row.get("readable_table")).lower() == "true":
            columns = str(row.get("columns", "")).split("|")
            for rec in classify_fields(columns):
                rec.update({
                    "source_file": row["path"],
                    "filename": row["filename"],
                })
                field_rows.append(rec)

    field_df = pd.DataFrame(field_rows)
    field_path = interim_dir / "field_inventory.csv"
    field_df.to_csv(field_path, index=False, encoding="utf-8-sig")

    # Build candidate mapping table for manual review.
    mapping_rows = []
    if not field_df.empty:
        for category in FIELD_HINTS.keys():
            sub = field_df[field_df["matched_categories"].str.contains(category, na=False)]
            for _, r in sub.iterrows():
                mapping_rows.append({
                    "target_concept": category,
                    "candidate_field": r["field"],
                    "source_file": r["source_file"],
                    "confidence": "keyword_match",
                    "manual_decision": "",
                    "notes": "",
                })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = interim_dir / "candidate_field_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    summary = {
        "num_files_scanned": int(len(inv_df)),
        "num_readable_tables": int(inv_df["readable_table"].sum()) if "readable_table" in inv_df else 0,
        "num_fields_detected": int(len(field_df)),
        "num_candidate_mappings": int(len(mapping_df)),
        "outputs": {
            "file_inventory": str(inv_path),
            "field_inventory": str(field_path),
            "candidate_field_mapping": str(mapping_path),
        }
    }

    summary_path = interim_dir / "step0_1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    report_path = report_dir / "step0_1_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.1 CommercialTwin-RAN Field Inspection Report\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            if k != "outputs":
                f.write(f"- {k}: {v}\n")
        f.write("\n## Output files\n\n")
        for k, v in summary["outputs"].items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Next manual checks\n\n")
        f.write("1. Open `field_inventory.csv` and check whether slice/time/PRB/KPI fields exist.\n")
        f.write("2. Open `candidate_field_mapping.csv` and mark useful fields in `manual_decision`.\n")
        f.write("3. Confirm whether the data supports eMBB + URLLC trace replay.\n")
        f.write("4. Do not assume mMTC is available in CommercialTwin; use ColO-RAN for three-slice validation.\n")

    print("Step 0.1 inspection completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()