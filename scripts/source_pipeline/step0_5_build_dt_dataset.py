import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin"
OUT_DIR = PROCESSED_DIR / "dt_dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "step0_5"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "commercial_twin_supervised_labels.csv"

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
    "urllc_delay_proxy_s_final",
    "urllc_delay_is_observed",
]

OPTIONAL_MISSING_FLAGS = [
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

TARGET_COLS = [
    "V_embb",
    "V_urllc",
    "V_total",
    "management_utility",
]


def robust_standardize_fit(train_df, feature_cols):
    stats = {}
    for c in feature_cols:
        vals = pd.to_numeric(train_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(vals.median())
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr < 1e-9:
            iqr = float(vals.std())
        if not np.isfinite(iqr) or iqr < 1e-9:
            iqr = 1.0
        stats[c] = {
            "median": median if np.isfinite(median) else 0.0,
            "scale": iqr,
        }
    return stats


def transform_features(df, feature_cols, stats):
    X = []
    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(stats[c]["median"])
        x = (vals - stats[c]["median"]) / stats[c]["scale"]
        x = x.clip(-20, 20)
        X.append(x.to_numpy(dtype=np.float32))
    return np.vstack(X).T.astype(np.float32)


def transform_targets(df, target_cols):
    Y = []
    for c in target_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(vals.median())
        Y.append(vals.to_numpy(dtype=np.float32))
    return np.vstack(Y).T.astype(np.float32)


def summarize_array(name, arr):
    return {
        f"{name}_shape": list(arr.shape),
        f"{name}_nan_count": int(np.isnan(arr).sum()),
        f"{name}_mean": np.nanmean(arr, axis=0).tolist() if arr.size else [],
        f"{name}_std": np.nanstd(arr, axis=0).tolist() if arr.size else [],
    }


def main():
    print(f"Reading labels: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    available_missing_flags = [c for c in OPTIONAL_MISSING_FLAGS if c in df.columns]
    final_features = available_features + available_missing_flags

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")

    print(f"Feature count: {len(final_features)}")
    print(f"Target count: {len(TARGET_COLS)}")

    split_counts = df["split"].value_counts().to_dict()
    print("Split counts:", split_counts)

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("Train split is empty.")

    scaler_stats = robust_standardize_fit(train_df, final_features)

    report = {
        "input_csv": str(INPUT_CSV),
        "num_rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "feature_names": final_features,
        "target_names": TARGET_COLS,
        "outputs": {},
    }

    for split in ["train", "calibration", "validation", "test"]:
        part = df[df["split"] == split].copy()
        if part.empty:
            print(f"[WARN] Empty split: {split}")
            continue

        X = transform_features(part, final_features, scaler_stats)
        Y = transform_targets(part, TARGET_COLS)

        x_path = OUT_DIR / f"X_{split}.npy"
        y_path = OUT_DIR / f"Y_{split}.npy"

        np.save(x_path, X)
        np.save(y_path, Y)

        report["outputs"][f"X_{split}"] = str(x_path)
        report["outputs"][f"Y_{split}"] = str(y_path)
        report.update(summarize_array(f"X_{split}", X))
        report.update(summarize_array(f"Y_{split}", Y))

    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(final_features, f, indent=2)

    with open(OUT_DIR / "target_names.json", "w", encoding="utf-8") as f:
        json.dump(TARGET_COLS, f, indent=2)

    with open(OUT_DIR / "scaler_state.json", "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    report_path = OUT_DIR / "step0_5_dt_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = RESULT_DIR / "step0_5_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Step 0.5 Digital Twin Dataset Construction Report\n\n")
        f.write(f"- Input rows: {len(df)}\n")
        f.write(f"- Feature count: {len(final_features)}\n")
        f.write(f"- Target count: {len(TARGET_COLS)}\n\n")

        f.write("## Split counts\n\n")
        for k, v in split_counts.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Features\n\n")
        for c in final_features:
            f.write(f"- {c}\n")

        f.write("\n## Targets\n\n")
        for c in TARGET_COLS:
            f.write(f"- {c}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This dataset is for the first supervised digital-twin sanity check.\n")
        f.write("- Features are robust-standardized using train split statistics only.\n")
        f.write("- Calibration, validation, and test splits are transformed using the train scaler.\n")
        f.write("- This step does not train the final CertiTwin model yet.\n")

    print("Step 0.5 completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()