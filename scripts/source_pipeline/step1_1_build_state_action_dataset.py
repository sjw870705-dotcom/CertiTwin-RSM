import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "commercial_twin_supervised_labels.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "state_action_dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "step1_1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLS = ["cluster", "slicing", "scheduling", "reservation"]

ID_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin", "split"]

TOTAL_PRB = 50.0

STATE_SOURCE_COLS = [
    "embb_buffer_bytes",
    "urllc_buffer_bytes",
    "embb_cqi",
    "urllc_cqi",
    "embb_ul_sinr",
    "urllc_ul_sinr",
    "embb_requested_prbs",
    "urllc_requested_prbs",
    "embb_throughput_mbps",
    "urllc_throughput_mbps",
    "embb_tx_error_pct",
    "urllc_tx_error_pct",
    "V_embb",
    "V_urllc",
    "V_total",
    "management_utility",
]

ACTION_COLS = [
    "candidate_embb_slice_prb",
    "candidate_urllc_slice_prb",
    "candidate_embb_share",
    "candidate_urllc_share",
    "delta_embb_prb",
    "delta_urllc_prb",
    "abs_delta_total_prb",
    "same_action_flag",
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
        median = float(vals.median()) if vals.notna().any() else 0.0
        q25 = float(vals.quantile(0.25)) if vals.notna().any() else 0.0
        q75 = float(vals.quantile(0.75)) if vals.notna().any() else 1.0
        scale = q75 - q25

        if not np.isfinite(scale) or scale < 1e-9:
            scale = float(vals.std()) if vals.notna().any() else 1.0
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0

        stats[c] = {
            "median": median if np.isfinite(median) else 0.0,
            "scale": scale,
        }
    return stats


def transform_features(df, feature_cols, stats):
    arrs = []
    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(stats[c]["median"])
        x = (vals - stats[c]["median"]) / stats[c]["scale"]
        x = x.clip(-20, 20)
        arrs.append(x.to_numpy(dtype=np.float32))
    return np.vstack(arrs).T.astype(np.float32)


def transform_targets(df, target_cols):
    arrs = []
    for c in target_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(vals.median())
        arrs.append(vals.to_numpy(dtype=np.float32))
    return np.vstack(arrs).T.astype(np.float32)


def add_prev_state_features(df):
    df = df.sort_values(GROUP_COLS + ["time_bin"]).copy()

    for c in STATE_SOURCE_COLS:
        if c in df.columns:
            df[f"{c}_prev"] = df.groupby(GROUP_COLS)[c].shift(1)

    # Previous action.
    if "embb_slice_prb" in df.columns:
        df["prev_embb_slice_prb"] = df.groupby(GROUP_COLS)["embb_slice_prb"].shift(1)
    if "urllc_slice_prb" in df.columns:
        df["prev_urllc_slice_prb"] = df.groupby(GROUP_COLS)["urllc_slice_prb"].shift(1)

    # Current logged action as candidate action.
    df["candidate_embb_slice_prb"] = pd.to_numeric(df["embb_slice_prb"], errors="coerce")
    df["candidate_urllc_slice_prb"] = pd.to_numeric(df["urllc_slice_prb"], errors="coerce")

    df["candidate_embb_share"] = df["candidate_embb_slice_prb"] / TOTAL_PRB
    df["candidate_urllc_share"] = df["candidate_urllc_slice_prb"] / TOTAL_PRB

    df["delta_embb_prb"] = df["candidate_embb_slice_prb"] - df["prev_embb_slice_prb"]
    df["delta_urllc_prb"] = df["candidate_urllc_slice_prb"] - df["prev_urllc_slice_prb"]

    df["abs_delta_total_prb"] = df["delta_embb_prb"].abs() + df["delta_urllc_prb"].abs()

    df["same_action_flag"] = (
        (df["delta_embb_prb"].abs() < 1e-9)
        & (df["delta_urllc_prb"].abs() < 1e-9)
    ).astype(int)

    return df


def build_feature_list(df):
    prev_state_features = [f"{c}_prev" for c in STATE_SOURCE_COLS if f"{c}_prev" in df.columns]

    context_features = []

    # Encode cluster/slicing/scheduling as numeric categorical codes.
    for c in ["cluster", "slicing", "scheduling"]:
        code_col = f"{c}_code"
        df[code_col] = df[c].astype("category").cat.codes
        context_features.append(code_col)

    previous_action_features = [
        "prev_embb_slice_prb",
        "prev_urllc_slice_prb",
    ]

    feature_cols = context_features + prev_state_features + previous_action_features + ACTION_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols


def summarize_dataset(df, feature_cols, target_cols):
    report = {
        "num_rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "feature_count": int(len(feature_cols)),
        "target_count": int(len(target_cols)),
        "feature_names": feature_cols,
        "target_names": target_cols,
    }

    # Missing rates.
    missing_rates = {}
    for c in feature_cols + target_cols:
        if c in df.columns:
            missing_rates[c] = float(df[c].isna().mean())
    report["missing_rates"] = missing_rates

    # Action summary.
    action_summary = {}
    for c in ["candidate_embb_slice_prb", "candidate_urllc_slice_prb", "delta_embb_prb", "delta_urllc_prb"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            action_summary[c] = {
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "p05": float(vals.quantile(0.05)),
                "p95": float(vals.quantile(0.95)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
    report["action_summary"] = action_summary

    target_summary = {}
    for c in target_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        target_summary[c] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "p05": float(vals.quantile(0.05)),
            "p95": float(vals.quantile(0.95)),
        }
    report["target_summary"] = target_summary

    return report


def main():
    print(f"Reading supervised labels: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print(f"Original rows: {len(df)}")

    needed = ["embb_slice_prb", "urllc_slice_prb", "split"] + TARGET_COLS
    missing_needed = [c for c in needed if c not in df.columns]
    if missing_needed:
        raise RuntimeError(f"Missing required columns: {missing_needed}")

    # Build state-action rows.
    df = add_prev_state_features(df)

    # Drop the first row of each reservation where previous state/action is unavailable.
    before = len(df)
    df = df.dropna(subset=["prev_embb_slice_prb", "prev_urllc_slice_prb"]).copy()
    after_prev = len(df)

    # Build feature list.
    df, feature_cols = build_feature_list(df)

    # Replace remaining feature NaN by group median then global median.
    for c in feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df.groupby(["cluster", "slicing", "scheduling"], dropna=False)[c].transform("median"))
            df[c] = df[c].fillna(df[c].median())
            df[c] = df[c].fillna(0.0)

    # Drop rows with missing targets.
    before_target = len(df)
    df = df.dropna(subset=TARGET_COLS).copy()
    after_target = len(df)

    # Save full sample table.
    sample_cols = ID_COLS + feature_cols + TARGET_COLS
    sample_cols = [c for c in sample_cols if c in df.columns]

    samples_path = OUT_DIR / "state_action_samples.csv"
    df[sample_cols].to_csv(samples_path, index=False, encoding="utf-8-sig")

    # Fit scaler on train only.
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise RuntimeError("Train split is empty after state-action construction.")

    scaler_stats = robust_standardize_fit(train_df, feature_cols)

    # Generate npy files.
    outputs = {
        "state_action_samples": str(samples_path),
    }

    for split in ["train", "calibration", "validation", "test"]:
        part = df[df["split"] == split].copy()
        if part.empty:
            print(f"[WARN] split is empty: {split}")
            continue

        X = transform_features(part, feature_cols, scaler_stats)
        Y = transform_targets(part, TARGET_COLS)

        x_path = OUT_DIR / f"X_{split}.npy"
        y_path = OUT_DIR / f"Y_{split}.npy"

        np.save(x_path, X)
        np.save(y_path, Y)

        outputs[f"X_{split}"] = str(x_path)
        outputs[f"Y_{split}"] = str(y_path)

    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(OUT_DIR / "target_names.json", "w", encoding="utf-8") as f:
        json.dump(TARGET_COLS, f, indent=2)

    with open(OUT_DIR / "scaler_state.json", "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    report = summarize_dataset(df, feature_cols, TARGET_COLS)
    report["rows_before_prev_drop"] = int(before)
    report["rows_after_prev_drop"] = int(after_prev)
    report["rows_before_target_drop"] = int(before_target)
    report["rows_after_target_drop"] = int(after_target)
    report["outputs"] = outputs
    report["important_note"] = (
        "This is the first formal state-action dataset. "
        "The state is built from previous-step features and the action is the current logged PRB allocation. "
        "Next steps will extend candidate actions beyond logged actions."
    )

    with open(OUT_DIR / "step1_1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(RESULT_DIR / "step1_1_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 1.1 State-Action Dataset Construction Report\n\n")
        f.write(f"- Original rows: {before}\n")
        f.write(f"- Rows after previous-state drop: {after_prev}\n")
        f.write(f"- Rows after target drop: {after_target}\n")
        f.write(f"- Feature count: {len(feature_cols)}\n")
        f.write(f"- Target count: {len(TARGET_COLS)}\n\n")

        f.write("## Split counts\n\n")
        for k, v in report["split_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Feature groups\n\n")
        f.write("### Features\n")
        for c in feature_cols:
            f.write(f"- {c}\n")

        f.write("\n### Targets\n")
        for c in TARGET_COLS:
            f.write(f"- {c}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- State features are previous-step features.\n")
        f.write("- Candidate action features are current logged PRB allocation features.\n")
        f.write("- This avoids directly using current post-action KPI as the input state.\n")
        f.write("- Later steps will add generated candidate actions for PF/Greedy/DT/PPO/SAC shielding.\n")

    print("Step 1.1 completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()