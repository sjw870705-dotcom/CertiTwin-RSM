import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_SAMPLES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_dataset"
    / "state_action_samples.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step1_1b"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PRB = 50.0
SEED = 20260604

MAX_BASE_ROWS_PER_SPLIT = {
    "train": 300_000,
    "calibration": 150_000,
    "validation": 100_000,
    "test": 150_000,
}

ID_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin", "split"]

STATE_FEATURE_COLS = [
    "cluster_code",
    "slicing_code",
    "scheduling_code",
    "embb_buffer_bytes_prev",
    "urllc_buffer_bytes_prev",
    "embb_cqi_prev",
    "urllc_cqi_prev",
    "embb_ul_sinr_prev",
    "urllc_ul_sinr_prev",
    "embb_requested_prbs_prev",
    "urllc_requested_prbs_prev",
    "embb_throughput_mbps_prev",
    "urllc_throughput_mbps_prev",
    "embb_tx_error_pct_prev",
    "urllc_tx_error_pct_prev",
    "V_embb_prev",
    "V_urllc_prev",
    "V_total_prev",
    "management_utility_prev",
    "prev_embb_slice_prb",
    "prev_urllc_slice_prb",
]

ACTION_FEATURE_COLS = [
    "candidate_embb_slice_prb",
    "candidate_urllc_slice_prb",
    "candidate_embb_share",
    "candidate_urllc_share",
    "delta_embb_prb",
    "delta_urllc_prb",
    "abs_delta_total_prb",
    "same_action_flag",
    "candidate_source_code",
    "matched_action_distance",
]

FEATURE_COLS = STATE_FEATURE_COLS + ACTION_FEATURE_COLS

TARGET_COLS = [
    "V_embb",
    "V_urllc",
    "V_total",
    "management_utility",
]


def action_key(e, u):
    return f"{round(float(e), 6)}_{round(float(u), 6)}"


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


def extract_observed_actions(df):
    actions = (
        df[["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    actions["candidate_embb_slice_prb"] = actions["candidate_embb_slice_prb"].astype(float)
    actions["candidate_urllc_slice_prb"] = actions["candidate_urllc_slice_prb"].astype(float)
    actions["total_prb"] = actions["candidate_embb_slice_prb"] + actions["candidate_urllc_slice_prb"]

    actions = actions[
        (actions["total_prb"] > 0)
        & (actions["total_prb"] <= TOTAL_PRB + 1e-6)
    ].copy()

    actions = actions.sort_values(
        ["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]
    ).reset_index(drop=True)

    actions["action_key"] = actions.apply(
        lambda r: action_key(r["candidate_embb_slice_prb"], r["candidate_urllc_slice_prb"]),
        axis=1
    )

    actions["candidate_source_code"] = np.arange(len(actions))

    return actions


def build_label_tables(df):
    tmp = df.copy()
    tmp["action_key"] = tmp.apply(
        lambda r: action_key(r["candidate_embb_slice_prb"], r["candidate_urllc_slice_prb"]),
        axis=1
    )

    context_cols = ["cluster_code", "slicing_code", "scheduling_code", "action_key"]

    context_label = (
        tmp[context_cols + TARGET_COLS]
        .groupby(context_cols, dropna=False)
        .mean()
        .reset_index()
    )

    context_label = context_label.rename(
        columns={c: f"label_{c}" for c in TARGET_COLS}
    )

    global_label = (
        tmp[["action_key"] + TARGET_COLS]
        .groupby(["action_key"], dropna=False)
        .mean()
        .reset_index()
    )

    global_label = global_label.rename(
        columns={c: f"global_{c}" for c in TARGET_COLS}
    )

    return context_label, global_label


def sample_base_rows(df):
    parts = []

    for split, max_n in MAX_BASE_ROWS_PER_SPLIT.items():
        part = df[df["split"] == split].copy()
        if len(part) > max_n:
            part = part.sample(n=max_n, random_state=SEED)
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def expand_with_all_observed_actions(base_df, actions):
    base_df = base_df.reset_index(drop=True)
    actions = actions.reset_index(drop=True)

    n_base = len(base_df)
    n_actions = len(actions)

    print(f"Expanding base rows {n_base} with {n_actions} observed actions...")

    repeated_base = base_df.loc[base_df.index.repeat(n_actions)].reset_index(drop=True)
    tiled_actions = pd.concat([actions] * n_base, ignore_index=True)

    expanded = repeated_base.copy()

    expanded["candidate_embb_slice_prb"] = tiled_actions["candidate_embb_slice_prb"].to_numpy()
    expanded["candidate_urllc_slice_prb"] = tiled_actions["candidate_urllc_slice_prb"].to_numpy()
    expanded["candidate_embb_share"] = expanded["candidate_embb_slice_prb"] / TOTAL_PRB
    expanded["candidate_urllc_share"] = expanded["candidate_urllc_slice_prb"] / TOTAL_PRB

    expanded["delta_embb_prb"] = expanded["candidate_embb_slice_prb"] - expanded["prev_embb_slice_prb"]
    expanded["delta_urllc_prb"] = expanded["candidate_urllc_slice_prb"] - expanded["prev_urllc_slice_prb"]
    expanded["abs_delta_total_prb"] = expanded["delta_embb_prb"].abs() + expanded["delta_urllc_prb"].abs()

    expanded["same_action_flag"] = (
        (expanded["delta_embb_prb"].abs() < 1e-9)
        & (expanded["delta_urllc_prb"].abs() < 1e-9)
    ).astype(int)

    expanded["candidate_source_code"] = tiled_actions["candidate_source_code"].to_numpy()
    expanded["action_key"] = tiled_actions["action_key"].to_numpy()

    return expanded


def attach_pseudo_labels(expanded, context_label, global_label):
    merged = expanded.merge(
        context_label,
        on=["cluster_code", "slicing_code", "scheduling_code", "action_key"],
        how="left"
    )

    merged = merged.merge(
        global_label,
        on="action_key",
        how="left"
    )

    # Fill context label with global action label.
    for c in TARGET_COLS:
        merged[c] = merged[f"label_{c}"].fillna(merged[f"global_{c}"])

    # matched distance is 0 because candidate actions are observed action pairs.
    merged["matched_action_distance"] = 0.0

    # Drop helper label columns.
    drop_cols = [f"label_{c}" for c in TARGET_COLS] + [f"global_{c}" for c in TARGET_COLS]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged


def main():
    print(f"Reading Step 1.1 samples: {INPUT_SAMPLES}")
    df = pd.read_csv(INPUT_SAMPLES)
    print(f"Rows loaded: {len(df)}")

    actions = extract_observed_actions(df)
    print(f"Observed action pairs: {len(actions)}")
    print(actions[["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]].to_string(index=False))

    actions_path = OUT_DIR / "observed_action_pairs.csv"
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    context_label, global_label = build_label_tables(df)

    base_df = sample_base_rows(df)
    print("Base split counts:")
    print(base_df["split"].value_counts())

    expanded = expand_with_all_observed_actions(base_df, actions)
    expanded = attach_pseudo_labels(expanded, context_label, global_label)

    # Ensure no missing features/targets.
    for c in FEATURE_COLS:
        if expanded[c].isna().any():
            expanded[c] = expanded[c].fillna(expanded[c].median())
            expanded[c] = expanded[c].fillna(0.0)

    for c in TARGET_COLS:
        if expanded[c].isna().any():
            expanded[c] = expanded[c].fillna(expanded[c].median())

    save_cols = ID_COLS + FEATURE_COLS + TARGET_COLS + ["action_key"]
    save_cols = [c for c in save_cols if c in expanded.columns]

    expanded_path = OUT_DIR / "candidate_expanded_samples.csv"
    expanded[save_cols].to_csv(expanded_path, index=False, encoding="utf-8-sig")

    train_df = expanded[expanded["split"] == "train"].copy()
    scaler_stats = robust_standardize_fit(train_df, FEATURE_COLS)

    outputs = {
        "candidate_expanded_samples": str(expanded_path),
        "observed_action_pairs": str(actions_path),
    }

    for split in ["train", "calibration", "validation", "test"]:
        part = expanded[expanded["split"] == split].copy()
        if part.empty:
            continue

        X = transform_features(part, FEATURE_COLS, scaler_stats)
        Y = transform_targets(part, TARGET_COLS)

        x_path = OUT_DIR / f"X_{split}.npy"
        y_path = OUT_DIR / f"Y_{split}.npy"

        np.save(x_path, X)
        np.save(y_path, Y)

        outputs[f"X_{split}"] = str(x_path)
        outputs[f"Y_{split}"] = str(y_path)

    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(OUT_DIR / "target_names.json", "w", encoding="utf-8") as f:
        json.dump(TARGET_COLS, f, indent=2)

    with open(OUT_DIR / "scaler_state.json", "w", encoding="utf-8") as f:
        json.dump(scaler_stats, f, indent=2)

    action_summary = {}
    for c in [
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
        "delta_embb_prb",
        "delta_urllc_prb",
        "abs_delta_total_prb",
        "same_action_flag",
        "matched_action_distance",
    ]:
        vals = pd.to_numeric(expanded[c], errors="coerce")
        action_summary[c] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p05": float(vals.quantile(0.05)),
            "p95": float(vals.quantile(0.95)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    report = {
        "input_samples": str(INPUT_SAMPLES),
        "base_rows_used": int(len(base_df)),
        "num_expanded_rows": int(len(expanded)),
        "observed_action_pair_count": int(len(actions)),
        "candidate_actions_per_state": int(len(actions)),
        "split_counts": {str(k): int(v) for k, v in expanded["split"].value_counts().to_dict().items()},
        "feature_count": int(len(FEATURE_COLS)),
        "target_count": int(len(TARGET_COLS)),
        "feature_names": FEATURE_COLS,
        "target_names": TARGET_COLS,
        "action_summary": action_summary,
        "outputs": outputs,
        "important_note": (
            "Fast Step 1.1B expands each sampled state with all observed slicing actions. "
            "This avoids slow per-row matching and creates a candidate-aware state-action dataset. "
            "Labels are pseudo labels from context/action averages, with global action averages as fallback."
        )
    }

    report_json = OUT_DIR / "step1_1b_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md = RESULT_DIR / "step1_1b_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Step 1.1B Candidate-Expanded State-Action Dataset Report\n\n")
        f.write(f"- Base rows used: {len(base_df)}\n")
        f.write(f"- Expanded rows: {len(expanded)}\n")
        f.write(f"- Observed action pairs: {len(actions)}\n")
        f.write(f"- Candidate actions per state: {len(actions)}\n")
        f.write(f"- Feature count: {len(FEATURE_COLS)}\n")
        f.write(f"- Target count: {len(TARGET_COLS)}\n\n")

        f.write("## Split counts\n\n")
        for k, v in report["split_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This dataset expands each state with all observed slicing actions.\n")
        f.write("- Candidate labels are pseudo labels based on context/action averages.\n")
        f.write("- This is suitable for training the first candidate-aware digital twin.\n")
        f.write("- Final main experiments will still require raw controller actions and paired shield evaluation.\n")

    print("Step 1.1B fast completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()