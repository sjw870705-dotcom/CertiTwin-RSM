import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_SAMPLES = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "state_action_dataset" / "state_action_samples.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "commercial_twin" / "state_action_candidate_dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "step1_1b"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_PRB = 50.0
SEED = 20260604

# To control file size. You can increase later.
MAX_BASE_ROWS_PER_SPLIT = {
    "train": 300_000,
    "calibration": 150_000,
    "validation": 100_000,
    "test": 150_000,
}

CANDIDATES_PER_STATE = 8

ID_COLS = ["cluster", "slicing", "scheduling", "reservation", "time_bin", "split"]

TARGET_COLS = ["V_embb", "V_urllc", "V_total", "management_utility"]

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

FINAL_FEATURE_COLS = STATE_FEATURE_COLS + ACTION_FEATURE_COLS


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

    # Keep feasible-ish actions.
    actions = actions[(actions["total_prb"] > 0) & (actions["total_prb"] <= TOTAL_PRB + 1e-6)].copy()
    actions = actions.sort_values(["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]).reset_index(drop=True)

    return actions


def choose_candidate_actions(row, observed_actions, rng):
    prev_e = float(row["prev_embb_slice_prb"])
    prev_u = float(row["prev_urllc_slice_prb"])
    logged_e = float(row["candidate_embb_slice_prb"])
    logged_u = float(row["candidate_urllc_slice_prb"])

    actions = []

    def add(e, u, source):
        e = float(np.clip(e, 0, TOTAL_PRB))
        u = float(np.clip(u, 0, TOTAL_PRB))
        if e + u > TOTAL_PRB:
            # Normalize down while preserving ratio.
            total = e + u
            e = e / total * TOTAL_PRB
            u = u / total * TOTAL_PRB
        actions.append((round(e, 6), round(u, 6), source))

    # 0. logged action.
    add(logged_e, logged_u, "logged")

    # 1. previous action.
    add(prev_e, prev_u, "previous")

    # 2. local perturbations.
    for de in [-5, 5]:
        add(logged_e + de, logged_u - de, "local_perturb")

    # 3. high-eMBB and high-URLLC observed actions.
    obs = observed_actions
    if len(obs) > 0:
        high_e = obs.iloc[obs["candidate_embb_slice_prb"].idxmax()]
        high_u = obs.iloc[obs["candidate_urllc_slice_prb"].idxmax()]
        balanced_idx = ((obs["candidate_embb_slice_prb"] - obs["candidate_urllc_slice_prb"]).abs()).idxmin()
        balanced = obs.iloc[balanced_idx]

        add(high_e["candidate_embb_slice_prb"], high_e["candidate_urllc_slice_prb"], "observed_high_embb")
        add(high_u["candidate_embb_slice_prb"], high_u["candidate_urllc_slice_prb"], "observed_high_urllc")
        add(balanced["candidate_embb_slice_prb"], balanced["candidate_urllc_slice_prb"], "observed_balanced")

        # random observed.
        for _ in range(2):
            r = obs.iloc[int(rng.integers(0, len(obs)))]
            add(r["candidate_embb_slice_prb"], r["candidate_urllc_slice_prb"], "observed_random")

    # Remove duplicates while keeping order.
    seen = set()
    unique = []
    for e, u, source in actions:
        key = (e, u)
        if key not in seen:
            seen.add(key)
            unique.append((e, u, source))

    # Ensure fixed candidate count.
    while len(unique) < CANDIDATES_PER_STATE and len(observed_actions) > 0:
        r = observed_actions.iloc[int(rng.integers(0, len(observed_actions)))]
        key = (float(r["candidate_embb_slice_prb"]), float(r["candidate_urllc_slice_prb"]))
        if key not in seen:
            seen.add(key)
            unique.append((key[0], key[1], "observed_random_fill"))

    return unique[:CANDIDATES_PER_STATE]


def build_matching_table(df):
    # For pseudo labels, use observed logged rows indexed by action pair and context.
    # Later final simulator/ns-O-RAN evaluation will replace this.
    table = df.copy()
    table["action_key"] = (
        table["candidate_embb_slice_prb"].round(6).astype(str)
        + "_"
        + table["candidate_urllc_slice_prb"].round(6).astype(str)
    )

    # Average labels by context + action.
    group_cols = ["cluster_code", "slicing_code", "scheduling_code", "action_key"]

    agg_cols = TARGET_COLS + [
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
    ]

    mt = (
        table[group_cols + agg_cols]
        .groupby(group_cols, dropna=False)
        .mean()
        .reset_index()
    )

    return mt


def find_pseudo_label(row, match_table, global_action_summary, cand_e, cand_u):
    action_key = f"{round(cand_e, 6)}_{round(cand_u, 6)}"

    # Try exact context + action match.
    sub = match_table[
        (match_table["cluster_code"] == row["cluster_code"])
        & (match_table["slicing_code"] == row["slicing_code"])
        & (match_table["scheduling_code"] == row["scheduling_code"])
        & (match_table["action_key"] == action_key)
    ]

    if len(sub) > 0:
        rec = sub.iloc[0]
        distance = 0.0
        return rec[TARGET_COLS].to_dict(), distance

    # Fallback: nearest action globally among observed action summaries.
    dist = (
        (global_action_summary["candidate_embb_slice_prb"] - cand_e).abs()
        + (global_action_summary["candidate_urllc_slice_prb"] - cand_u).abs()
    )
    idx = dist.idxmin()
    rec = global_action_summary.loc[idx]
    distance = float(dist.loc[idx])

    return rec[TARGET_COLS].to_dict(), distance


def build_global_action_summary(df):
    return (
        df[["candidate_embb_slice_prb", "candidate_urllc_slice_prb"] + TARGET_COLS]
        .groupby(["candidate_embb_slice_prb", "candidate_urllc_slice_prb"], dropna=False)
        .mean()
        .reset_index()
    )


def source_to_code(source):
    mapping = {
        "logged": 0,
        "previous": 1,
        "local_perturb": 2,
        "observed_high_embb": 3,
        "observed_high_urllc": 4,
        "observed_balanced": 5,
        "observed_random": 6,
        "observed_random_fill": 7,
    }
    return mapping.get(source, 99)


def main():
    rng = np.random.default_rng(SEED)

    print(f"Reading Step 1.1 samples: {INPUT_SAMPLES}")
    df = pd.read_csv(INPUT_SAMPLES)

    print(f"Rows loaded: {len(df)}")

    observed_actions = extract_observed_actions(df)
    observed_actions_path = OUT_DIR / "observed_action_pairs.csv"
    observed_actions.to_csv(observed_actions_path, index=False, encoding="utf-8-sig")

    print(f"Observed action pairs: {len(observed_actions)}")

    match_table = build_matching_table(df)
    global_action_summary = build_global_action_summary(df)

    rows = []

    for split, max_n in MAX_BASE_ROWS_PER_SPLIT.items():
        part = df[df["split"] == split].copy()
        if len(part) > max_n:
            part = part.sample(n=max_n, random_state=SEED)
        print(f"Expanding split={split}, base_rows={len(part)}")

        for _, row in tqdm(part.iterrows(), total=len(part), desc=f"expand {split}"):
            candidates = choose_candidate_actions(row, observed_actions, rng)

            for cand_e, cand_u, source in candidates:
                labels, matched_dist = find_pseudo_label(
                    row, match_table, global_action_summary, cand_e, cand_u
                )

                rec = {}

                for c in ID_COLS:
                    if c in row:
                        rec[c] = row[c]

                for c in STATE_FEATURE_COLS:
                    rec[c] = row[c]

                rec["candidate_embb_slice_prb"] = cand_e
                rec["candidate_urllc_slice_prb"] = cand_u
                rec["candidate_embb_share"] = cand_e / TOTAL_PRB
                rec["candidate_urllc_share"] = cand_u / TOTAL_PRB
                rec["delta_embb_prb"] = cand_e - float(row["prev_embb_slice_prb"])
                rec["delta_urllc_prb"] = cand_u - float(row["prev_urllc_slice_prb"])
                rec["abs_delta_total_prb"] = abs(rec["delta_embb_prb"]) + abs(rec["delta_urllc_prb"])
                rec["same_action_flag"] = int(abs(rec["delta_embb_prb"]) < 1e-9 and abs(rec["delta_urllc_prb"]) < 1e-9)
                rec["candidate_source"] = source
                rec["candidate_source_code"] = source_to_code(source)
                rec["matched_action_distance"] = matched_dist

                for t in TARGET_COLS:
                    rec[t] = labels[t]

                rows.append(rec)

    expanded = pd.DataFrame(rows)

    expanded_path = OUT_DIR / "candidate_expanded_samples.csv"
    expanded.to_csv(expanded_path, index=False, encoding="utf-8-sig")

    # Fit scaler and generate npy.
    feature_cols = FINAL_FEATURE_COLS
    train_df = expanded[expanded["split"] == "train"].copy()
    scaler_stats = robust_standardize_fit(train_df, feature_cols)

    outputs = {
        "candidate_expanded_samples": str(expanded_path),
        "observed_action_pairs": str(observed_actions_path),
    }

    for split in ["train", "calibration", "validation", "test"]:
        part = expanded[expanded["split"] == split].copy()
        if part.empty:
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

    action_summary = {}
    for c in ["candidate_embb_slice_prb", "candidate_urllc_slice_prb", "delta_embb_prb", "delta_urllc_prb", "abs_delta_total_prb", "matched_action_distance"]:
        vals = pd.to_numeric(expanded[c], errors="coerce")
        action_summary[c] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p05": float(vals.quantile(0.05)),
            "p95": float(vals.quantile(0.95)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    source_counts = expanded["candidate_source"].value_counts().to_dict()

    report = {
        "input_samples": str(INPUT_SAMPLES),
        "num_base_rows_used": int(sum(MAX_BASE_ROWS_PER_SPLIT.values())),
        "num_expanded_rows": int(len(expanded)),
        "candidates_per_state": CANDIDATES_PER_STATE,
        "observed_action_pair_count": int(len(observed_actions)),
        "split_counts": {str(k): int(v) for k, v in expanded["split"].value_counts().to_dict().items()},
        "feature_count": int(len(feature_cols)),
        "target_count": int(len(TARGET_COLS)),
        "feature_names": feature_cols,
        "target_names": TARGET_COLS,
        "candidate_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "action_summary": action_summary,
        "outputs": outputs,
        "important_note": (
            "This is a candidate-expanded pseudo-label dataset. "
            "Labels for non-logged candidate actions are assigned by context/action matching. "
            "Final main experiments should still use ns-O-RAN interaction or high-fidelity evaluation where possible."
        )
    }

    with open(OUT_DIR / "step1_1b_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(RESULT_DIR / "step1_1b_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 1.1B Candidate-Expanded State-Action Dataset Report\n\n")
        f.write(f"- Base rows used: {report['num_base_rows_used']}\n")
        f.write(f"- Expanded rows: {report['num_expanded_rows']}\n")
        f.write(f"- Candidates per state: {CANDIDATES_PER_STATE}\n")
        f.write(f"- Observed action pairs: {len(observed_actions)}\n")
        f.write(f"- Feature count: {len(feature_cols)}\n")
        f.write(f"- Target count: {len(TARGET_COLS)}\n\n")

        f.write("## Split counts\n\n")
        for k, v in report["split_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Candidate source counts\n\n")
        for k, v in report["candidate_source_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This dataset expands each state with multiple candidate actions.\n")
        f.write("- Non-logged candidate labels are pseudo-labels from nearest observed action matching.\n")
        f.write("- Use this dataset to train a first formal candidate-aware digital twin.\n")
        f.write("- Later ns-O-RAN interaction will replace pseudo labels for final main experiments.\n")

    print("Step 1.1B completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()