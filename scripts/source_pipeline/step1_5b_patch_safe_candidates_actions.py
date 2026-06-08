import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAFE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "candidate_safe_set"
)

CAND_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step1_5b"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_FILE = SAFE_DIR / "step1_5_safe_labeled_candidates.csv"
CAND_FILE = CAND_DIR / "candidate_expanded_samples.csv"

OUT_FILE = SAFE_DIR / "step1_5_safe_labeled_candidates_patched.csv"

ACTION_COLS = [
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
    "action_key",
]


def main():
    print(f"Reading safe candidates: {SAFE_FILE}")
    safe_df = pd.read_csv(SAFE_FILE)

    print(f"Reading candidate expanded samples: {CAND_FILE}")
    cand_df = pd.read_csv(CAND_FILE)

    n_safe = len(safe_df)
    n_cand = len(cand_df)

    print(f"safe rows: {n_safe}")
    print(f"candidate rows: {n_cand}")

    # Step 1.4 saved only the first 300000 test rows.
    # Step 1.5 uses that capped file, so safe_df should have 300000 rows.
    # We need the corresponding first 300000 test candidate rows.
    cand_test = cand_df[cand_df["split"] == "test"].copy().reset_index(drop=True)

    if len(cand_test) < n_safe:
        raise RuntimeError(
            f"Not enough test candidate rows. cand_test={len(cand_test)}, safe={n_safe}"
        )

    cand_test = cand_test.head(n_safe).copy()

    available_action_cols = [c for c in ACTION_COLS if c in cand_test.columns]

    print("Patching action columns:")
    for c in available_action_cols:
        print(f"  - {c}")
        safe_df[c] = cand_test[c].values

    # candidate_id and candidate_group_id should already exist from Step 1.5.
    # If not, recreate them.
    if "candidate_group_id" not in safe_df.columns:
        safe_df["candidate_group_id"] = safe_df.index // 5

    if "candidate_id" not in safe_df.columns:
        safe_df["candidate_id"] = safe_df.index % 5

    # Sanity check.
    required = [
        "candidate_group_id",
        "candidate_id",
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
        "true_V_embb",
        "true_V_urllc",
        "true_management_utility",
        "pred_mean_V_embb",
        "pred_mean_V_urllc",
        "pred_mean_management_utility",
        "upper_V_embb_q90",
        "upper_V_urllc_q90",
        "true_joint_safe",
        "cert_joint_safe",
    ]

    missing = [c for c in required if c not in safe_df.columns]
    if missing:
        raise RuntimeError(f"Still missing columns after patch: {missing}")

    safe_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    report = {
        "safe_file": str(SAFE_FILE),
        "candidate_file": str(CAND_FILE),
        "output_file": str(OUT_FILE),
        "safe_rows": int(n_safe),
        "candidate_test_rows": int(len(cand_df[cand_df["split"] == "test"])),
        "patched_action_cols": available_action_cols,
        "status": "patched successfully",
        "important_note": (
            "This patch restores candidate action columns that were not preserved in Step 1.4 prediction output."
        ),
    }

    report_json = SAFE_DIR / "step1_5b_patch_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md = RESULT_DIR / "step1_5b_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Step 1.5B Patch Safe Candidates with Action Columns\n\n")
        f.write(f"- safe rows: {n_safe}\n")
        f.write(f"- output file: `{OUT_FILE}`\n\n")
        f.write("## Patched columns\n\n")
        for c in available_action_cols:
            f.write(f"- {c}\n")
        f.write("\n## Important interpretation\n\n")
        f.write("- Step 1.4 prediction output did not preserve candidate action fields.\n")
        f.write("- This patch restores action fields from the corresponding test candidate-expanded samples.\n")
        f.write("- Step 1.6 should use the patched file.\n")

    print("Step 1.5B patch completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()