import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CTRL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

CAND_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "state_action_candidate_dataset"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step2_1b"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CTRL_FILE = CTRL_DIR / "controller_candidate_groups.csv"
CAND_FILE = CAND_DIR / "candidate_expanded_samples.csv"

OUT_FILE = CTRL_DIR / "controller_candidate_groups_patched.csv"

STATE_COLS = [
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


def main():
    print(f"Reading controller candidate groups: {CTRL_FILE}")
    ctrl_df = pd.read_csv(CTRL_FILE)

    print(f"Reading candidate expanded samples: {CAND_FILE}")
    cand_df = pd.read_csv(CAND_FILE)

    n_ctrl = len(ctrl_df)
    print(f"controller rows: {n_ctrl}")
    print(f"candidate expanded rows: {len(cand_df)}")

    # Step 1.4/1.5/2.1 used the first capped 300000 test candidates.
    # So patch from the first 300000 rows of test candidate-expanded samples.
    cand_test = cand_df[cand_df["split"] == "test"].copy().reset_index(drop=True)

    if len(cand_test) < n_ctrl:
        raise RuntimeError(
            f"Not enough test candidate rows. cand_test={len(cand_test)}, controller={n_ctrl}"
        )

    cand_test = cand_test.head(n_ctrl).copy()

    available_state_cols = [c for c in STATE_COLS if c in cand_test.columns]

    missing_in_source = [c for c in STATE_COLS if c not in cand_test.columns]
    if missing_in_source:
        raise RuntimeError(f"State columns missing in candidate source: {missing_in_source}")

    print("Patching state columns:")
    for c in available_state_cols:
        print(f"  - {c}")
        ctrl_df[c] = cand_test[c].values

    # Basic alignment check: candidate actions should match.
    action_cols = ["candidate_embb_slice_prb", "candidate_urllc_slice_prb"]
    for c in action_cols:
        if c in ctrl_df.columns and c in cand_test.columns:
            max_diff = (ctrl_df[c].astype(float) - cand_test[c].astype(float)).abs().max()
            print(f"alignment check {c}: max_diff={max_diff}")
            if max_diff > 1e-6:
                raise RuntimeError(
                    f"Action alignment failed for {c}. max_diff={max_diff}. "
                    "Do not continue; check row ordering."
                )

    required_after_patch = [
        "candidate_group_id",
        "candidate_id",
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
        "true_V_embb",
        "true_V_urllc",
        "true_V_total",
        "true_management_utility",
        "cert_joint_safe",
    ] + STATE_COLS

    missing_after_patch = [c for c in required_after_patch if c not in ctrl_df.columns]
    if missing_after_patch:
        raise RuntimeError(f"Still missing columns after patch: {missing_after_patch}")

    ctrl_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    report = {
        "input_controller_file": str(CTRL_FILE),
        "input_candidate_file": str(CAND_FILE),
        "output_file": str(OUT_FILE),
        "controller_rows": int(n_ctrl),
        "candidate_test_rows": int(len(cand_df[cand_df["split"] == "test"])),
        "patched_state_cols": available_state_cols,
        "status": "patched successfully",
        "important_note": (
            "This patch restores previous-state features into controller_candidate_groups. "
            "Step 2.4 RL wrapper should use controller_candidate_groups_patched.csv."
        ),
    }

    report_json = CTRL_DIR / "step2_1b_patch_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md = RESULT_DIR / "step2_1b_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Step 2.1B Patch Controller Groups with State Features\n\n")
        f.write(f"- controller rows: {n_ctrl}\n")
        f.write(f"- output file: `{OUT_FILE}`\n\n")

        f.write("## Patched state columns\n\n")
        for c in available_state_cols:
            f.write(f"- {c}\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- Step 2.1 controller candidate groups lacked previous-state features.\n")
        f.write("- This patch restores them from the corresponding test candidate-expanded rows.\n")
        f.write("- Step 2.4 should use the patched controller candidate file.\n")

    print("Step 2.1B patch completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()