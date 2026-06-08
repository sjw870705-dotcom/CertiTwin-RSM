import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

INPUT_CANDIDATES = EVAL_DIR / "controller_candidate_groups_patched.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "rl_env"
)

RESULT_DIR = PROJECT_ROOT / "results" / "step2_4"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = OUT_DIR / "rl_env_groups.npz"
OUTPUT_METADATA = OUT_DIR / "rl_env_metadata.json"
OUTPUT_REPORT_JSON = OUT_DIR / "step2_4_env_report.json"
OUTPUT_REPORT_MD = RESULT_DIR / "step2_4_report.md"

GROUP_SIZE = 5
SEED = 20260604

# Reward design for raw RL controllers.
# CertiTwin shield later uses the same raw checkpoint/action output.
LAMBDA_RISK = 0.80
LAMBDA_UNSAFE = 0.30
LAMBDA_RECONFIG = 0.05

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

ACTION_COLS = [
    "candidate_embb_slice_prb",
    "candidate_urllc_slice_prb",
    "candidate_embb_share",
    "candidate_urllc_share",
    "delta_embb_prb",
    "delta_urllc_prb",
    "abs_delta_total_prb",
]

TARGET_COLS = [
    "true_V_embb",
    "true_V_urllc",
    "true_V_total",
    "true_management_utility",
    "true_joint_safe",
    "cert_joint_safe",
    "pred_mean_management_utility",
    "pred_mean_V_embb",
    "pred_mean_V_urllc",
    "upper_V_embb_q90",
    "upper_V_urllc_q90",
]


def robust_scale_fit(values):
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)
    return median.astype(np.float32), scale.astype(np.float32)


def robust_scale_transform(values, median, scale):
    x = (values - median) / scale
    x = np.clip(x, -20, 20)
    x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
    return x.astype(np.float32)


def check_columns(df):
    required = ["candidate_group_id", "candidate_id"] + STATE_COLS + ACTION_COLS + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def build_group_arrays(df):
    df = df.sort_values(["candidate_group_id", "candidate_id"]).copy()

    group_counts = df.groupby("candidate_group_id").size()
    if not (group_counts == GROUP_SIZE).all():
        raise RuntimeError("Not all candidate groups have exactly 5 candidates.")

    num_groups = int(df["candidate_group_id"].nunique())

    # State: one state per group, taken from candidate_id 0 because state is identical across candidates.
    first = df.groupby("candidate_group_id").first().reset_index()

    state = first[STATE_COLS].to_numpy(dtype=np.float32)

    # Candidate action/features/labels: shape [G, A, F]
    action_features = df[ACTION_COLS].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE, len(ACTION_COLS))

    true_V_embb = df["true_V_embb"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    true_V_urllc = df["true_V_urllc"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    true_V_total = df["true_V_total"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    true_utility = df["true_management_utility"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)

    true_safe = df["true_joint_safe"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    cert_safe = df["cert_joint_safe"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)

    pred_utility = df["pred_mean_management_utility"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    pred_V_embb = df["pred_mean_V_embb"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    pred_V_urllc = df["pred_mean_V_urllc"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)

    upper_V_embb = df["upper_V_embb_q90"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)
    upper_V_urllc = df["upper_V_urllc_q90"].to_numpy(dtype=np.float32).reshape(num_groups, GROUP_SIZE)

    candidate_ids = df["candidate_id"].to_numpy(dtype=np.int64).reshape(num_groups, GROUP_SIZE)
    group_ids = first["candidate_group_id"].to_numpy(dtype=np.int64)

    return {
        "state": state,
        "action_features": action_features,
        "true_V_embb": true_V_embb,
        "true_V_urllc": true_V_urllc,
        "true_V_total": true_V_total,
        "true_utility": true_utility,
        "true_safe": true_safe,
        "cert_safe": cert_safe,
        "pred_utility": pred_utility,
        "pred_V_embb": pred_V_embb,
        "pred_V_urllc": pred_V_urllc,
        "upper_V_embb": upper_V_embb,
        "upper_V_urllc": upper_V_urllc,
        "candidate_ids": candidate_ids,
        "group_ids": group_ids,
    }


def compute_rewards(arrays):
    true_utility = arrays["true_utility"]
    true_V_embb = arrays["true_V_embb"]
    true_V_urllc = arrays["true_V_urllc"]
    true_safe = arrays["true_safe"]

    action_features = arrays["action_features"]
    abs_delta_total_prb_idx = ACTION_COLS.index("abs_delta_total_prb")
    reconfig = action_features[:, :, abs_delta_total_prb_idx] / 50.0

    joint_risk = 0.5 * true_V_embb + 0.5 * true_V_urllc
    unsafe_penalty = 1.0 - true_safe

    reward = (
        true_utility
        - LAMBDA_RISK * joint_risk
        - LAMBDA_UNSAFE * unsafe_penalty
        - LAMBDA_RECONFIG * reconfig
    )

    return reward.astype(np.float32)


def main():
    print(f"Reading controller candidates: {INPUT_CANDIDATES}")
    df = pd.read_csv(INPUT_CANDIDATES)

    check_columns(df)

    print(f"Candidate rows: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")

    arrays = build_group_arrays(df)
    reward = compute_rewards(arrays)

    # Observation for RL: group state only.
    # Candidate action features are available for later actor variants, but initial PPO can use group state.
    state_median, state_scale = robust_scale_fit(arrays["state"])
    state_scaled = robust_scale_transform(arrays["state"], state_median, state_scale)

    action_flat = arrays["action_features"].reshape(-1, len(ACTION_COLS))
    action_median, action_scale = robust_scale_fit(action_flat)
    action_scaled = robust_scale_transform(action_flat, action_median, action_scale).reshape(
        arrays["action_features"].shape
    )

    # Split groups into train/validation/test by group id order.
    num_groups = arrays["state"].shape[0]
    rng = np.random.default_rng(SEED)
    indices = np.arange(num_groups)
    rng.shuffle(indices)

    n_train = int(num_groups * 0.60)
    n_val = int(num_groups * 0.20)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])

    np.savez_compressed(
        OUTPUT_NPZ,
        state=state_scaled,
        action_features=action_scaled,
        reward=reward,

        true_V_embb=arrays["true_V_embb"],
        true_V_urllc=arrays["true_V_urllc"],
        true_V_total=arrays["true_V_total"],
        true_utility=arrays["true_utility"],
        true_safe=arrays["true_safe"],
        cert_safe=arrays["cert_safe"],

        pred_utility=arrays["pred_utility"],
        pred_V_embb=arrays["pred_V_embb"],
        pred_V_urllc=arrays["pred_V_urllc"],
        upper_V_embb=arrays["upper_V_embb"],
        upper_V_urllc=arrays["upper_V_urllc"],

        candidate_ids=arrays["candidate_ids"],
        group_ids=arrays["group_ids"],

        train_idx=train_idx,
        validation_idx=val_idx,
        test_idx=test_idx,
    )

    reward_summary = {
        "reward_mean": float(reward.mean()),
        "reward_median": float(np.median(reward)),
        "reward_p05": float(np.quantile(reward, 0.05)),
        "reward_p95": float(np.quantile(reward, 0.95)),
        "best_action_reward_mean": float(reward.max(axis=1).mean()),
        "random_action_reward_mean": float(reward.mean(axis=1).mean()),
    }

    safety_summary = {
        "true_safe_candidate_ratio": float(arrays["true_safe"].mean()),
        "cert_safe_candidate_ratio": float(arrays["cert_safe"].mean()),
        "groups_with_true_safe_rate": float((arrays["true_safe"].sum(axis=1) > 0).mean()),
        "groups_with_cert_safe_rate": float((arrays["cert_safe"].sum(axis=1) > 0).mean()),
        "mean_true_safe_candidates_per_group": float(arrays["true_safe"].sum(axis=1).mean()),
        "mean_cert_safe_candidates_per_group": float(arrays["cert_safe"].sum(axis=1).mean()),
    }

    metadata = {
        "input_candidates": str(INPUT_CANDIDATES),
        "output_npz": str(OUTPUT_NPZ),
        "num_groups": int(num_groups),
        "group_size": GROUP_SIZE,
        "state_dim": int(arrays["state"].shape[1]),
        "action_feature_dim": int(arrays["action_features"].shape[2]),
        "state_cols": STATE_COLS,
        "action_cols": ACTION_COLS,
        "reward_definition": {
            "reward": "utility - lambda_risk * joint_risk - lambda_unsafe * unsafe_penalty - lambda_reconfig * reconfig",
            "lambda_risk": LAMBDA_RISK,
            "lambda_unsafe": LAMBDA_UNSAFE,
            "lambda_reconfig": LAMBDA_RECONFIG,
        },
        "split_counts": {
            "train_groups": int(len(train_idx)),
            "validation_groups": int(len(val_idx)),
            "test_groups": int(len(test_idx)),
        },
        "reward_summary": reward_summary,
        "safety_summary": safety_summary,
        "important_note": (
            "This wrapper defines a one-step discrete-action RL environment. "
            "Each episode corresponds to one candidate group, and the action is candidate_id in {0,...,4}. "
            "Raw PPO/SAC controllers will be trained/evaluated here; CertiTwin shielding will reuse their selected raw actions."
        ),
    }

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.4 RL Environment Wrapper Report\n\n")

        f.write("## Environment interface\n\n")
        f.write("- Observation: group-level state vector\n")
        f.write("- Action: discrete candidate_id in {0,1,2,3,4}\n")
        f.write("- Reward: utility - risk penalty - unsafe penalty - reconfiguration penalty\n")
        f.write("- Episode: one-step candidate-group decision\n\n")

        f.write("## Dataset summary\n\n")
        f.write(f"- num groups: {num_groups}\n")
        f.write(f"- group size: {GROUP_SIZE}\n")
        f.write(f"- state dim: {arrays['state'].shape[1]}\n")
        f.write(f"- action feature dim: {arrays['action_features'].shape[2]}\n\n")

        f.write("## Split counts\n\n")
        for k, v in metadata["split_counts"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Reward summary\n\n")
        for k, v in reward_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Safety summary\n\n")
        for k, v in safety_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        f.write(f"- rl_env_groups: `{OUTPUT_NPZ}`\n")
        f.write(f"- metadata: `{OUTPUT_METADATA}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- This step builds the unified RL environment wrapper.\n")
        f.write("- It does not train PPO/SAC yet.\n")
        f.write("- Step 2.5 will train a discrete PPO baseline and evaluate raw vs shielded actions.\n")

    print("Step 2.4 completed.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()