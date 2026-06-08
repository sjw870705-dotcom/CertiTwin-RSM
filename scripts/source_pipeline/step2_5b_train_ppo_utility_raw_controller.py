import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "rl_env"
)

EVAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
)

MODEL_DIR = PROJECT_ROOT / "models" / "ppo_utility_controller"
RESULT_DIR = PROJECT_ROOT / "results" / "step2_5b"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RL_NPZ = RL_DIR / "rl_env_groups.npz"
CANDIDATES_FILE = EVAL_DIR / "controller_candidate_groups_patched.csv"

MODEL_PATH = MODEL_DIR / "ppo_utility_policy.pt"
TRAIN_LOG = MODEL_DIR / "step2_5b_ppo_utility_training_log.csv"
METRICS_JSON = MODEL_DIR / "step2_5b_ppo_utility_metrics.json"

RAW_ACTIONS_CSV = EVAL_DIR / "step2_5b_ppo_utility_raw_actions.csv"
PAIRWISE_CSV = EVAL_DIR / "step2_5b_ppo_utility_raw_vs_shielded_pairwise.csv"
SUMMARY_CSV = EVAL_DIR / "step2_5b_ppo_utility_shielding_summary.csv"
REPORT_MD = RESULT_DIR / "step2_5b_report.md"

SEED = 20260604

# PPO settings
NUM_UPDATES = 300
BATCH_SIZE = 2048
MINIBATCH_SIZE = 512
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4

# Utility-oriented reward.
# Important: no explicit unsafe penalty and no large risk penalty.
LAMBDA_RECONFIG = 0.02

# Shield settings
Q_LEVEL = 90
LAMBDA_DISTANCE = 0.02
LAMBDA_UPPER_RISK = 0.50


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_env():
    data = np.load(RL_NPZ)

    env = {
        "state": data["state"].astype(np.float32),
        "action_features": data["action_features"].astype(np.float32),

        "true_V_embb": data["true_V_embb"].astype(np.float32),
        "true_V_urllc": data["true_V_urllc"].astype(np.float32),
        "true_V_total": data["true_V_total"].astype(np.float32),
        "true_utility": data["true_utility"].astype(np.float32),
        "true_safe": data["true_safe"].astype(np.float32),
        "cert_safe": data["cert_safe"].astype(np.float32),

        "candidate_ids": data["candidate_ids"].astype(np.int64),
        "group_ids": data["group_ids"].astype(np.int64),

        "train_idx": data["train_idx"].astype(np.int64),
        "validation_idx": data["validation_idx"].astype(np.int64),
        "test_idx": data["test_idx"].astype(np.int64),
    }

    return env


def compute_utility_reward(env):
    """
    Utility-oriented reward:
        reward = true_utility - lambda_reconfig * reconfiguration_cost

    action_features shape: [G, A, F]
    In Step 2.4, action_cols were:
        candidate_embb_slice_prb
        candidate_urllc_slice_prb
        candidate_embb_share
        candidate_urllc_share
        delta_embb_prb
        delta_urllc_prb
        abs_delta_total_prb

    So abs_delta_total_prb is index 6.
    """
    true_utility = env["true_utility"]
    abs_delta_total_prb = env["action_features"][:, :, 6]

    # action_features were robust-scaled in Step 2.4, so do not use them directly for
    # a physically exact PRB penalty. We only use a small penalty to discourage unstable actions.
    # Since it is scaled, clip to avoid extreme penalty.
    reconfig_scaled = np.clip(np.abs(abs_delta_total_prb), 0, 5)

    reward = true_utility - LAMBDA_RECONFIG * reconfig_scaled

    return reward.astype(np.float32)


def train_ppo(env, reward_matrix, device):
    state = torch.from_numpy(env["state"]).to(device)
    reward = torch.from_numpy(reward_matrix).to(device)

    train_idx = env["train_idx"]
    state_dim = env["state"].shape[1]
    action_dim = reward_matrix.shape[1]

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    rng = np.random.default_rng(SEED)

    logs = []

    for update in range(1, NUM_UPDATES + 1):
        batch_idx_np = rng.choice(train_idx, size=BATCH_SIZE, replace=True)
        batch_idx = torch.from_numpy(batch_idx_np).long().to(device)

        obs = state[batch_idx]

        with torch.no_grad():
            actions, old_logprob, _, old_value = model.act(obs)
            rewards = reward[batch_idx, actions]
            returns = rewards
            advantages = returns - old_value
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(BATCH_SIZE, device=device)

            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb = perm[start:start + MINIBATCH_SIZE]

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logprob = old_logprob[mb]
                mb_returns = returns[mb]
                mb_adv = advantages[mb]

                logprob, entropy, value = model.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(logprob - mb_old_logprob)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((value - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if update % 10 == 0 or update == 1:
            eval_train = evaluate_policy(model, env, reward_matrix, env["train_idx"][:min(5000, len(env["train_idx"]))], device)
            eval_val = evaluate_policy(model, env, reward_matrix, env["validation_idx"], device)

            rec = {
                "update": update,
                "train_reward_mean": eval_train["reward_mean"],
                "val_reward_mean": eval_val["reward_mean"],
                "val_true_unsafe_rate": eval_val["true_unsafe_rate"],
                "val_utility_mean": eval_val["utility_mean"],
                "val_cert_safe_rate": eval_val["cert_safe_rate"],
            }
            logs.append(rec)

            print(
                f"Update {update:04d} | "
                f"train_reward={rec['train_reward_mean']:.4f} | "
                f"val_reward={rec['val_reward_mean']:.4f} | "
                f"val_unsafe={rec['val_true_unsafe_rate']:.4f} | "
                f"val_utility={rec['val_utility_mean']:.4f} | "
                f"val_cert_safe={rec['val_cert_safe_rate']:.4f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "seed": SEED,
            "reward_type": "utility_oriented",
            "lambda_reconfig": LAMBDA_RECONFIG,
        },
        MODEL_PATH,
    )

    pd.DataFrame(logs).to_csv(TRAIN_LOG, index=False, encoding="utf-8-sig")

    return model, logs


@torch.no_grad()
def select_actions(model, states_np, indices, device, deterministic=True):
    model.eval()
    states = torch.from_numpy(states_np[indices]).to(device)
    logits, _ = model(states)

    if deterministic:
        actions = torch.argmax(logits, dim=-1)
    else:
        dist = Categorical(logits=logits)
        actions = dist.sample()

    return actions.cpu().numpy().astype(np.int64)


@torch.no_grad()
def evaluate_policy(model, env, reward_matrix, indices, device):
    actions = select_actions(model, env["state"], indices, device, deterministic=True)

    reward = reward_matrix[indices, actions]
    true_safe = env["true_safe"][indices, actions]
    cert_safe = env["cert_safe"][indices, actions]
    utility = env["true_utility"][indices, actions]
    v_embb = env["true_V_embb"][indices, actions]
    v_urllc = env["true_V_urllc"][indices, actions]
    v_total = env["true_V_total"][indices, actions]

    return {
        "reward_mean": float(reward.mean()),
        "true_safe_rate": float(true_safe.mean()),
        "true_unsafe_rate": float(1.0 - true_safe.mean()),
        "cert_safe_rate": float(cert_safe.mean()),
        "cert_unsafe_rate": float(1.0 - cert_safe.mean()),
        "utility_mean": float(utility.mean()),
        "V_embb_mean": float(v_embb.mean()),
        "V_urllc_mean": float(v_urllc.mean()),
        "V_total_mean": float(v_total.mean()),
    }


def build_raw_actions_csv(model, env, reward_matrix, device):
    test_idx = env["test_idx"]
    actions = select_actions(model, env["state"], test_idx, device, deterministic=True)

    rows = []

    for i, gid_index in enumerate(test_idx):
        action = int(actions[i])
        group_id = int(env["group_ids"][gid_index])
        candidate_id = int(env["candidate_ids"][gid_index, action])

        rows.append({
            "controller": "PPO-Utility",
            "candidate_group_id": group_id,
            "raw_candidate_id": candidate_id,
            "raw_action_index": action,

            "raw_true_V_embb": float(env["true_V_embb"][gid_index, action]),
            "raw_true_V_urllc": float(env["true_V_urllc"][gid_index, action]),
            "raw_true_V_total": float(env["true_V_total"][gid_index, action]),
            "raw_true_management_utility": float(env["true_utility"][gid_index, action]),

            "raw_true_joint_safe": int(env["true_safe"][gid_index, action]),
            "raw_cert_joint_safe": int(env["cert_safe"][gid_index, action]),
            "raw_reward": float(reward_matrix[gid_index, action]),
        })

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(RAW_ACTIONS_CSV, index=False, encoding="utf-8-sig")
    return raw_df


def shield_actions(raw_df, candidates):
    cand_groups = {gid: g.copy() for gid, g in candidates.groupby("candidate_group_id", sort=False)}

    pair_rows = []

    for _, raw in raw_df.iterrows():
        gid = int(raw["candidate_group_id"])
        group = cand_groups[gid]

        raw_candidate_id = int(raw["raw_candidate_id"])
        raw_row = group[group["candidate_id"] == raw_candidate_id]
        if len(raw_row) != 1:
            raise RuntimeError(f"Cannot locate raw candidate {raw_candidate_id} in group {gid}")
        raw_cand = raw_row.iloc[0]

        if int(raw_cand["cert_joint_safe"]) == 1:
            shielded = raw_cand
            mode = "pass"
        else:
            cert_safe = group[group["cert_joint_safe"] == 1].copy()

            if len(cert_safe) > 0:
                de = (cert_safe["candidate_embb_slice_prb"] - raw_cand["candidate_embb_slice_prb"]).abs()
                du = (cert_safe["candidate_urllc_slice_prb"] - raw_cand["candidate_urllc_slice_prb"]).abs()
                dist = de + du
                upper_risk = 0.5 * cert_safe["upper_V_embb_q90"] + 0.5 * cert_safe["upper_V_urllc_q90"]

                score = (
                    cert_safe["pred_mean_management_utility"]
                    - LAMBDA_UPPER_RISK * upper_risk
                    - LAMBDA_DISTANCE * dist
                )

                shielded = cert_safe.loc[score.idxmax()]
                mode = "shield_to_certified_safe"
            else:
                upper_risk = 0.5 * group["upper_V_embb_q90"] + 0.5 * group["upper_V_urllc_q90"]
                shielded = group.loc[upper_risk.idxmin()]
                mode = "fallback_no_certified_safe"

        pair_rows.append({
            "controller": "PPO-Utility",
            "shielded_controller": "CertiTwin-PPO-Utility",
            "candidate_group_id": gid,

            "raw_candidate_id": int(raw_candidate_id),
            "shield_candidate_id": int(shielded["candidate_id"]),
            "shield_mode": mode,

            "raw_true_joint_safe": int(raw_cand["true_joint_safe"]),
            "shield_true_joint_safe": int(shielded["true_joint_safe"]),

            "raw_cert_joint_safe": int(raw_cand["cert_joint_safe"]),
            "shield_cert_joint_safe": int(shielded["cert_joint_safe"]),

            "raw_true_V_embb": float(raw_cand["true_V_embb"]),
            "shield_true_V_embb": float(shielded["true_V_embb"]),
            "raw_true_V_urllc": float(raw_cand["true_V_urllc"]),
            "shield_true_V_urllc": float(shielded["true_V_urllc"]),
            "raw_true_V_total": float(raw_cand["true_V_total"]),
            "shield_true_V_total": float(shielded["true_V_total"]),

            "raw_true_management_utility": float(raw_cand["true_management_utility"]),
            "shield_true_management_utility": float(shielded["true_management_utility"]),

            "raw_embb_slice_prb": float(raw_cand["candidate_embb_slice_prb"]),
            "raw_urllc_slice_prb": float(raw_cand["candidate_urllc_slice_prb"]),
            "shield_embb_slice_prb": float(shielded["candidate_embb_slice_prb"]),
            "shield_urllc_slice_prb": float(shielded["candidate_urllc_slice_prb"]),
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(PAIRWISE_CSV, index=False, encoding="utf-8-sig")

    return pair_df


def summarize_pairwise(pair_df):
    g = pair_df

    raw_safe = g["raw_true_joint_safe"].astype(int).to_numpy()
    shield_safe = g["shield_true_joint_safe"].astype(int).to_numpy()
    raw_cert = g["raw_cert_joint_safe"].astype(int).to_numpy()
    shield_cert = g["shield_cert_joint_safe"].astype(int).to_numpy()

    raw_u = g["raw_true_management_utility"].to_numpy()
    shield_u = g["shield_true_management_utility"].to_numpy()

    changed = g["raw_candidate_id"].to_numpy() != g["shield_candidate_id"].to_numpy()

    dist = (
        np.abs(g["raw_embb_slice_prb"].to_numpy() - g["shield_embb_slice_prb"].to_numpy())
        + np.abs(g["raw_urllc_slice_prb"].to_numpy() - g["shield_urllc_slice_prb"].to_numpy())
    )

    modes = g["shield_mode"].astype(str).to_numpy()

    rec = {
        "controller": "PPO-Utility",
        "shielded_controller": "CertiTwin-PPO-Utility",
        "num_groups": int(len(g)),

        "raw_true_safe_rate": float(raw_safe.mean()),
        "shielded_true_safe_rate": float(shield_safe.mean()),
        "raw_true_unsafe_rate": float(1.0 - raw_safe.mean()),
        "shielded_true_unsafe_rate": float(1.0 - shield_safe.mean()),

        "raw_certified_safe_rate": float(raw_cert.mean()),
        "shielded_certified_safe_rate": float(shield_cert.mean()),

        "raw_V_embb_mean": float(g["raw_true_V_embb"].mean()),
        "shielded_V_embb_mean": float(g["shield_true_V_embb"].mean()),
        "raw_V_urllc_mean": float(g["raw_true_V_urllc"].mean()),
        "shielded_V_urllc_mean": float(g["shield_true_V_urllc"].mean()),
        "raw_V_total_mean": float(g["raw_true_V_total"].mean()),
        "shielded_V_total_mean": float(g["shield_true_V_total"].mean()),

        "utility_raw_mean": float(raw_u.mean()),
        "utility_shielded_mean": float(shield_u.mean()),
        "utility_retention": float(np.mean(shield_u / (raw_u + 1e-9))),

        "action_changed_rate": float(changed.mean()),
        "mean_action_distance": float(dist.mean()),
        "p95_action_distance": float(np.quantile(dist, 0.95)),

        "pass_through_rate": float(np.mean(modes == "pass")),
        "shield_to_safe_rate": float(np.mean(modes == "shield_to_certified_safe")),
        "fallback_rate": float(np.mean(modes == "fallback_no_certified_safe")),
    }

    rec["unsafe_reduction_abs"] = rec["raw_true_unsafe_rate"] - rec["shielded_true_unsafe_rate"]
    if rec["raw_true_unsafe_rate"] > 1e-12:
        rec["unsafe_reduction_rel"] = rec["unsafe_reduction_abs"] / rec["raw_true_unsafe_rate"]
    else:
        rec["unsafe_reduction_rel"] = 0.0

    rec["utility_loss_abs"] = rec["utility_raw_mean"] - rec["utility_shielded_mean"]

    summary_df = pd.DataFrame([rec])
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    return rec


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = load_env()
    reward_matrix = compute_utility_reward(env)

    print("Utility-oriented reward summary:")
    print({
        "mean": float(reward_matrix.mean()),
        "median": float(np.median(reward_matrix)),
        "p05": float(np.quantile(reward_matrix, 0.05)),
        "p95": float(np.quantile(reward_matrix, 0.95)),
        "best_action_reward_mean": float(reward_matrix.max(axis=1).mean()),
        "random_action_reward_mean": float(reward_matrix.mean(axis=1).mean()),
    })

    model, logs = train_ppo(env, reward_matrix, device)

    train_metrics = evaluate_policy(model, env, reward_matrix, env["train_idx"], device)
    val_metrics = evaluate_policy(model, env, reward_matrix, env["validation_idx"], device)
    test_metrics = evaluate_policy(model, env, reward_matrix, env["test_idx"], device)

    raw_df = build_raw_actions_csv(model, env, reward_matrix, device)

    print(f"Reading candidates for shielding: {CANDIDATES_FILE}")
    candidates = pd.read_csv(CANDIDATES_FILE)

    pair_df = shield_actions(raw_df, candidates)
    shielding_summary = summarize_pairwise(pair_df)

    metrics = {
        "config": {
            "seed": SEED,
            "reward_type": "utility_oriented",
            "lambda_reconfig": LAMBDA_RECONFIG,
            "num_updates": NUM_UPDATES,
            "batch_size": BATCH_SIZE,
            "minibatch_size": MINIBATCH_SIZE,
            "ppo_epochs": PPO_EPOCHS,
            "clip_eps": CLIP_EPS,
            "entropy_coef": ENTROPY_COEF,
            "value_coef": VALUE_COEF,
            "learning_rate": LR,
            "device": device,
        },
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "shielding_summary": shielding_summary,
        "outputs": {
            "model": str(MODEL_PATH),
            "train_log": str(TRAIN_LOG),
            "raw_actions": str(RAW_ACTIONS_CSV),
            "pairwise": str(PAIRWISE_CSV),
            "summary": str(SUMMARY_CSV),
        },
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 2.5B Utility-Oriented PPO and CertiTwin-PPO Report\n\n")

        f.write("## PPO setup\n\n")
        for k, v in metrics["config"].items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Raw PPO-Utility metrics\n\n")
        for split_name, m in [
            ("train", train_metrics),
            ("validation", val_metrics),
            ("test", test_metrics),
        ]:
            f.write(f"### {split_name}\n")
            for k, v in m.items():
                f.write(f"- {k}: {v}\n")

        f.write("\n## PPO-Utility vs CertiTwin-PPO-Utility\n\n")
        for k, v in shielding_summary.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in metrics["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Important interpretation\n\n")
        f.write("- Raw PPO-Utility and CertiTwin-PPO-Utility use the same trained PPO checkpoint.\n")
        f.write("- PPO-Utility is trained with utility-oriented reward and no explicit unsafe penalty.\n")
        f.write("- CertiTwin only modifies PPO-selected actions at execution time.\n")

    print("Step 2.5B completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()