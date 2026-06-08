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

RESULT_DIR = PROJECT_ROOT / "results" / "step4_1b"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_FILE = EVAL_DIR / "controller_candidate_groups_patched.csv"

RAW_CLASSICAL = EVAL_DIR / "raw_controller_actions_classical.csv"
RAW_PPO_UTILITY = EVAL_DIR / "step2_5b_ppo_utility_raw_actions.csv"
RAW_PPO_PENALTY = EVAL_DIR / "step2_5_ppo_raw_actions.csv"

OUT_SUMMARY = RESULT_DIR / "step4_1b_certificate_inflation_summary.csv"
OUT_OVERALL = RESULT_DIR / "step4_1b_certificate_inflation_overall.csv"
OUT_PAPER = RESULT_DIR / "step4_1b_certificate_inflation_paper.csv"
OUT_REPORT_JSON = RESULT_DIR / "step4_1b_certificate_inflation_report.json"
OUT_REPORT_MD = RESULT_DIR / "step4_1b_report.md"

SEED = 20260604

NOISE_SCALES = [0.25, 0.50, 0.75, 1.00]
ALPHA_LIST = [1.00, 1.25, 1.50, 2.00, 2.50]

EPS_EMBB = 0.29489304824809814
EPS_URLLC = 0.09153290412936986

# From Step 1.4 q90 calibration.
Q90_EMBB = 5.3484
Q90_URLLC = 2.7258

LAMBDA_DISTANCE = 0.02
LAMBDA_UPPER_RISK = 0.50


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_raw_actions():
    for p in [RAW_CLASSICAL, RAW_PPO_UTILITY, RAW_PPO_PENALTY]:
        assert_exists(p)

    classical = pd.read_csv(RAW_CLASSICAL)

    ppo_utility = pd.read_csv(RAW_PPO_UTILITY).copy()
    ppo_utility["controller"] = "PPO"

    ppo_penalty = pd.read_csv(RAW_PPO_PENALTY).copy()
    ppo_penalty["controller"] = "PPO-Penalty"

    raw_all = pd.concat([classical, ppo_utility, ppo_penalty], ignore_index=True)

    required = [
        "controller",
        "candidate_group_id",
        "raw_candidate_id",
        "raw_true_joint_safe",
        "raw_true_management_utility",
    ]

    missing = [c for c in required if c not in raw_all.columns]
    if missing:
        raise RuntimeError(f"Raw actions missing columns: {missing}")

    return raw_all


def add_perturbed_predictions(candidates, noise_scale, seed):
    df = candidates.copy()
    rng = np.random.default_rng(seed)

    pred_cols = [
        "pred_mean_V_embb",
        "pred_mean_V_urllc",
        "pred_mean_management_utility",
    ]

    std_cols = [
        "pred_std_V_embb",
        "pred_std_V_urllc",
    ]

    for c in pred_cols:
        base = df[c].to_numpy(dtype=float)
        scale = np.std(base)
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0

        noise = rng.normal(loc=0.0, scale=noise_scale * scale, size=len(df))
        df[f"pert_{c}"] = np.clip(base + noise, 0.0, None)

    for c in std_cols:
        base = df[c].to_numpy(dtype=float)
        mult_noise = rng.normal(loc=0.0, scale=0.25 * noise_scale, size=len(df))
        pert = base * (1.0 + mult_noise)
        df[f"pert_{c}"] = np.clip(pert, 1e-6, None)

    return df


def add_inflated_certificate(df, alpha):
    out = df.copy()

    out["inflated_upper_V_embb"] = (
        out["pert_pred_mean_V_embb"]
        + alpha * Q90_EMBB * np.maximum(out["pert_pred_std_V_embb"], 1e-4)
    )

    out["inflated_upper_V_urllc"] = (
        out["pert_pred_mean_V_urllc"]
        + alpha * Q90_URLLC * np.maximum(out["pert_pred_std_V_urllc"], 1e-4)
    )

    out["safe_inflated_cert"] = (
        (out["inflated_upper_V_embb"] <= EPS_EMBB)
        & (out["inflated_upper_V_urllc"] <= EPS_URLLC)
    ).astype(int)

    return out


def shield_score(candidate_row, raw_cand):
    upper_risk = (
        0.5 * float(candidate_row["inflated_upper_V_embb"])
        + 0.5 * float(candidate_row["inflated_upper_V_urllc"])
    )

    distance = (
        abs(float(candidate_row["candidate_embb_slice_prb"]) - float(raw_cand["candidate_embb_slice_prb"]))
        + abs(float(candidate_row["candidate_urllc_slice_prb"]) - float(raw_cand["candidate_urllc_slice_prb"]))
    )

    utility = float(candidate_row["pert_pred_mean_management_utility"])

    return utility - LAMBDA_UPPER_RISK * upper_risk - LAMBDA_DISTANCE * distance


def apply_inflated_shield(group, raw_row):
    raw_candidate_id = int(raw_row["raw_candidate_id"])
    raw_match = group[group["candidate_id"] == raw_candidate_id]

    if len(raw_match) != 1:
        raise RuntimeError(
            f"Cannot find raw candidate_id={raw_candidate_id} "
            f"in group={raw_row['candidate_group_id']}"
        )

    raw_cand = raw_match.iloc[0]

    if int(raw_cand["safe_inflated_cert"]) == 1:
        return raw_cand, "pass"

    safe_pool = group[group["safe_inflated_cert"] == 1].copy()

    if len(safe_pool) > 0:
        scores = safe_pool.apply(lambda r: shield_score(r, raw_cand), axis=1)
        return safe_pool.loc[scores.idxmax()], "shield_to_safe"

    # Conservative fallback: least inflated upper risk.
    upper_risk = 0.5 * group["inflated_upper_V_embb"] + 0.5 * group["inflated_upper_V_urllc"]
    return group.loc[upper_risk.idxmin()], "fallback_no_safe"


def run_one_setting(candidates, raw_actions, noise_scale, alpha):
    cand_groups = {
        int(gid): group.copy()
        for gid, group in candidates.groupby("candidate_group_id", sort=False)
    }

    rows = []

    for _, raw in raw_actions.iterrows():
        gid = int(raw["candidate_group_id"])
        group = cand_groups.get(gid)

        if group is None:
            raise RuntimeError(f"Missing candidate group {gid}")

        shielded, mode = apply_inflated_shield(group, raw)

        raw_candidate_id = int(raw["raw_candidate_id"])
        raw_cand = group[group["candidate_id"] == raw_candidate_id].iloc[0]

        rows.append({
            "noise_scale": float(noise_scale),
            "alpha": float(alpha),
            "controller": raw["controller"],
            "candidate_group_id": gid,

            "raw_candidate_id": raw_candidate_id,
            "shield_candidate_id": int(shielded["candidate_id"]),
            "shield_mode": mode,

            "raw_true_joint_safe": int(raw_cand["true_joint_safe"]),
            "shield_true_joint_safe": int(shielded["true_joint_safe"]),

            "raw_true_management_utility": float(raw_cand["true_management_utility"]),
            "shield_true_management_utility": float(shielded["true_management_utility"]),

            "raw_true_V_embb": float(raw_cand["true_V_embb"]),
            "shield_true_V_embb": float(shielded["true_V_embb"]),

            "raw_true_V_urllc": float(raw_cand["true_V_urllc"]),
            "shield_true_V_urllc": float(shielded["true_V_urllc"]),

            "raw_embb_slice_prb": float(raw_cand["candidate_embb_slice_prb"]),
            "raw_urllc_slice_prb": float(raw_cand["candidate_urllc_slice_prb"]),

            "shield_embb_slice_prb": float(shielded["candidate_embb_slice_prb"]),
            "shield_urllc_slice_prb": float(shielded["candidate_urllc_slice_prb"]),
        })

    return pd.DataFrame(rows)


def evaluate_summary(pair_df):
    rows = []

    for (noise_scale, alpha, controller), g in pair_df.groupby(["noise_scale", "alpha", "controller"]):
        raw_safe = g["raw_true_joint_safe"].astype(int).to_numpy()
        shield_safe = g["shield_true_joint_safe"].astype(int).to_numpy()

        raw_u = g["raw_true_management_utility"].to_numpy()
        shield_u = g["shield_true_management_utility"].to_numpy()

        changed = g["raw_candidate_id"].to_numpy() != g["shield_candidate_id"].to_numpy()

        dist = (
            np.abs(g["raw_embb_slice_prb"].to_numpy() - g["shield_embb_slice_prb"].to_numpy())
            + np.abs(g["raw_urllc_slice_prb"].to_numpy() - g["shield_urllc_slice_prb"].to_numpy())
        )

        modes = g["shield_mode"].astype(str).to_numpy()

        rec = {
            "noise_scale": float(noise_scale),
            "alpha": float(alpha),
            "controller": controller,
            "num_groups": int(len(g)),

            "raw_true_unsafe_rate": float(1.0 - raw_safe.mean()),
            "shielded_true_unsafe_rate": float(1.0 - shield_safe.mean()),
            "utility_raw_mean": float(raw_u.mean()),
            "utility_shielded_mean": float(shield_u.mean()),
            "utility_retention": float(np.mean(shield_u / (raw_u + 1e-9))),

            "action_changed_rate": float(changed.mean()),
            "mean_action_distance": float(dist.mean()),
            "p95_action_distance": float(np.quantile(dist, 0.95)),

            "pass_through_rate": float(np.mean(modes == "pass")),
            "shield_to_safe_rate": float(np.mean(modes == "shield_to_safe")),
            "fallback_rate": float(np.mean(modes == "fallback_no_safe")),
        }

        rec["unsafe_reduction_abs"] = rec["raw_true_unsafe_rate"] - rec["shielded_true_unsafe_rate"]
        if rec["raw_true_unsafe_rate"] > 1e-12:
            rec["unsafe_reduction_rel"] = rec["unsafe_reduction_abs"] / rec["raw_true_unsafe_rate"]
        else:
            rec["unsafe_reduction_rel"] = 0.0

        rows.append(rec)

    return pd.DataFrame(rows)


def evaluate_overall(summary):
    rows = []

    for (noise_scale, alpha), g in summary.groupby(["noise_scale", "alpha"]):
        risky = g[g["raw_true_unsafe_rate"] > 0.01]

        rec = {
            "noise_scale": float(noise_scale),
            "alpha": float(alpha),
            "num_controllers": int(len(g)),
            "max_shielded_unsafe_rate": float(g["shielded_true_unsafe_rate"].max()),
            "mean_shielded_unsafe_rate": float(g["shielded_true_unsafe_rate"].mean()),
            "min_utility_retention": float(g["utility_retention"].min()),
            "mean_utility_retention": float(g["utility_retention"].mean()),
            "max_fallback_rate": float(g["fallback_rate"].max()),
            "mean_action_distance": float(g["mean_action_distance"].mean()),
            "max_action_changed_rate": float(g["action_changed_rate"].max()),
        }

        if len(risky) > 0:
            rec["mean_unsafe_reduction_rel_on_risky"] = float(risky["unsafe_reduction_rel"].mean())
            rec["min_unsafe_reduction_rel_on_risky"] = float(risky["unsafe_reduction_rel"].min())
        else:
            rec["mean_unsafe_reduction_rel_on_risky"] = 0.0
            rec["min_unsafe_reduction_rel_on_risky"] = 0.0

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(["noise_scale", "alpha"]).reset_index(drop=True)
    return out


def make_paper_table(overall):
    paper = overall.copy()

    paper["Max shielded unsafe (%)"] = 100.0 * paper["max_shielded_unsafe_rate"]
    paper["Mean shielded unsafe (%)"] = 100.0 * paper["mean_shielded_unsafe_rate"]
    paper["Min utility retention (%)"] = 100.0 * paper["min_utility_retention"]
    paper["Mean utility retention (%)"] = 100.0 * paper["mean_utility_retention"]
    paper["Max fallback (%)"] = 100.0 * paper["max_fallback_rate"]
    paper["Mean unsafe reduction on risky (%)"] = 100.0 * paper["mean_unsafe_reduction_rel_on_risky"]

    paper = paper[
        [
            "noise_scale",
            "alpha",
            "Max shielded unsafe (%)",
            "Mean shielded unsafe (%)",
            "Mean unsafe reduction on risky (%)",
            "Min utility retention (%)",
            "Mean utility retention (%)",
            "mean_action_distance",
            "Max fallback (%)",
        ]
    ].copy()

    paper = paper.rename(
        columns={
            "noise_scale": "Noise scale",
            "alpha": "Inflation alpha",
            "mean_action_distance": "Mean action distance",
        }
    )

    for c in paper.columns:
        if c not in ["Noise scale", "Inflation alpha"]:
            paper[c] = paper[c].astype(float).round(4)

    return paper


def select_recommended_alpha(overall):
    """
    Pick the smallest alpha for each noise scale that meets:
    - max shielded unsafe <= 5%
    - min utility retention >= 90%
    - max fallback <= 10%

    If none satisfies, pick alpha with lowest max unsafe, breaking ties by utility retention.
    """
    rows = []

    for ns, g in overall.groupby("noise_scale"):
        g = g.sort_values("alpha").copy()

        feasible = g[
            (g["max_shielded_unsafe_rate"] <= 0.05)
            & (g["min_utility_retention"] >= 0.90)
            & (g["max_fallback_rate"] <= 0.10)
        ]

        if len(feasible) > 0:
            chosen = feasible.iloc[0].copy()
            reason = "smallest alpha satisfying unsafe<=5%, retention>=90%, fallback<=10%"
        else:
            chosen = (
                g.sort_values(
                    ["max_shielded_unsafe_rate", "min_utility_retention"],
                    ascending=[True, False],
                )
                .iloc[0]
                .copy()
            )
            reason = "no feasible alpha; selected lowest unsafe rate with highest retention tie-break"

        rec = chosen.to_dict()
        rec["selection_reason"] = reason
        rows.append(rec)

    return pd.DataFrame(rows)


def make_quality_checks(overall, recommended):
    checks = {}

    checks["has_all_noise_alpha_grid"] = bool(
        len(overall) == len(NOISE_SCALES) * len(ALPHA_LIST)
    )

    checks["inflation_reduces_unsafe_monotonic_or_near"] = True

    # Check that within each noise scale, alpha=2.5 is safer or equal than alpha=1.0.
    for ns, g in overall.groupby("noise_scale"):
        a1 = g[g["alpha"] == 1.0]
        a25 = g[g["alpha"] == 2.5]
        if len(a1) != 1 or len(a25) != 1:
            checks["inflation_reduces_unsafe_monotonic_or_near"] = False
            break
        if float(a25["max_shielded_unsafe_rate"].iloc[0]) > float(a1["max_shielded_unsafe_rate"].iloc[0]) + 1e-12:
            checks["inflation_reduces_unsafe_monotonic_or_near"] = False
            break

    checks["recommended_alpha_exists_each_noise"] = bool(
        len(recommended) == len(NOISE_SCALES)
    )

    checks["recommended_retention_ok"] = bool(
        (recommended["min_utility_retention"] >= 0.85).all()
    )

    checks["recommended_fallback_ok"] = bool(
        (recommended["max_fallback_rate"] <= 0.20).all()
    )

    checks["overall_inflation_ready"] = bool(all(checks.values()))

    return checks


def main():
    assert_exists(CANDIDATES_FILE)

    print(f"Reading candidates: {CANDIDATES_FILE}")
    base_candidates = pd.read_csv(CANDIDATES_FILE)

    print("Loading raw actions...")
    raw_actions = load_raw_actions()

    print(f"Candidate rows: {len(base_candidates)}")
    print(f"Raw action rows: {len(raw_actions)}")
    print(f"Controllers: {sorted(raw_actions['controller'].unique().tolist())}")

    all_pairwise = []

    for ns in NOISE_SCALES:
        print(f"\n=== noise_scale={ns} ===")

        perturbed = add_perturbed_predictions(
            base_candidates,
            noise_scale=ns,
            seed=SEED + int(ns * 1000),
        )

        for alpha in ALPHA_LIST:
            print(f"Running alpha={alpha}")

            inflated = add_inflated_certificate(perturbed, alpha=alpha)
            pair = run_one_setting(inflated, raw_actions, ns, alpha)
            all_pairwise.append(pair)

    pairwise = pd.concat(all_pairwise, ignore_index=True)

    summary = evaluate_summary(pairwise)
    overall = evaluate_overall(summary)
    paper = make_paper_table(overall)
    recommended = select_recommended_alpha(overall)
    checks = make_quality_checks(overall, recommended)

    pairwise_path = RESULT_DIR / "step4_1b_certificate_inflation_pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

    recommended_path = RESULT_DIR / "step4_1b_recommended_alpha.csv"
    recommended.to_csv(recommended_path, index=False, encoding="utf-8-sig")

    report = {
        "input_files": {
            "candidates": str(CANDIDATES_FILE),
            "raw_classical": str(RAW_CLASSICAL),
            "raw_ppo_utility": str(RAW_PPO_UTILITY),
            "raw_ppo_penalty": str(RAW_PPO_PENALTY),
        },
        "noise_scales": NOISE_SCALES,
        "alpha_list": ALPHA_LIST,
        "num_candidate_rows": int(len(base_candidates)),
        "num_raw_action_rows": int(len(raw_actions)),
        "num_pairwise_rows": int(len(pairwise)),
        "controllers": sorted(raw_actions["controller"].unique().tolist()),
        "quality_checks": checks,
        "outputs": {
            "pairwise": str(pairwise_path),
            "summary": str(OUT_SUMMARY),
            "overall": str(OUT_OVERALL),
            "paper": str(OUT_PAPER),
            "recommended_alpha": str(recommended_path),
            "report_md": str(OUT_REPORT_MD),
        },
        "important_note": (
            "Step 4.1B evaluates perturbation-aware certificate inflation. "
            "The inflation factor alpha scales the calibrated q90 uncertainty term to trade utility for safety under degraded twin reliability."
        ),
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 4.1B Perturbation-Aware Certificate Inflation\n\n")

        f.write("## Setup\n\n")
        f.write(f"- noise scales: {NOISE_SCALES}\n")
        f.write(f"- alpha list: {ALPHA_LIST}\n")
        f.write(f"- candidate rows: {len(base_candidates)}\n")
        f.write(f"- raw action rows: {len(raw_actions)}\n")
        f.write(f"- pairwise rows: {len(pairwise)}\n")
        f.write(f"- controllers: {sorted(raw_actions['controller'].unique().tolist())}\n\n")

        f.write("## Overall certificate-inflation summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Recommended alpha by noise scale\n\n")
        f.write(recommended.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Quality checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Interpretation\n\n")
        f.write(
            "- Increasing alpha inflates the calibrated certificate and generally reduces unsafe decisions under perturbed twin predictions.\n"
        )
        f.write(
            "- Larger alpha may increase conservativeness, action changes, or fallback, forming a safety-utility tradeoff.\n"
        )
        f.write(
            "- The recommended alpha table can be used to describe perturbation-aware deployment tuning.\n"
        )

    print("Step 4.1B completed.")
    print(json.dumps(report, indent=2))
    print("\nOverall certificate inflation summary:")
    print(overall.to_string(index=False))
    print("\nRecommended alpha:")
    print(recommended.to_string(index=False))


if __name__ == "__main__":
    main()