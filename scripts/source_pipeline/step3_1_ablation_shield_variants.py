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

RESULT_DIR = PROJECT_ROOT / "results" / "step3_1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_FILE = EVAL_DIR / "controller_candidate_groups_patched.csv"

RAW_CLASSICAL = EVAL_DIR / "raw_controller_actions_classical.csv"
RAW_PPO_UTILITY = EVAL_DIR / "step2_5b_ppo_utility_raw_actions.csv"
RAW_PPO_PENALTY = EVAL_DIR / "step2_5_ppo_raw_actions.csv"

OUT_PAIRWISE = RESULT_DIR / "step3_1_ablation_pairwise.csv"
OUT_SUMMARY = RESULT_DIR / "step3_1_ablation_summary.csv"
OUT_REPORT_JSON = RESULT_DIR / "step3_1_ablation_report.json"
OUT_REPORT_MD = RESULT_DIR / "step3_1_report.md"

Q_LEVEL = 90

EPS_EMBB = 0.29489304824809814
EPS_URLLC = 0.09153290412936986

LAMBDA_DISTANCE = 0.02
LAMBDA_UPPER_RISK = 0.50

ABLATION_VARIANTS = [
    "Full-CertiTwin",
    "Raw-Pred-Safe",
    "Uncalibrated-Upper",
    "No-Distance-Projection",
    "Risk-Min-Projection",
]


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_raw_actions():
    for p in [RAW_CLASSICAL, RAW_PPO_UTILITY, RAW_PPO_PENALTY]:
        assert_exists(p)

    classical = pd.read_csv(RAW_CLASSICAL)
    ppo_utility = pd.read_csv(RAW_PPO_UTILITY)
    ppo_penalty = pd.read_csv(RAW_PPO_PENALTY)

    # Normalize PPO names.
    ppo_utility = ppo_utility.copy()
    ppo_utility["controller"] = "PPO"

    ppo_penalty = ppo_penalty.copy()
    ppo_penalty["controller"] = "PPO-Penalty"

    raw_all = pd.concat([classical, ppo_utility, ppo_penalty], ignore_index=True)

    required = [
        "controller",
        "candidate_group_id",
        "raw_candidate_id",
        "raw_true_joint_safe",
        "raw_cert_joint_safe",
        "raw_true_management_utility",
    ]

    missing = [c for c in required if c not in raw_all.columns]
    if missing:
        raise RuntimeError(f"Raw actions missing columns: {missing}")

    return raw_all


def add_variant_safe_labels(candidates):
    df = candidates.copy()

    # Full calibrated certified set.
    df["safe_full_certitwin"] = (
        (df["upper_V_embb_q90"] <= EPS_EMBB)
        & (df["upper_V_urllc_q90"] <= EPS_URLLC)
    ).astype(int)

    # Raw predicted mean only.
    df["safe_raw_pred"] = (
        (df["pred_mean_V_embb"] <= EPS_EMBB)
        & (df["pred_mean_V_urllc"] <= EPS_URLLC)
    ).astype(int)

    # Uncalibrated upper: mean + 1.0 * sigma.
    # If pred_std columns are unavailable, fail loudly.
    required_sigma = ["pred_std_V_embb", "pred_std_V_urllc"]
    missing_sigma = [c for c in required_sigma if c not in df.columns]
    if missing_sigma:
        raise RuntimeError(f"Missing sigma columns for Uncalibrated-Upper: {missing_sigma}")

    df["uncal_upper_V_embb"] = df["pred_mean_V_embb"] + df["pred_std_V_embb"]
    df["uncal_upper_V_urllc"] = df["pred_mean_V_urllc"] + df["pred_std_V_urllc"]

    df["safe_uncalibrated_upper"] = (
        (df["uncal_upper_V_embb"] <= EPS_EMBB)
        & (df["uncal_upper_V_urllc"] <= EPS_URLLC)
    ).astype(int)

    return df


def get_safe_col(variant):
    if variant == "Full-CertiTwin":
        return "safe_full_certitwin"
    if variant == "Raw-Pred-Safe":
        return "safe_raw_pred"
    if variant == "Uncalibrated-Upper":
        return "safe_uncalibrated_upper"
    if variant in ["No-Distance-Projection", "Risk-Min-Projection"]:
        return "safe_full_certitwin"
    raise ValueError(f"Unknown variant: {variant}")


def shield_score(candidate_row, raw_cand, variant):
    upper_risk = (
        0.5 * float(candidate_row["upper_V_embb_q90"])
        + 0.5 * float(candidate_row["upper_V_urllc_q90"])
    )

    distance = (
        abs(float(candidate_row["candidate_embb_slice_prb"]) - float(raw_cand["candidate_embb_slice_prb"]))
        + abs(float(candidate_row["candidate_urllc_slice_prb"]) - float(raw_cand["candidate_urllc_slice_prb"]))
    )

    utility = float(candidate_row["pred_mean_management_utility"])

    if variant == "Full-CertiTwin":
        return utility - LAMBDA_UPPER_RISK * upper_risk - LAMBDA_DISTANCE * distance

    if variant == "Raw-Pred-Safe":
        # Same projection style, but using raw predicted safe set.
        return utility - LAMBDA_UPPER_RISK * upper_risk - LAMBDA_DISTANCE * distance

    if variant == "Uncalibrated-Upper":
        # Same projection style, but using uncalibrated upper safe set.
        return utility - LAMBDA_UPPER_RISK * upper_risk - LAMBDA_DISTANCE * distance

    if variant == "No-Distance-Projection":
        return utility - LAMBDA_UPPER_RISK * upper_risk

    if variant == "Risk-Min-Projection":
        return -upper_risk

    raise ValueError(f"Unknown variant: {variant}")


def apply_variant_shield(group, raw_row, variant):
    raw_candidate_id = int(raw_row["raw_candidate_id"])
    raw_match = group[group["candidate_id"] == raw_candidate_id]

    if len(raw_match) != 1:
        raise RuntimeError(
            f"Cannot find raw candidate_id={raw_candidate_id} in group={raw_row['candidate_group_id']}"
        )

    raw_cand = raw_match.iloc[0]
    safe_col = get_safe_col(variant)

    # Pass-through if raw action is safe under the variant rule.
    if int(raw_cand[safe_col]) == 1:
        return raw_cand, "pass"

    safe_pool = group[group[safe_col] == 1].copy()

    if len(safe_pool) > 0:
        scores = safe_pool.apply(lambda r: shield_score(r, raw_cand, variant), axis=1)
        return safe_pool.loc[scores.idxmax()], "shield_to_safe"

    # Fallback if the variant's safe set is empty.
    # Use least calibrated upper risk for all variants to make fallback conservative.
    upper_risk = 0.5 * group["upper_V_embb_q90"] + 0.5 * group["upper_V_urllc_q90"]
    return group.loc[upper_risk.idxmin()], "fallback_no_safe"


def evaluate_pair_rows(pair_df):
    rows = []

    for (variant, controller), g in pair_df.groupby(["variant", "controller"]):
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
            "variant": variant,
            "controller": controller,
            "num_groups": int(len(g)),

            "raw_true_unsafe_rate": float(1.0 - raw_safe.mean()),
            "shielded_true_unsafe_rate": float(1.0 - shield_safe.mean()),
            "raw_true_safe_rate": float(raw_safe.mean()),
            "shielded_true_safe_rate": float(shield_safe.mean()),

            "utility_raw_mean": float(raw_u.mean()),
            "utility_shielded_mean": float(shield_u.mean()),
            "utility_retention": float(np.mean(shield_u / (raw_u + 1e-9))),

            "action_changed_rate": float(changed.mean()),
            "mean_action_distance": float(dist.mean()),
            "p95_action_distance": float(np.quantile(dist, 0.95)),

            "pass_through_rate": float(np.mean(modes == "pass")),
            "shield_to_safe_rate": float(np.mean(modes == "shield_to_safe")),
            "fallback_rate": float(np.mean(modes == "fallback_no_safe")),

            "raw_V_embb_mean": float(g["raw_true_V_embb"].mean()),
            "shielded_V_embb_mean": float(g["shield_true_V_embb"].mean()),
            "raw_V_urllc_mean": float(g["raw_true_V_urllc"].mean()),
            "shielded_V_urllc_mean": float(g["shield_true_V_urllc"].mean()),
            "raw_V_total_mean": float(g["raw_true_V_total"].mean()),
            "shielded_V_total_mean": float(g["shield_true_V_total"].mean()),
        }

        rec["unsafe_reduction_abs"] = rec["raw_true_unsafe_rate"] - rec["shielded_true_unsafe_rate"]
        if rec["raw_true_unsafe_rate"] > 1e-12:
            rec["unsafe_reduction_rel"] = rec["unsafe_reduction_abs"] / rec["raw_true_unsafe_rate"]
        else:
            rec["unsafe_reduction_rel"] = 0.0

        rec["utility_loss_abs"] = rec["utility_raw_mean"] - rec["utility_shielded_mean"]

        rows.append(rec)

    summary = pd.DataFrame(rows)

    controller_order = {
        "PF": 1,
        "Greedy-SLA": 2,
        "DT-Top1": 3,
        "PPO": 4,
        "PPO-Penalty": 5,
    }

    variant_order = {
        "Full-CertiTwin": 1,
        "Raw-Pred-Safe": 2,
        "Uncalibrated-Upper": 3,
        "No-Distance-Projection": 4,
        "Risk-Min-Projection": 5,
    }

    summary["variant_order"] = summary["variant"].map(variant_order).fillna(99)
    summary["controller_order"] = summary["controller"].map(controller_order).fillna(99)

    summary = (
        summary.sort_values(["variant_order", "controller_order"])
        .drop(columns=["variant_order", "controller_order"])
        .reset_index(drop=True)
    )

    return summary


def build_pairwise(candidates, raw_actions):
    cand_groups = {
        int(gid): group.copy()
        for gid, group in candidates.groupby("candidate_group_id", sort=False)
    }

    pair_rows = []

    for variant in ABLATION_VARIANTS:
        print(f"Running ablation variant: {variant}")

        for _, raw in raw_actions.iterrows():
            gid = int(raw["candidate_group_id"])
            group = cand_groups.get(gid)

            if group is None:
                raise RuntimeError(f"Missing candidate group {gid}")

            shielded, mode = apply_variant_shield(group, raw, variant)

            raw_candidate_id = int(raw["raw_candidate_id"])
            raw_cand = group[group["candidate_id"] == raw_candidate_id].iloc[0]

            pair_rows.append({
                "variant": variant,
                "controller": raw["controller"],
                "candidate_group_id": gid,

                "raw_candidate_id": raw_candidate_id,
                "shield_candidate_id": int(shielded["candidate_id"]),
                "shield_mode": mode,

                "raw_true_joint_safe": int(raw_cand["true_joint_safe"]),
                "shield_true_joint_safe": int(shielded["true_joint_safe"]),

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

    return pd.DataFrame(pair_rows)


def summarize_variant_overall(summary):
    rows = []

    for variant, g in summary.groupby("variant"):
        risky = g[g["raw_true_unsafe_rate"] > 0.01]

        rec = {
            "variant": variant,
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

    order = {
        "Full-CertiTwin": 1,
        "Raw-Pred-Safe": 2,
        "Uncalibrated-Upper": 3,
        "No-Distance-Projection": 4,
        "Risk-Min-Projection": 5,
    }

    out["variant_order"] = out["variant"].map(order).fillna(99)
    out = out.sort_values("variant_order").drop(columns=["variant_order"]).reset_index(drop=True)

    return out


def evaluate_ablation_quality(overall):
    checks = {}

    full = overall[overall["variant"] == "Full-CertiTwin"]
    if len(full) == 1:
        full = full.iloc[0]
        checks["full_low_shielded_unsafe"] = bool(full["max_shielded_unsafe_rate"] <= 0.001)
        checks["full_utility_retention_ok"] = bool(full["min_utility_retention"] >= 0.90)
        checks["full_no_fallback"] = bool(full["max_fallback_rate"] <= 0.001)
    else:
        checks["full_low_shielded_unsafe"] = False
        checks["full_utility_retention_ok"] = False
        checks["full_no_fallback"] = False

    # Compare raw prediction safe set with full CertiTwin.
    raw_pred = overall[overall["variant"] == "Raw-Pred-Safe"]
    if len(raw_pred) == 1 and len(overall[overall["variant"] == "Full-CertiTwin"]) == 1:
        raw_pred = raw_pred.iloc[0]
        full = overall[overall["variant"] == "Full-CertiTwin"].iloc[0]
        checks["calibration_improves_or_matches_safety"] = bool(
            full["max_shielded_unsafe_rate"] <= raw_pred["max_shielded_unsafe_rate"] + 1e-12
        )
    else:
        checks["calibration_improves_or_matches_safety"] = False

    checks["overall_ablation_ready"] = bool(all(checks.values()))

    return checks


def main():
    assert_exists(CANDIDATES_FILE)

    print(f"Reading candidates: {CANDIDATES_FILE}")
    candidates = pd.read_csv(CANDIDATES_FILE)
    candidates = add_variant_safe_labels(candidates)

    print("Loading raw actions...")
    raw_actions = load_raw_actions()

    print(f"Candidate rows: {len(candidates)}")
    print(f"Raw action rows: {len(raw_actions)}")
    print(f"Controllers: {sorted(raw_actions['controller'].unique().tolist())}")

    pairwise = build_pairwise(candidates, raw_actions)
    summary = evaluate_pair_rows(pairwise)
    overall = summarize_variant_overall(summary)
    checks = evaluate_ablation_quality(overall)

    pairwise.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    overall_path = RESULT_DIR / "step3_1_ablation_overall.csv"
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    report = {
        "input_files": {
            "candidates": str(CANDIDATES_FILE),
            "raw_classical": str(RAW_CLASSICAL),
            "raw_ppo_utility": str(RAW_PPO_UTILITY),
            "raw_ppo_penalty": str(RAW_PPO_PENALTY),
        },
        "variants": ABLATION_VARIANTS,
        "num_candidate_rows": int(len(candidates)),
        "num_raw_action_rows": int(len(raw_actions)),
        "num_pairwise_rows": int(len(pairwise)),
        "controllers": sorted(raw_actions["controller"].unique().tolist()),
        "overall_quality_checks": checks,
        "outputs": {
            "pairwise": str(OUT_PAIRWISE),
            "summary": str(OUT_SUMMARY),
            "overall": str(overall_path),
            "report_md": str(OUT_REPORT_MD),
        },
        "important_note": (
            "Step 3.1 evaluates shield variants on the same raw controller actions. "
            "This isolates whether calibrated upper bounds and projection design are necessary."
        ),
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 3.1 Core Ablation: Shield Variants\n\n")

        f.write("## Variants\n\n")
        for v in ABLATION_VARIANTS:
            f.write(f"- {v}\n")

        f.write("\n## Dataset summary\n\n")
        f.write(f"- candidate rows: {len(candidates)}\n")
        f.write(f"- raw action rows: {len(raw_actions)}\n")
        f.write(f"- pairwise rows: {len(pairwise)}\n")
        f.write(f"- controllers: {sorted(raw_actions['controller'].unique().tolist())}\n\n")

        f.write("## Overall variant summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Quality checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Interpretation\n\n")
        f.write("- Full-CertiTwin uses calibrated q90 upper bounds and distance-aware projection.\n")
        f.write("- Raw-Pred-Safe removes calibration and uses predicted mean safety only.\n")
        f.write("- Uncalibrated-Upper uses mean + sigma without conformal/calibration quantile.\n")
        f.write("- No-Distance-Projection removes the distance-to-raw-action term.\n")
        f.write("- Risk-Min-Projection ignores utility and selects the lowest certified upper risk action.\n")

    print("Step 3.1 completed.")
    print(json.dumps(report, indent=2))
    print("\nOverall ablation summary:")
    print(overall.to_string(index=False))
    print("\nDetailed ablation summary head:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()