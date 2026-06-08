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

RESULT_DIR = PROJECT_ROOT / "results" / "step4_2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_FILE = EVAL_DIR / "controller_candidate_groups_patched.csv"

RAW_CLASSICAL = EVAL_DIR / "raw_controller_actions_classical.csv"
RAW_PPO_UTILITY = EVAL_DIR / "step2_5b_ppo_utility_raw_actions.csv"
RAW_PPO_PENALTY = EVAL_DIR / "step2_5_ppo_raw_actions.csv"

OUT_SUMMARY = RESULT_DIR / "step4_2_sla_threshold_shift_summary.csv"
OUT_OVERALL = RESULT_DIR / "step4_2_sla_threshold_shift_overall.csv"
OUT_PAPER = RESULT_DIR / "step4_2_sla_threshold_shift_paper.csv"
OUT_REPORT_JSON = RESULT_DIR / "step4_2_sla_threshold_shift_report.json"
OUT_REPORT_MD = RESULT_DIR / "step4_2_report.md"

THRESHOLD_SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]

BASE_EPS_EMBB = 0.29489304824809814
BASE_EPS_URLLC = 0.09153290412936986

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
        "raw_true_management_utility",
    ]

    missing = [c for c in required if c not in raw_all.columns]
    if missing:
        raise RuntimeError(f"Raw actions missing columns: {missing}")

    return raw_all


def add_threshold_safe_labels(candidates, eps_embb, eps_urllc):
    df = candidates.copy()

    # True safety under shifted SLA.
    df["true_safe_shift_embb"] = (df["true_V_embb"] <= eps_embb).astype(int)
    df["true_safe_shift_urllc"] = (df["true_V_urllc"] <= eps_urllc).astype(int)
    df["true_joint_safe_shift"] = (
        (df["true_safe_shift_embb"] == 1)
        & (df["true_safe_shift_urllc"] == 1)
    ).astype(int)

    # Certified safety under shifted SLA.
    df["cert_safe_shift_embb"] = (df["upper_V_embb_q90"] <= eps_embb).astype(int)
    df["cert_safe_shift_urllc"] = (df["upper_V_urllc_q90"] <= eps_urllc).astype(int)
    df["cert_joint_safe_shift"] = (
        (df["cert_safe_shift_embb"] == 1)
        & (df["cert_safe_shift_urllc"] == 1)
    ).astype(int)

    # Raw-pred safety under shifted SLA, useful for diagnostics.
    df["raw_pred_safe_shift_embb"] = (df["pred_mean_V_embb"] <= eps_embb).astype(int)
    df["raw_pred_safe_shift_urllc"] = (df["pred_mean_V_urllc"] <= eps_urllc).astype(int)
    df["raw_pred_joint_safe_shift"] = (
        (df["raw_pred_safe_shift_embb"] == 1)
        & (df["raw_pred_safe_shift_urllc"] == 1)
    ).astype(int)

    return df


def safe_availability_summary(candidates, threshold_scale):
    g = (
        candidates.groupby("candidate_group_id")
        .agg(
            num_candidates=("candidate_id", "count"),
            num_true_safe=("true_joint_safe_shift", "sum"),
            num_cert_safe=("cert_joint_safe_shift", "sum"),
            num_raw_pred_safe=("raw_pred_joint_safe_shift", "sum"),
        )
        .reset_index()
    )

    rec = {
        "threshold_scale": float(threshold_scale),
        "num_groups": int(len(g)),
        "mean_candidates_per_group": float(g["num_candidates"].mean()),
        "true_safe_candidate_ratio": float(candidates["true_joint_safe_shift"].mean()),
        "cert_safe_candidate_ratio": float(candidates["cert_joint_safe_shift"].mean()),
        "raw_pred_safe_candidate_ratio": float(candidates["raw_pred_joint_safe_shift"].mean()),
        "groups_with_true_safe_rate": float((g["num_true_safe"] > 0).mean()),
        "groups_with_cert_safe_rate": float((g["num_cert_safe"] > 0).mean()),
        "groups_with_raw_pred_safe_rate": float((g["num_raw_pred_safe"] > 0).mean()),
        "mean_true_safe_candidates_per_group": float(g["num_true_safe"].mean()),
        "mean_cert_safe_candidates_per_group": float(g["num_cert_safe"].mean()),
        "mean_raw_pred_safe_candidates_per_group": float(g["num_raw_pred_safe"].mean()),
    }

    return rec


def shield_score(candidate_row, raw_cand):
    upper_risk = (
        0.5 * float(candidate_row["upper_V_embb_q90"])
        + 0.5 * float(candidate_row["upper_V_urllc_q90"])
    )

    distance = (
        abs(float(candidate_row["candidate_embb_slice_prb"]) - float(raw_cand["candidate_embb_slice_prb"]))
        + abs(float(candidate_row["candidate_urllc_slice_prb"]) - float(raw_cand["candidate_urllc_slice_prb"]))
    )

    utility = float(candidate_row["pred_mean_management_utility"])

    return utility - LAMBDA_UPPER_RISK * upper_risk - LAMBDA_DISTANCE * distance


def apply_shifted_shield(group, raw_row):
    raw_candidate_id = int(raw_row["raw_candidate_id"])
    raw_match = group[group["candidate_id"] == raw_candidate_id]

    if len(raw_match) != 1:
        raise RuntimeError(
            f"Cannot find raw candidate_id={raw_candidate_id} "
            f"in group={raw_row['candidate_group_id']}"
        )

    raw_cand = raw_match.iloc[0]

    # Pass through if raw action is certified safe under shifted SLA.
    if int(raw_cand["cert_joint_safe_shift"]) == 1:
        return raw_cand, "pass"

    safe_pool = group[group["cert_joint_safe_shift"] == 1].copy()

    if len(safe_pool) > 0:
        scores = safe_pool.apply(lambda r: shield_score(r, raw_cand), axis=1)
        return safe_pool.loc[scores.idxmax()], "shield_to_certified_safe"

    # Fallback when no certified-safe candidate exists under strict SLA.
    upper_risk = 0.5 * group["upper_V_embb_q90"] + 0.5 * group["upper_V_urllc_q90"]
    return group.loc[upper_risk.idxmin()], "fallback_no_certified_safe"


def run_one_scale(candidates, raw_actions, threshold_scale, eps_embb, eps_urllc):
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

        raw_candidate_id = int(raw["raw_candidate_id"])
        raw_cand = group[group["candidate_id"] == raw_candidate_id].iloc[0]

        shielded, mode = apply_shifted_shield(group, raw)

        rows.append({
            "threshold_scale": float(threshold_scale),
            "eps_embb": float(eps_embb),
            "eps_urllc": float(eps_urllc),
            "controller": raw["controller"],
            "candidate_group_id": gid,

            "raw_candidate_id": raw_candidate_id,
            "shield_candidate_id": int(shielded["candidate_id"]),
            "shield_mode": mode,

            "raw_true_joint_safe_shift": int(raw_cand["true_joint_safe_shift"]),
            "shield_true_joint_safe_shift": int(shielded["true_joint_safe_shift"]),

            "raw_cert_joint_safe_shift": int(raw_cand["cert_joint_safe_shift"]),
            "shield_cert_joint_safe_shift": int(shielded["cert_joint_safe_shift"]),

            "raw_true_management_utility": float(raw_cand["true_management_utility"]),
            "shield_true_management_utility": float(shielded["true_management_utility"]),

            "raw_true_V_embb": float(raw_cand["true_V_embb"]),
            "shield_true_V_embb": float(shielded["true_V_embb"]),

            "raw_true_V_urllc": float(raw_cand["true_V_urllc"]),
            "shield_true_V_urllc": float(shielded["true_V_urllc"]),

            "raw_true_V_total": float(raw_cand["true_V_total"]),
            "shield_true_V_total": float(shielded["true_V_total"]),

            "raw_embb_slice_prb": float(raw_cand["candidate_embb_slice_prb"]),
            "raw_urllc_slice_prb": float(raw_cand["candidate_urllc_slice_prb"]),

            "shield_embb_slice_prb": float(shielded["candidate_embb_slice_prb"]),
            "shield_urllc_slice_prb": float(shielded["candidate_urllc_slice_prb"]),
        })

    return pd.DataFrame(rows)


def evaluate_summary(pair_df):
    rows = []

    for (threshold_scale, controller), g in pair_df.groupby(["threshold_scale", "controller"]):
        raw_safe = g["raw_true_joint_safe_shift"].astype(int).to_numpy()
        shield_safe = g["shield_true_joint_safe_shift"].astype(int).to_numpy()

        raw_cert = g["raw_cert_joint_safe_shift"].astype(int).to_numpy()
        shield_cert = g["shield_cert_joint_safe_shift"].astype(int).to_numpy()

        raw_u = g["raw_true_management_utility"].to_numpy()
        shield_u = g["shield_true_management_utility"].to_numpy()

        changed = g["raw_candidate_id"].to_numpy() != g["shield_candidate_id"].to_numpy()

        dist = (
            np.abs(g["raw_embb_slice_prb"].to_numpy() - g["shield_embb_slice_prb"].to_numpy())
            + np.abs(g["raw_urllc_slice_prb"].to_numpy() - g["shield_urllc_slice_prb"].to_numpy())
        )

        modes = g["shield_mode"].astype(str).to_numpy()

        rec = {
            "threshold_scale": float(threshold_scale),
            "controller": controller,
            "num_groups": int(len(g)),

            "raw_true_unsafe_rate": float(1.0 - raw_safe.mean()),
            "shielded_true_unsafe_rate": float(1.0 - shield_safe.mean()),
            "raw_certified_unsafe_rate": float(1.0 - raw_cert.mean()),
            "shielded_certified_unsafe_rate": float(1.0 - shield_cert.mean()),

            "utility_raw_mean": float(raw_u.mean()),
            "utility_shielded_mean": float(shield_u.mean()),
            "utility_retention": float(np.mean(shield_u / (raw_u + 1e-9))),

            "action_changed_rate": float(changed.mean()),
            "mean_action_distance": float(dist.mean()),
            "p95_action_distance": float(np.quantile(dist, 0.95)),

            "pass_through_rate": float(np.mean(modes == "pass")),
            "shield_to_safe_rate": float(np.mean(modes == "shield_to_certified_safe")),
            "fallback_rate": float(np.mean(modes == "fallback_no_certified_safe")),

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

        rows.append(rec)

    return pd.DataFrame(rows)


def evaluate_overall(summary, availability_rows):
    rows = []

    availability = pd.DataFrame(availability_rows)

    for threshold_scale, g in summary.groupby("threshold_scale"):
        risky = g[g["raw_true_unsafe_rate"] > 0.01]

        rec = {
            "threshold_scale": float(threshold_scale),
            "num_controllers": int(len(g)),
            "max_raw_unsafe_rate": float(g["raw_true_unsafe_rate"].max()),
            "max_shielded_unsafe_rate": float(g["shielded_true_unsafe_rate"].max()),
            "mean_shielded_unsafe_rate": float(g["shielded_true_unsafe_rate"].mean()),
            "min_utility_retention": float(g["utility_retention"].min()),
            "mean_utility_retention": float(g["utility_retention"].mean()),
            "max_fallback_rate": float(g["fallback_rate"].max()),
            "mean_fallback_rate": float(g["fallback_rate"].mean()),
            "mean_action_distance": float(g["mean_action_distance"].mean()),
            "max_action_changed_rate": float(g["action_changed_rate"].max()),
        }

        if len(risky) > 0:
            rec["mean_unsafe_reduction_rel_on_risky"] = float(risky["unsafe_reduction_rel"].mean())
            rec["min_unsafe_reduction_rel_on_risky"] = float(risky["unsafe_reduction_rel"].min())
        else:
            rec["mean_unsafe_reduction_rel_on_risky"] = 0.0
            rec["min_unsafe_reduction_rel_on_risky"] = 0.0

        av = availability[availability["threshold_scale"] == threshold_scale]
        if len(av) == 1:
            for c in [
                "true_safe_candidate_ratio",
                "cert_safe_candidate_ratio",
                "groups_with_true_safe_rate",
                "groups_with_cert_safe_rate",
                "mean_true_safe_candidates_per_group",
                "mean_cert_safe_candidates_per_group",
            ]:
                rec[c] = float(av[c].iloc[0])

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values("threshold_scale").reset_index(drop=True)
    return out


def make_paper_table(overall):
    paper = overall.copy()

    paper["SLA scale"] = paper["threshold_scale"]
    paper["Groups with cert-safe (%)"] = 100.0 * paper["groups_with_cert_safe_rate"]
    paper["Cert-safe candidate ratio (%)"] = 100.0 * paper["cert_safe_candidate_ratio"]
    paper["Max raw unsafe (%)"] = 100.0 * paper["max_raw_unsafe_rate"]
    paper["Max shielded unsafe (%)"] = 100.0 * paper["max_shielded_unsafe_rate"]
    paper["Mean unsafe reduction on risky (%)"] = 100.0 * paper["mean_unsafe_reduction_rel_on_risky"]
    paper["Min utility retention (%)"] = 100.0 * paper["min_utility_retention"]
    paper["Max fallback (%)"] = 100.0 * paper["max_fallback_rate"]
    paper["Mean action distance"] = paper["mean_action_distance"]

    paper = paper[
        [
            "SLA scale",
            "Groups with cert-safe (%)",
            "Cert-safe candidate ratio (%)",
            "Max raw unsafe (%)",
            "Max shielded unsafe (%)",
            "Mean unsafe reduction on risky (%)",
            "Min utility retention (%)",
            "Max fallback (%)",
            "Mean action distance",
        ]
    ].copy()

    for c in paper.columns:
        if c != "SLA scale":
            paper[c] = paper[c].astype(float).round(4)

    return paper


def quality_checks(overall):
    checks = {}

    checks["has_all_threshold_scales"] = bool(
        set(overall["threshold_scale"].round(4).tolist()) == set(THRESHOLD_SCALES)
    )

    # Default SLA should recover the known strong result.
    default = overall[overall["threshold_scale"] == 1.0]
    if len(default) == 1:
        default = default.iloc[0]
        checks["default_scale_safe_ok"] = bool(default["max_shielded_unsafe_rate"] <= 0.001)
        checks["default_scale_retention_ok"] = bool(default["min_utility_retention"] >= 0.90)
        checks["default_scale_fallback_ok"] = bool(default["max_fallback_rate"] <= 0.01)
    else:
        checks["default_scale_safe_ok"] = False
        checks["default_scale_retention_ok"] = False
        checks["default_scale_fallback_ok"] = False

    # Across shifted thresholds, utility retention should remain reasonable.
    checks["retention_shift_ok"] = bool((overall["min_utility_retention"] >= 0.80).all())

    # Fallback may rise under strict SLA, but should not dominate.
    checks["fallback_shift_ok"] = bool((overall["max_fallback_rate"] <= 0.30).all())

    # Shield should reduce unsafe for risky controllers.
    checks["unsafe_reduction_shift_ok"] = bool(
        (overall["mean_unsafe_reduction_rel_on_risky"] >= 0.80).all()
    )

    checks["overall_sla_shift_ready"] = bool(all(checks.values()))

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

    pairwise_list = []
    availability_rows = []

    for scale in THRESHOLD_SCALES:
        eps_embb = BASE_EPS_EMBB * scale
        eps_urllc = BASE_EPS_URLLC * scale

        print(f"\n=== threshold_scale={scale}, eps_embb={eps_embb}, eps_urllc={eps_urllc} ===")

        candidates = add_threshold_safe_labels(base_candidates, eps_embb, eps_urllc)
        availability_rows.append(safe_availability_summary(candidates, scale))

        pair = run_one_scale(candidates, raw_actions, scale, eps_embb, eps_urllc)
        pairwise_list.append(pair)

    pairwise = pd.concat(pairwise_list, ignore_index=True)
    summary = evaluate_summary(pairwise)
    overall = evaluate_overall(summary, availability_rows)
    paper = make_paper_table(overall)
    checks = quality_checks(overall)

    pairwise_path = RESULT_DIR / "step4_2_sla_threshold_shift_pairwise.csv"
    availability_path = RESULT_DIR / "step4_2_sla_threshold_shift_availability.csv"

    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(availability_rows).to_csv(availability_path, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

    report = {
        "input_files": {
            "candidates": str(CANDIDATES_FILE),
            "raw_classical": str(RAW_CLASSICAL),
            "raw_ppo_utility": str(RAW_PPO_UTILITY),
            "raw_ppo_penalty": str(RAW_PPO_PENALTY),
        },
        "threshold_scales": THRESHOLD_SCALES,
        "base_eps": {
            "eps_embb": BASE_EPS_EMBB,
            "eps_urllc": BASE_EPS_URLLC,
        },
        "num_candidate_rows": int(len(base_candidates)),
        "num_raw_action_rows": int(len(raw_actions)),
        "num_pairwise_rows": int(len(pairwise)),
        "controllers": sorted(raw_actions["controller"].unique().tolist()),
        "quality_checks": checks,
        "outputs": {
            "pairwise": str(pairwise_path),
            "availability": str(availability_path),
            "summary": str(OUT_SUMMARY),
            "overall": str(OUT_OVERALL),
            "paper": str(OUT_PAPER),
            "report_md": str(OUT_REPORT_MD),
        },
        "important_note": (
            "Step 4.2 evaluates robustness under shifted SLA thresholds. "
            "Threshold scale <1.0 indicates stricter SLA; scale >1.0 indicates looser SLA."
        ),
    }

    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 4.2 SLA Threshold Shift Robustness\n\n")

        f.write("## Setup\n\n")
        f.write(f"- threshold scales: {THRESHOLD_SCALES}\n")
        f.write(f"- base eps_embb: {BASE_EPS_EMBB}\n")
        f.write(f"- base eps_urllc: {BASE_EPS_URLLC}\n")
        f.write(f"- candidate rows: {len(base_candidates)}\n")
        f.write(f"- raw action rows: {len(raw_actions)}\n")
        f.write(f"- pairwise rows: {len(pairwise)}\n")
        f.write(f"- controllers: {sorted(raw_actions['controller'].unique().tolist())}\n\n")

        f.write("## Paper robustness table\n\n")
        f.write(paper.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Overall summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Quality checks\n\n")
        for k, v in checks.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n## Output files\n\n")
        for k, v in report["outputs"].items():
            f.write(f"- {k}: `{v}`\n")

        f.write("\n## Interpretation\n\n")
        f.write("- Strict SLA thresholds reduce certified-safe candidate availability and may increase fallback.\n")
        f.write("- Loose SLA thresholds increase pass-through and reduce intervention.\n")
        f.write("- CertiTwin should retain strong unsafe reduction around the default operating point while exposing the expected safety-availability tradeoff under strict thresholds.\n")

    print("Step 4.2 completed.")
    print(json.dumps(report, indent=2))
    print("\nPaper table:")
    print(paper.to_string(index=False))
    print("\nOverall summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()