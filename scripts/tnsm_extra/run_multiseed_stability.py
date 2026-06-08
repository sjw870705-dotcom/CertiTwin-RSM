import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CANDIDATES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "commercial_twin"
    / "controller_eval"
    / "controller_candidate_groups_patched.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "tnsm_extra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DETAIL = OUT_DIR / "table_tnsm_2_3_multiseed_detail.csv"
OUT_SUMMARY = OUT_DIR / "table_tnsm_2_3_multiseed_summary.csv"
OUT_PAPER = OUT_DIR / "table_tnsm_2_3_multiseed_paper.csv"
OUT_REPORT = OUT_DIR / "tnsm_2_3_multiseed_report.json"

EPS = 1e-12

SEEDS = [0, 1, 2, 3, 4]

# Candidate sampling sizes used to stress test stability.
# K=5 is the real replay setting. If groups contain only 5 candidates, K=5 is deterministic.
# K=3 is a stricter subsampling stress test; it helps evaluate sensitivity to candidate availability.
K_SETTINGS = [5, 3]

CONTROLLER_SCORE_COLUMNS = {
    "PF": "pf_like_score",
    "Greedy-SLA": "greedy_sla_score",
    "DT-Top1": "pred_mean_management_utility",
}

METHODS = [
    "Full-CertiTwin-q90",
    "SLA-Rule-Shield",
    "Fixed-UCB-1sigma-Shield",
    "Fixed-UCB-1.64sigma-Shield",
]


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + "\n".join(df.columns)
        )


def load_candidates():
    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CANDIDATES}")

    print(f"Reading candidates: {INPUT_CANDIDATES}")
    df = pd.read_csv(INPUT_CANDIDATES)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "split",
        "candidate_group_id",
        "candidate_id",
        "candidate_embb_slice_prb",
        "candidate_urllc_slice_prb",
        "true_V_embb",
        "pred_mean_V_embb",
        "pred_std_V_embb",
        "true_V_urllc",
        "pred_mean_V_urllc",
        "pred_std_V_urllc",
        "true_management_utility",
        "pred_mean_management_utility",
        "upper_V_embb_q90",
        "upper_V_urllc_q90",
        "true_joint_safe",
        "raw_pred_joint_safe",
        "cert_joint_safe",
    ]

    for c in CONTROLLER_SCORE_COLUMNS.values():
        required.append(c)

    require_columns(df, required)

    if "test" in set(df["split"].astype(str).str.lower()):
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        print(f"Using split=test, rows={len(df)}")
    else:
        print("WARNING: split=test not found. Using all rows.")

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce")
    df = df.dropna(subset=["candidate_id"]).copy()
    df["candidate_id"] = df["candidate_id"].astype(int)

    for c in ["true_joint_safe", "raw_pred_joint_safe", "cert_joint_safe"]:
        df[c] = df[c].astype(int)

    df["pred_mean_V_total_recomputed"] = (
        df["pred_mean_V_embb"].astype(float) + df["pred_mean_V_urllc"].astype(float)
    )

    df["fixed_ucb_1_embb"] = (
        df["pred_mean_V_embb"].astype(float) + 1.0 * df["pred_std_V_embb"].astype(float)
    )
    df["fixed_ucb_1_urllc"] = (
        df["pred_mean_V_urllc"].astype(float) + 1.0 * df["pred_std_V_urllc"].astype(float)
    )
    df["fixed_ucb_164_embb"] = (
        df["pred_mean_V_embb"].astype(float) + 1.64 * df["pred_std_V_embb"].astype(float)
    )
    df["fixed_ucb_164_urllc"] = (
        df["pred_mean_V_urllc"].astype(float) + 1.64 * df["pred_std_V_urllc"].astype(float)
    )

    df["fixed_ucb_1_total"] = df["fixed_ucb_1_embb"] + df["fixed_ucb_1_urllc"]
    df["fixed_ucb_164_total"] = df["fixed_ucb_164_embb"] + df["fixed_ucb_164_urllc"]
    df["cert_q90_total"] = df["upper_V_embb_q90"].astype(float) + df["upper_V_urllc_q90"].astype(float)

    # Infer SLA thresholds from true safe candidates.
    safe_df = df[df["true_joint_safe"] == 1]
    if len(safe_df) == 0:
        raise RuntimeError("No true safe rows found. Cannot infer SLA thresholds.")

    eps_embb = float(safe_df["true_V_embb"].max())
    eps_urllc = float(safe_df["true_V_urllc"].max())

    df["fixed_ucb_1_joint_safe"] = (
        (df["fixed_ucb_1_embb"] <= eps_embb) & (df["fixed_ucb_1_urllc"] <= eps_urllc)
    ).astype(int)

    df["fixed_ucb_164_joint_safe"] = (
        (df["fixed_ucb_164_embb"] <= eps_embb) & (df["fixed_ucb_164_urllc"] <= eps_urllc)
    ).astype(int)

    return df, eps_embb, eps_urllc


def action_distance(row, raw_row):
    return (
        abs(float(row["candidate_embb_slice_prb"]) - float(raw_row["candidate_embb_slice_prb"]))
        + abs(float(row["candidate_urllc_slice_prb"]) - float(raw_row["candidate_urllc_slice_prb"]))
    )


def select_raw_candidate(group, controller, rng):
    score_col = CONTROLLER_SCORE_COLUMNS[controller]

    # Add tiny seed-dependent jitter only for tie-breaking stability tests.
    g = group.copy()
    jitter = rng.normal(0.0, 1e-9, size=len(g))
    g["_score_jittered"] = g[score_col].astype(float).to_numpy() + jitter

    return g.sort_values(["_score_jittered", "candidate_id"], ascending=[False, True]).iloc[0]


def select_candidate_subset(group, raw_row, k, rng):
    """
    For K=5, use all candidates in the real replay setting.
    For K<available candidates, keep raw action and sample the remaining candidates.
    This produces multi-seed variation without changing the original labels.
    """
    group = group.sort_values("candidate_id").copy()
    n = len(group)

    if k >= n:
        return group.copy()

    raw_cid = int(raw_row["candidate_id"])
    raw_part = group[group["candidate_id"].astype(int) == raw_cid].copy()
    rest = group[group["candidate_id"].astype(int) != raw_cid].copy()

    sample_n = max(0, k - len(raw_part))
    if sample_n > 0:
        sampled_idx = rng.choice(rest.index.to_numpy(), size=sample_n, replace=False)
        sampled = rest.loc[sampled_idx].copy()
        out = pd.concat([raw_part, sampled], ignore_index=True)
    else:
        out = raw_part.copy()

    return out.sort_values("candidate_id").copy()


def choose_min_risk(group, risk_col):
    return group.sort_values([risk_col, "candidate_id"], ascending=[True, True]).iloc[0]


def choose_by_score(candidates, raw_row, risk_col, rng, lambda_r=0.05, lambda_d=0.01):
    if len(candidates) == 0:
        return None

    best_idx = None
    best_score = -1e100

    for idx, row in candidates.iterrows():
        utility = float(row["pred_mean_management_utility"])
        risk = float(row[risk_col])
        dist = action_distance(row, raw_row)
        # Tiny jitter only for tie-breaking.
        score = utility - lambda_r * risk - lambda_d * dist + rng.normal(0.0, 1e-9)

        if score > best_score:
            best_score = score
            best_idx = idx

    return candidates.loc[best_idx]


def evaluate_group(group, controller, method, k_setting, rng):
    raw_full = select_raw_candidate(group, controller, rng)
    subset = select_candidate_subset(group, raw_full, k_setting, rng)

    raw = subset[subset["candidate_id"].astype(int) == int(raw_full["candidate_id"])].iloc[0]

    if method == "SLA-Rule-Shield":
        if int(raw["raw_pred_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = subset[subset["raw_pred_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_min_risk(safe_candidates, "pred_mean_V_total_recomputed")
                fallback = 0
            else:
                exe = choose_min_risk(subset, "pred_mean_V_total_recomputed")
                fallback = 1

        exe_pred_safe = int(exe["raw_pred_joint_safe"])

    elif method == "Fixed-UCB-1sigma-Shield":
        if int(raw["fixed_ucb_1_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = subset[subset["fixed_ucb_1_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "fixed_ucb_1_total", rng)
                fallback = 0
            else:
                exe = choose_min_risk(subset, "fixed_ucb_1_total")
                fallback = 1

        exe_pred_safe = int(exe["fixed_ucb_1_joint_safe"])

    elif method == "Fixed-UCB-1.64sigma-Shield":
        if int(raw["fixed_ucb_164_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = subset[subset["fixed_ucb_164_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "fixed_ucb_164_total", rng)
                fallback = 0
            else:
                exe = choose_min_risk(subset, "fixed_ucb_164_total")
                fallback = 1

        exe_pred_safe = int(exe["fixed_ucb_164_joint_safe"])

    elif method == "Full-CertiTwin-q90":
        if int(raw["cert_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = subset[subset["cert_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "cert_q90_total", rng)
                fallback = 0
            else:
                exe = choose_min_risk(subset, "cert_q90_total")
                fallback = 1

        exe_pred_safe = int(exe["cert_joint_safe"])

    else:
        raise ValueError(f"Unknown method: {method}")

    raw_true_safe = int(raw["true_joint_safe"])
    exe_true_safe = int(exe["true_joint_safe"])

    raw_utility = float(raw["true_management_utility"])
    exe_utility = float(exe["true_management_utility"])

    dist = action_distance(exe, raw)

    return {
        "raw_candidate_id": int(raw["candidate_id"]),
        "exec_candidate_id": int(exe["candidate_id"]),
        "effective_k": int(len(subset)),
        "raw_unsafe": 1.0 - raw_true_safe,
        "exec_unsafe": 1.0 - exe_true_safe,
        "raw_utility": raw_utility,
        "exec_utility": exe_utility,
        "utility_retention": exe_utility / (raw_utility + EPS),
        "action_changed": float(dist > EPS),
        "action_distance": dist,
        "fallback": float(fallback),
        "false_safe": float((exe_pred_safe == 1) and (exe_true_safe == 0)),
        "safe_candidate_ratio": float(subset["cert_joint_safe"].mean()),
    }


def run_multiseed(df):
    detail_rows = []
    group_list = list(df.groupby("candidate_group_id"))

    for seed in SEEDS:
        print(f"Running seed={seed} ...")
        rng = np.random.default_rng(seed)

        # Seed-dependent group order to avoid hidden ordering artifacts.
        order = rng.permutation(len(group_list))
        shuffled_groups = [group_list[i] for i in order]

        for k_setting in K_SETTINGS:
            for gid, group in shuffled_groups:
                group = group.sort_values("candidate_id").copy()

                for controller in CONTROLLER_SCORE_COLUMNS.keys():
                    for method in METHODS:
                        result = evaluate_group(group, controller, method, k_setting, rng)
                        result.update(
                            {
                                "seed": seed,
                                "k_setting": k_setting,
                                "candidate_group_id": gid,
                                "controller": controller,
                                "method": method,
                                "num_candidates_original": int(len(group)),
                            }
                        )
                        detail_rows.append(result)

    return pd.DataFrame(detail_rows)


def summarize_seed_level(detail):
    """
    First summarize by seed/controller/method/k, then compute mean±std across seeds.
    """
    seed_rows = []

    for (seed, k_setting, controller, method), g in detail.groupby(["seed", "k_setting", "controller", "method"]):
        raw_unsafe = float(g["raw_unsafe"].mean())
        shielded_unsafe = float(g["exec_unsafe"].mean())
        unsafe_reduction = (raw_unsafe - shielded_unsafe) / (raw_unsafe + EPS)

        seed_rows.append(
            {
                "seed": int(seed),
                "k_setting": int(k_setting),
                "controller": controller,
                "method": method,
                "num_groups": int(len(g)),
                "mean_effective_k": float(g["effective_k"].mean()),
                "raw_unsafe_rate": raw_unsafe,
                "shielded_unsafe_rate": shielded_unsafe,
                "unsafe_reduction_rel": unsafe_reduction,
                "utility_retention_mean": float(g["utility_retention"].mean()),
                "action_changed_rate": float(g["action_changed"].mean()),
                "mean_action_distance": float(g["action_distance"].mean()),
                "fallback_rate": float(g["fallback"].mean()),
                "false_safe_rate": float(g["false_safe"].mean()),
                "safe_candidate_ratio": float(g["safe_candidate_ratio"].mean()),
            }
        )

    seed_summary = pd.DataFrame(seed_rows)

    final_rows = []
    metric_cols = [
        "raw_unsafe_rate",
        "shielded_unsafe_rate",
        "unsafe_reduction_rel",
        "utility_retention_mean",
        "action_changed_rate",
        "mean_action_distance",
        "fallback_rate",
        "false_safe_rate",
        "safe_candidate_ratio",
        "mean_effective_k",
    ]

    for (k_setting, controller, method), g in seed_summary.groupby(["k_setting", "controller", "method"]):
        row = {
            "k_setting": int(k_setting),
            "controller": controller,
            "method": method,
            "num_seeds": int(g["seed"].nunique()),
            "num_groups_per_seed": int(g["num_groups"].iloc[0]),
        }

        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0

        final_rows.append(row)

    summary = pd.DataFrame(final_rows)

    controller_order = {"PF": 0, "Greedy-SLA": 1, "DT-Top1": 2}
    method_order = {
        "Full-CertiTwin-q90": 0,
        "SLA-Rule-Shield": 1,
        "Fixed-UCB-1sigma-Shield": 2,
        "Fixed-UCB-1.64sigma-Shield": 3,
    }

    summary["_c_order"] = summary["controller"].map(controller_order)
    summary["_m_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(["k_setting", "_c_order", "_m_order"]).drop(columns=["_c_order", "_m_order"])

    return seed_summary, summary


def fmt_mean_std(mean, std, scale=1.0, digits=4):
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def make_paper_table(summary):
    """
    Main paper table: focus on Full-CertiTwin-q90 at real K=5 and stress-test K=3.
    Other method details remain in summary CSV.
    """
    rows = []

    sub = summary[summary["method"] == "Full-CertiTwin-q90"].copy()

    for _, r in sub.iterrows():
        rows.append(
            {
                "K setting": int(r["k_setting"]),
                "Controller": r["controller"],
                "Raw unsafe (%)": fmt_mean_std(
                    r["raw_unsafe_rate_mean"], r["raw_unsafe_rate_std"], scale=100.0
                ),
                "Shielded unsafe (%)": fmt_mean_std(
                    r["shielded_unsafe_rate_mean"], r["shielded_unsafe_rate_std"], scale=100.0
                ),
                "Utility retention (%)": fmt_mean_std(
                    r["utility_retention_mean_mean"], r["utility_retention_mean_std"], scale=100.0
                ),
                "Mean distance": fmt_mean_std(
                    r["mean_action_distance_mean"], r["mean_action_distance_std"], scale=1.0
                ),
                "Fallback (%)": fmt_mean_std(
                    r["fallback_rate_mean"], r["fallback_rate_std"], scale=100.0
                ),
                "False-safe (%)": fmt_mean_std(
                    r["false_safe_rate_mean"], r["false_safe_rate_std"], scale=100.0
                ),
            }
        )

    paper = pd.DataFrame(rows)
    return paper


def main():
    df, eps_embb, eps_urllc = load_candidates()

    print(f"Rows used: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")
    print(f"Seeds: {SEEDS}")
    print(f"K settings: {K_SETTINGS}")
    print(f"Inferred SLA thresholds: eps_embb={eps_embb:.8f}, eps_urllc={eps_urllc:.8f}")

    detail = run_multiseed(df)
    seed_summary, summary = summarize_seed_level(detail)
    paper = make_paper_table(summary)

    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

    # Extract compact key findings for report.
    full_real = summary[
        (summary["method"] == "Full-CertiTwin-q90")
        & (summary["k_setting"] == 5)
    ]

    report = {
        "status": "completed",
        "input": str(INPUT_CANDIDATES),
        "outputs": {
            "detail": str(OUT_DETAIL),
            "summary": str(OUT_SUMMARY),
            "paper": str(OUT_PAPER),
        },
        "rows_used": int(len(df)),
        "candidate_groups": int(df["candidate_group_id"].nunique()),
        "seeds": SEEDS,
        "k_settings": K_SETTINGS,
        "controllers": list(CONTROLLER_SCORE_COLUMNS.keys()),
        "methods": METHODS,
        "sla_thresholds_inferred_from_true_safe_rows": {
            "eps_embb": eps_embb,
            "eps_urllc": eps_urllc,
        },
        "full_certitwin_real_k5_key_metrics": full_real[
            [
                "controller",
                "shielded_unsafe_rate_mean",
                "shielded_unsafe_rate_std",
                "utility_retention_mean_mean",
                "utility_retention_mean_std",
                "fallback_rate_mean",
                "fallback_rate_std",
                "false_safe_rate_mean",
                "false_safe_rate_std",
            ]
        ].to_dict(orient="records"),
        "important_note": (
            "This is evaluation-level multi-seed stability. It varies candidate subset sampling, "
            "tie-breaking jitter, and evaluation order. It does not retrain the digital twin or PPO. "
            "Use it as stability evidence for the runtime shielding procedure, not as a full training "
            "multi-seed retraining study."
        ),
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTNSM-2.3 completed.")
    print("\nPaper table:")
    print(paper.to_string(index=False))
    print(f"\nSaved:\n{OUT_DETAIL}\n{OUT_SUMMARY}\n{OUT_PAPER}\n{OUT_REPORT}")


if __name__ == "__main__":
    main()