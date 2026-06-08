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

OUT_DETAIL = OUT_DIR / "table_tnsm_2_1b_controller_rule_shield_detail.csv"
OUT_SUMMARY = OUT_DIR / "table_tnsm_2_1b_controller_rule_shield_summary.csv"
OUT_PAPER = OUT_DIR / "table_tnsm_2_1b_controller_rule_shield_paper.csv"
OUT_REPORT = OUT_DIR / "tnsm_2_1b_controller_rule_shield_report.json"

EPS = 1e-12

METHODS = [
    "Full-CertiTwin-q90",
    "SLA-Rule-Shield",
    "Fixed-UCB-1sigma-Shield",
    "Fixed-UCB-1.64sigma-Shield",
]

# Controller definitions inferred from available score columns.
# Higher score is better.
CONTROLLER_SCORE_COLUMNS = {
    "PF": "pf_like_score",
    "Greedy-SLA": "greedy_sla_score",
    "DT-Top1": "pred_mean_management_utility",
}


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
        "upper_V_total_q90",
        "true_joint_safe",
        "raw_pred_joint_safe",
        "cert_joint_safe",
    ]

    # Score columns for raw controller construction.
    for c in CONTROLLER_SCORE_COLUMNS.values():
        required.append(c)

    require_columns(df, required)

    if "test" in set(df["split"].astype(str).str.lower()):
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        print(f"Using split=test, rows={len(df)}")
    else:
        print("WARNING: test split not found. Using all rows.")

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce")
    df = df.dropna(subset=["candidate_id"]).copy()
    df["candidate_id"] = df["candidate_id"].astype(int)

    for c in ["true_joint_safe", "raw_pred_joint_safe", "cert_joint_safe"]:
        df[c] = df[c].astype(int)

    # Recomputed risks.
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

    # Infer thresholds from true safe candidates.
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


def select_raw_candidate(group, controller):
    score_col = CONTROLLER_SCORE_COLUMNS[controller]

    # Highest score wins. candidate_id is tie-breaker for reproducibility.
    return (
        group.sort_values([score_col, "candidate_id"], ascending=[False, True])
        .iloc[0]
    )


def choose_min_risk(group, risk_col):
    return group.sort_values([risk_col, "candidate_id"], ascending=[True, True]).iloc[0]


def choose_by_score(candidates, raw_row, risk_col, lambda_r=0.05, lambda_d=0.01):
    """
    CertiTwin-style utility-risk-distance projection.
    Used by Full-CertiTwin and Fixed-UCB variants.
    """
    if len(candidates) == 0:
        return None

    best_idx = None
    best_score = -1e100

    for idx, row in candidates.iterrows():
        utility = float(row["pred_mean_management_utility"])
        risk = float(row[risk_col])
        dist = action_distance(row, raw_row)
        score = utility - lambda_r * risk - lambda_d * dist

        if score > best_score:
            best_score = score
            best_idx = idx

    return candidates.loc[best_idx]


def evaluate_group(group, controller, method):
    raw = select_raw_candidate(group, controller)

    if method == "SLA-Rule-Shield":
        # Simple management rule:
        # raw predicted mean safe -> pass-through;
        # otherwise choose predicted-safe candidate with minimum predicted mean risk.
        if int(raw["raw_pred_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = group[group["raw_pred_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_min_risk(safe_candidates, "pred_mean_V_total_recomputed")
                fallback = 0
            else:
                exe = choose_min_risk(group, "pred_mean_V_total_recomputed")
                fallback = 1

        exe_pred_safe = int(exe["raw_pred_joint_safe"])

    elif method == "Fixed-UCB-1sigma-Shield":
        # Fixed UCB: mean + 1.0 sigma, no calibration split.
        if int(raw["fixed_ucb_1_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = group[group["fixed_ucb_1_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "fixed_ucb_1_total")
                fallback = 0
            else:
                exe = choose_min_risk(group, "fixed_ucb_1_total")
                fallback = 1

        exe_pred_safe = int(exe["fixed_ucb_1_joint_safe"])

    elif method == "Fixed-UCB-1.64sigma-Shield":
        # Fixed UCB: mean + 1.64 sigma, no calibration split.
        if int(raw["fixed_ucb_164_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = group[group["fixed_ucb_164_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "fixed_ucb_164_total")
                fallback = 0
            else:
                exe = choose_min_risk(group, "fixed_ucb_164_total")
                fallback = 1

        exe_pred_safe = int(exe["fixed_ucb_164_joint_safe"])

    elif method == "Full-CertiTwin-q90":
        # Calibrated q90 certificate.
        if int(raw["cert_joint_safe"]) == 1:
            exe = raw
            fallback = 0
        else:
            safe_candidates = group[group["cert_joint_safe"] == 1]
            if len(safe_candidates) > 0:
                exe = choose_by_score(safe_candidates, raw, "cert_q90_total")
                fallback = 0
            else:
                exe = choose_min_risk(group, "cert_q90_total")
                fallback = 1

        exe_pred_safe = int(exe["cert_joint_safe"])

    else:
        raise ValueError(f"Unknown method: {method}")

    raw_true_safe = int(raw["true_joint_safe"])
    exe_true_safe = int(exe["true_joint_safe"])

    raw_utility = float(raw["true_management_utility"])
    exe_utility = float(exe["true_management_utility"])

    dist = action_distance(exe, raw)
    changed = float(dist > EPS)
    false_safe = float((exe_pred_safe == 1) and (exe_true_safe == 0))

    return {
        "raw_candidate_id": int(raw["candidate_id"]),
        "exec_candidate_id": int(exe["candidate_id"]),
        "raw_true_safe": raw_true_safe,
        "exec_true_safe": exe_true_safe,
        "raw_unsafe": 1.0 - raw_true_safe,
        "exec_unsafe": 1.0 - exe_true_safe,
        "raw_utility": raw_utility,
        "exec_utility": exe_utility,
        "utility_retention": exe_utility / (raw_utility + EPS),
        "action_changed": changed,
        "action_distance": dist,
        "fallback": float(fallback),
        "false_safe": false_safe,
        "exec_pred_safe": float(exe_pred_safe),
        "raw_embb_prb": float(raw["candidate_embb_slice_prb"]),
        "raw_urllc_prb": float(raw["candidate_urllc_slice_prb"]),
        "exec_embb_prb": float(exe["candidate_embb_slice_prb"]),
        "exec_urllc_prb": float(exe["candidate_urllc_slice_prb"]),
    }


def summarize_detail(detail):
    rows = []

    for (controller, method), g in detail.groupby(["controller", "method"]):
        raw_unsafe = float(g["raw_unsafe"].mean())
        shielded_unsafe = float(g["exec_unsafe"].mean())
        unsafe_reduction = (raw_unsafe - shielded_unsafe) / (raw_unsafe + EPS)

        rows.append(
            {
                "controller": controller,
                "method": method,
                "num_groups": int(len(g)),
                "raw_unsafe_rate": raw_unsafe,
                "shielded_unsafe_rate": shielded_unsafe,
                "unsafe_reduction_rel": unsafe_reduction,
                "raw_utility_mean": float(g["raw_utility"].mean()),
                "shielded_utility_mean": float(g["exec_utility"].mean()),
                "utility_retention_mean": float(g["utility_retention"].mean()),
                "action_changed_rate": float(g["action_changed"].mean()),
                "mean_action_distance": float(g["action_distance"].mean()),
                "fallback_rate": float(g["fallback"].mean()),
                "false_safe_rate": float(g["false_safe"].mean()),
            }
        )

    summary = pd.DataFrame(rows)

    controller_order = {"PF": 0, "Greedy-SLA": 1, "DT-Top1": 2}
    method_order = {
        "Full-CertiTwin-q90": 0,
        "SLA-Rule-Shield": 1,
        "Fixed-UCB-1sigma-Shield": 2,
        "Fixed-UCB-1.64sigma-Shield": 3,
    }

    summary["_c_order"] = summary["controller"].map(controller_order)
    summary["_m_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(["_c_order", "_m_order"]).drop(columns=["_c_order", "_m_order"])
    return summary


def make_paper_table(summary):
    rows = []

    for method, g in summary.groupby("method"):
        rows.append(
            {
                "Method": method,
                "Controllers": int(g["controller"].nunique()),
                "Max shielded unsafe (%)": 100.0 * float(g["shielded_unsafe_rate"].max()),
                "Mean shielded unsafe (%)": 100.0 * float(g["shielded_unsafe_rate"].mean()),
                "Min utility retention (%)": 100.0 * float(g["utility_retention_mean"].min()),
                "Mean action distance": float(g["mean_action_distance"].mean()),
                "Max fallback (%)": 100.0 * float(g["fallback_rate"].max()),
                "Max false-safe (%)": 100.0 * float(g["false_safe_rate"].max()),
            }
        )

    paper = pd.DataFrame(rows)

    method_order = {
        "Full-CertiTwin-q90": 0,
        "SLA-Rule-Shield": 1,
        "Fixed-UCB-1sigma-Shield": 2,
        "Fixed-UCB-1.64sigma-Shield": 3,
    }
    paper["_order"] = paper["Method"].map(method_order)
    paper = paper.sort_values("_order").drop(columns=["_order"])

    for c in paper.columns:
        if c not in ["Method", "Controllers"]:
            paper[c] = paper[c].astype(float).round(4)

    return paper


def main():
    df, eps_embb, eps_urllc = load_candidates()

    print(f"Rows used: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")
    print(f"Inferred SLA thresholds: eps_embb={eps_embb:.8f}, eps_urllc={eps_urllc:.8f}")

    detail_rows = []
    controllers_used = list(CONTROLLER_SCORE_COLUMNS.keys())

    for gid, group in df.groupby("candidate_group_id"):
        group = group.sort_values("candidate_id").copy()

        for controller in controllers_used:
            for method in METHODS:
                result = evaluate_group(group, controller, method)
                result.update(
                    {
                        "controller": controller,
                        "method": method,
                        "candidate_group_id": gid,
                        "num_candidates_in_group": int(len(group)),
                        "safe_ratio_true": float(group["true_joint_safe"].mean()),
                        "safe_ratio_pred_mean": float(group["raw_pred_joint_safe"].mean()),
                        "safe_ratio_cert_q90": float(group["cert_joint_safe"].mean()),
                        "safe_ratio_fixed_ucb_1": float(group["fixed_ucb_1_joint_safe"].mean()),
                        "safe_ratio_fixed_ucb_164": float(group["fixed_ucb_164_joint_safe"].mean()),
                    }
                )
                detail_rows.append(result)

    detail = pd.DataFrame(detail_rows)
    summary = summarize_detail(detail)
    paper = make_paper_table(summary)

    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

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
        "controllers_used": controllers_used,
        "methods": METHODS,
        "sla_thresholds_inferred_from_true_safe_rows": {
            "eps_embb": eps_embb,
            "eps_urllc": eps_urllc,
        },
        "important_note": (
            "This is controller-level paired rule-shield evaluation for controllers whose "
            "raw actions can be reconstructed from available score columns: PF, Greedy-SLA, "
            "and DT-Top1. PPO and PPO-Penalty are not included here because this candidate "
            "file does not expose their raw selected actions or policy scores."
        ),
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTNSM-2.1B completed.")
    print("\nController-level summary:")
    print(summary.to_string(index=False))

    print("\nPaper table:")
    print(paper.to_string(index=False))

    print(f"\nSaved:\n{OUT_DETAIL}\n{OUT_SUMMARY}\n{OUT_PAPER}\n{OUT_REPORT}")


if __name__ == "__main__":
    main()