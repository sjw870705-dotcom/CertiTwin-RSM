import json
import time
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

OUT_DETAIL = OUT_DIR / "table_tnsm_2_2_runtime_candidate_size_detail.csv"
OUT_SUMMARY = OUT_DIR / "table_tnsm_2_2_runtime_candidate_size_summary.csv"
OUT_PAPER = OUT_DIR / "table_tnsm_2_2_runtime_candidate_size_paper.csv"
OUT_REPORT = OUT_DIR / "tnsm_2_2_runtime_candidate_scalability_report.json"

EPS = 1e-12

# Candidate-size settings. The script will automatically clip K to available candidates.
K_LIST = [5, 10, 20, 50, 100]

# Controllers reconstructed from available score columns.
CONTROLLER_SCORE_COLUMNS = {
    "PF": "pf_like_score",
    "Greedy-SLA": "greedy_sla_score",
    "DT-Top1": "pred_mean_management_utility",
}

RANDOM_SEED = 0
MAX_GROUPS_PER_K = None
# If the script is too slow, set MAX_GROUPS_PER_K = 10000 for a quick diagnostic.
# For paper-ready results, keep None.


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
        print("WARNING: test split not found. Using all rows.")

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce")
    df = df.dropna(subset=["candidate_id"]).copy()
    df["candidate_id"] = df["candidate_id"].astype(int)

    for c in ["true_joint_safe", "raw_pred_joint_safe", "cert_joint_safe"]:
        df[c] = df[c].astype(int)

    df["cert_q90_total"] = (
        df["upper_V_embb_q90"].astype(float) + df["upper_V_urllc_q90"].astype(float)
    )

    df["pred_mean_V_total_recomputed"] = (
        df["pred_mean_V_embb"].astype(float) + df["pred_mean_V_urllc"].astype(float)
    )

    return df


def action_distance(row, raw_row):
    return (
        abs(float(row["candidate_embb_slice_prb"]) - float(raw_row["candidate_embb_slice_prb"]))
        + abs(float(row["candidate_urllc_slice_prb"]) - float(raw_row["candidate_urllc_slice_prb"]))
    )


def select_raw_candidate(group, controller):
    score_col = CONTROLLER_SCORE_COLUMNS[controller]
    return group.sort_values([score_col, "candidate_id"], ascending=[False, True]).iloc[0]


def select_subset(group, k, rng):
    """
    Candidate-size sensitivity:
    Keep raw-controller-selected candidate plus top candidates by predicted utility.
    This avoids accidentally removing the raw action from the candidate set.
    """
    group = group.sort_values("candidate_id").copy()
    n = len(group)
    k_eff = min(k, n)

    if k_eff >= n:
        return group.copy(), n

    # Use a stable high-utility subset as candidate candidates.
    # This represents a practical near-RT candidate pre-filtering policy.
    top = group.sort_values(
        ["pred_mean_management_utility", "candidate_id"],
        ascending=[False, True],
    ).head(k_eff).copy()

    return top, k_eff


def choose_certitwin_action(group, raw):
    """
    Full CertiTwin q90 shielding:
    pass-through if raw is certified safe;
    else choose certified-safe candidate by utility-risk-distance score;
    else fallback to minimum certified total risk.
    """
    t0 = time.perf_counter()

    # Stage 1: digital twin inference.
    # In this script predictions are precomputed, so we measure array extraction cost
    # as a proxy for online table/model-output handling.
    pred_utility = group["pred_mean_management_utility"].to_numpy(dtype=float)
    cert_total = group["cert_q90_total"].to_numpy(dtype=float)
    cert_safe = group["cert_joint_safe"].to_numpy(dtype=int)
    true_safe = group["true_joint_safe"].to_numpy(dtype=int)
    _ = pred_utility.mean() + cert_total.mean() + cert_safe.mean() + true_safe.mean()
    t1 = time.perf_counter()

    # Stage 2: certificate construction/filtering.
    # q90 upper bounds are precomputed. Runtime operation is safe-set filtering.
    safe_candidates = group[group["cert_joint_safe"] == 1]
    t2 = time.perf_counter()

    # Stage 3: shielding decision.
    if int(raw["cert_joint_safe"]) == 1:
        exe = raw
        fallback = 0
    else:
        if len(safe_candidates) > 0:
            best_idx = None
            best_score = -1e100
            for idx, row in safe_candidates.iterrows():
                utility = float(row["pred_mean_management_utility"])
                risk = float(row["cert_q90_total"])
                dist = action_distance(row, raw)
                score = utility - 0.05 * risk - 0.01 * dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
            exe = group.loc[best_idx]
            fallback = 0
        else:
            exe = group.sort_values(["cert_q90_total", "candidate_id"], ascending=[True, True]).iloc[0]
            fallback = 1
    t3 = time.perf_counter()

    timing = {
        "twin_output_handling_ms": (t1 - t0) * 1000.0,
        "certificate_filtering_ms": (t2 - t1) * 1000.0,
        "projection_ms": (t3 - t2) * 1000.0,
        "total_shielding_ms": (t3 - t0) * 1000.0,
    }

    return exe, fallback, timing


def evaluate_one(group, controller, k, rng):
    raw_full = select_raw_candidate(group, controller)

    subset, k_eff = select_subset(group, k, rng)

    # Ensure raw action is present in the subset.
    if int(raw_full["candidate_id"]) not in set(subset["candidate_id"].astype(int)):
        subset = pd.concat([subset, raw_full.to_frame().T], ignore_index=True)
        subset = subset.drop_duplicates(subset=["candidate_group_id", "candidate_id"])
        subset = subset.sort_values("candidate_id").copy()
        k_eff = len(subset)

    raw = subset[subset["candidate_id"].astype(int) == int(raw_full["candidate_id"])].iloc[0]

    exe, fallback, timing = choose_certitwin_action(subset, raw)

    raw_true_safe = int(raw["true_joint_safe"])
    exe_true_safe = int(exe["true_joint_safe"])

    raw_utility = float(raw["true_management_utility"])
    exe_utility = float(exe["true_management_utility"])

    dist = action_distance(exe, raw)

    return {
        "controller": controller,
        "target_k": k,
        "effective_k": k_eff,
        "raw_candidate_id": int(raw["candidate_id"]),
        "exec_candidate_id": int(exe["candidate_id"]),
        "raw_unsafe": 1.0 - raw_true_safe,
        "exec_unsafe": 1.0 - exe_true_safe,
        "raw_utility": raw_utility,
        "exec_utility": exe_utility,
        "utility_retention": exe_utility / (raw_utility + EPS),
        "action_changed": float(dist > EPS),
        "action_distance": dist,
        "fallback": float(fallback),
        "safe_candidate_ratio": float(subset["cert_joint_safe"].mean()),
        **timing,
    }


def summarize(detail):
    rows = []

    for (controller, target_k), g in detail.groupby(["controller", "target_k"]):
        rows.append(
            {
                "controller": controller,
                "target_k": int(target_k),
                "mean_effective_k": float(g["effective_k"].mean()),
                "num_groups": int(len(g)),
                "raw_unsafe_rate": float(g["raw_unsafe"].mean()),
                "shielded_unsafe_rate": float(g["exec_unsafe"].mean()),
                "utility_retention_mean": float(g["utility_retention"].mean()),
                "action_changed_rate": float(g["action_changed"].mean()),
                "mean_action_distance": float(g["action_distance"].mean()),
                "fallback_rate": float(g["fallback"].mean()),
                "safe_candidate_ratio": float(g["safe_candidate_ratio"].mean()),
                "twin_output_handling_ms_mean": float(g["twin_output_handling_ms"].mean()),
                "certificate_filtering_ms_mean": float(g["certificate_filtering_ms"].mean()),
                "projection_ms_mean": float(g["projection_ms"].mean()),
                "total_shielding_ms_mean": float(g["total_shielding_ms"].mean()),
                "total_shielding_ms_p95": float(g["total_shielding_ms"].quantile(0.95)),
                "total_shielding_ms_p99": float(g["total_shielding_ms"].quantile(0.99)),
            }
        )

    out = pd.DataFrame(rows)
    controller_order = {"PF": 0, "Greedy-SLA": 1, "DT-Top1": 2}
    out["_c_order"] = out["controller"].map(controller_order)
    out = out.sort_values(["_c_order", "target_k"]).drop(columns=["_c_order"])
    return out


def make_paper_table(summary):
    """
    Compact table aggregated across controllers for main text.
    Controller-level details remain in OUT_SUMMARY.
    """
    rows = []
    for k, g in summary.groupby("target_k"):
        rows.append(
            {
                "Target K": int(k),
                "Mean effective K": float(g["mean_effective_k"].mean()),
                "Mean total latency (ms)": float(g["total_shielding_ms_mean"].mean()),
                "P95 latency (ms)": float(g["total_shielding_ms_p95"].mean()),
                "P99 latency (ms)": float(g["total_shielding_ms_p99"].mean()),
                "Shielded unsafe (%)": 100.0 * float(g["shielded_unsafe_rate"].mean()),
                "Utility retention (%)": 100.0 * float(g["utility_retention_mean"].min()),
                "Fallback (%)": 100.0 * float(g["fallback_rate"].max()),
                "Safe candidate ratio (%)": 100.0 * float(g["safe_candidate_ratio"].mean()),
            }
        )

    paper = pd.DataFrame(rows).sort_values("Target K")

    for c in paper.columns:
        if c != "Target K":
            paper[c] = paper[c].astype(float).round(4)

    return paper


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = load_candidates()
    print(f"Rows used: {len(df)}")
    print(f"Candidate groups: {df['candidate_group_id'].nunique()}")
    print(f"Controllers: {list(CONTROLLER_SCORE_COLUMNS.keys())}")
    print(f"K_LIST: {K_LIST}")

    group_ids = sorted(df["candidate_group_id"].unique())
    if MAX_GROUPS_PER_K is not None:
        group_ids = group_ids[:MAX_GROUPS_PER_K]
        print(f"Using first {len(group_ids)} groups for quick run.")

    df = df[df["candidate_group_id"].isin(group_ids)].copy()

    detail_rows = []
    grouped = list(df.groupby("candidate_group_id"))

    for k in K_LIST:
        print(f"Running K={k} ...")
        t_start = time.perf_counter()

        for gid, group in grouped:
            group = group.sort_values("candidate_id").copy()
            for controller in CONTROLLER_SCORE_COLUMNS.keys():
                result = evaluate_one(group, controller, k, rng)
                result.update(
                    {
                        "candidate_group_id": gid,
                        "num_candidates_original": int(len(group)),
                    }
                )
                detail_rows.append(result)

        t_end = time.perf_counter()
        print(f"K={k} done in {t_end - t_start:.2f}s")

    detail = pd.DataFrame(detail_rows)
    summary = summarize(detail)
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
        "controllers": list(CONTROLLER_SCORE_COLUMNS.keys()),
        "k_list": K_LIST,
        "random_seed": RANDOM_SEED,
        "important_note": (
            "This runtime script measures CertiTwin-RSM shielding overhead using precomputed "
            "digital-twin outputs. Therefore, the reported latency corresponds to runtime "
            "certificate filtering and projection over finite candidate sets, plus lightweight "
            "prediction-output handling. If an online neural-network forward pass is required, "
            "its measured inference latency should be added separately."
        ),
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTNSM-2.2 completed.")
    print("\nPaper table:")
    print(paper.to_string(index=False))
    print(f"\nSaved:\n{OUT_DETAIL}\n{OUT_SUMMARY}\n{OUT_PAPER}\n{OUT_REPORT}")


if __name__ == "__main__":
    main()