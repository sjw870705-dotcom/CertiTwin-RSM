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

OUT_DETAIL = OUT_DIR / "table_tnsm_2_2b_synthetic_runtime_detail.csv"
OUT_SUMMARY = OUT_DIR / "table_tnsm_2_2b_synthetic_runtime_summary.csv"
OUT_PAPER = OUT_DIR / "table_tnsm_2_2b_synthetic_runtime_paper.csv"
OUT_FIT = OUT_DIR / "table_tnsm_2_2b_linear_fit.csv"
OUT_REPORT = OUT_DIR / "tnsm_2_2b_synthetic_runtime_report.json"

RANDOM_SEED = 0

# Synthetic candidate-size settings for runtime scaling only.
K_LIST = [5, 10, 20, 50, 100, 200, 500, 1000]

# Use enough groups for stable latency estimates.
# 10000 is usually enough; if runtime is acceptable, you may increase to 20000 or 60000.
MAX_GROUPS = 10000

# Repeat benchmark to reduce timing noise.
REPETITIONS = 5

# Warm-up rounds are not recorded.
WARMUP_REPETITIONS = 1

CONTROLLERS = {
    "PF": "pf_like_score",
    "Greedy-SLA": "greedy_sla_score",
    "DT-Top1": "pred_mean_management_utility",
}

EPS = 1e-12


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
        "pred_mean_management_utility",
        "upper_V_embb_q90",
        "upper_V_urllc_q90",
        "cert_joint_safe",
    ]

    for score_col in CONTROLLERS.values():
        required.append(score_col)

    require_columns(df, required)

    if "test" in set(df["split"].astype(str).str.lower()):
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        print(f"Using split=test, rows={len(df)}")
    else:
        print("WARNING: split=test not found. Using all rows.")

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce")
    df = df.dropna(subset=["candidate_id"]).copy()
    df["candidate_id"] = df["candidate_id"].astype(int)

    df["cert_joint_safe"] = df["cert_joint_safe"].astype(int)

    df["cert_q90_total"] = (
        df["upper_V_embb_q90"].astype(float) + df["upper_V_urllc_q90"].astype(float)
    )

    return df


def make_group_records(df, max_groups):
    """
    Convert pandas groups into compact numpy records for fast benchmarking.
    """
    group_ids = sorted(df["candidate_group_id"].unique())
    if max_groups is not None:
        group_ids = group_ids[:max_groups]

    df = df[df["candidate_group_id"].isin(group_ids)].copy()

    records = []

    for gid, g in df.groupby("candidate_group_id"):
        g = g.sort_values("candidate_id").copy()

        rec = {
            "candidate_group_id": gid,
            "utility": g["pred_mean_management_utility"].to_numpy(dtype=float),
            "risk": g["cert_q90_total"].to_numpy(dtype=float),
            "safe": g["cert_joint_safe"].to_numpy(dtype=int),
            "embb": g["candidate_embb_slice_prb"].to_numpy(dtype=float),
            "urllc": g["candidate_urllc_slice_prb"].to_numpy(dtype=float),
            "scores": {},
        }

        for controller, score_col in CONTROLLERS.items():
            rec["scores"][controller] = g[score_col].to_numpy(dtype=float)

        records.append(rec)

    return records


def expand_to_k(arr, k, jitter_scale=0.0, rng=None):
    """
    Expand an array to length k by deterministic tiling.
    For runtime benchmark, value distribution is less important than candidate length.
    """
    n = len(arr)
    if n >= k:
        return arr[:k].copy()

    reps = int(np.ceil(k / n))
    out = np.tile(arr, reps)[:k].astype(float)

    if jitter_scale > 0 and rng is not None:
        out = out + rng.normal(0.0, jitter_scale, size=k)

    return out


def synthesize_group(rec, k, rng):
    """
    Build a synthetic K-candidate group from a real 5-candidate group.
    This is only for runtime scaling. It is not used for safety/utility claims.
    """
    utility = expand_to_k(rec["utility"], k, jitter_scale=1e-6, rng=rng)
    risk = expand_to_k(rec["risk"], k, jitter_scale=1e-6, rng=rng)

    safe = expand_to_k(rec["safe"], k, jitter_scale=0.0, rng=None)
    safe = (safe >= 0.5).astype(int)

    embb = expand_to_k(rec["embb"], k, jitter_scale=0.0, rng=None)
    urllc = expand_to_k(rec["urllc"], k, jitter_scale=0.0, rng=None)

    scores = {}
    for controller in CONTROLLERS.keys():
        scores[controller] = expand_to_k(rec["scores"][controller], k, jitter_scale=1e-6, rng=rng)

    # Candidate ids are synthetic and only used for tie-breaking.
    candidate_id = np.arange(k, dtype=int)

    return {
        "utility": utility,
        "risk": risk,
        "safe": safe,
        "embb": embb,
        "urllc": urllc,
        "scores": scores,
        "candidate_id": candidate_id,
    }


def certitwin_runtime_decision(group, controller):
    """
    Vectorized CertiTwin-RSM decision over synthetic candidate arrays.

    Timing stages:
    1. output handling: array access and basic construction;
    2. certificate filtering: safe-set index construction;
    3. projection: pass-through / utility-risk-distance scoring / fallback.
    """
    t0 = time.perf_counter()

    utility = group["utility"]
    risk = group["risk"]
    safe = group["safe"]
    embb = group["embb"]
    urllc = group["urllc"]
    scores = group["scores"][controller]

    # Lightweight output handling.
    # This mimics using precomputed or just-produced digital-twin outputs.
    _ = utility[0] + risk[0] + safe[0]
    t1 = time.perf_counter()

    # Raw action from reconstructed controller score.
    # Tie-break by smallest candidate id through np.argmax on stable order.
    raw_idx = int(np.argmax(scores))

    # Certificate filtering.
    safe_idx = np.flatnonzero(safe == 1)
    t2 = time.perf_counter()

    # Projection or fallback.
    if safe[raw_idx] == 1:
        exec_idx = raw_idx
        fallback = 0
    else:
        if len(safe_idx) > 0:
            dist = np.abs(embb[safe_idx] - embb[raw_idx]) + np.abs(urllc[safe_idx] - urllc[raw_idx])
            projection_score = utility[safe_idx] - 0.05 * risk[safe_idx] - 0.01 * dist
            best_local = int(np.argmax(projection_score))
            exec_idx = int(safe_idx[best_local])
            fallback = 0
        else:
            exec_idx = int(np.argmin(risk))
            fallback = 1

    t3 = time.perf_counter()

    action_distance = abs(embb[exec_idx] - embb[raw_idx]) + abs(urllc[exec_idx] - urllc[raw_idx])

    return {
        "output_handling_ms": (t1 - t0) * 1000.0,
        "certificate_filtering_ms": (t2 - t1) * 1000.0,
        "projection_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
        "fallback": float(fallback),
        "safe_candidate_ratio": float(np.mean(safe)),
        "action_distance": float(action_distance),
        "raw_idx": raw_idx,
        "exec_idx": exec_idx,
    }


def run_benchmark(records):
    rng = np.random.default_rng(RANDOM_SEED)

    detail_rows = []

    # Warm-up to reduce first-run overhead.
    print("Warm-up...")
    for k in K_LIST[:2]:
        for rec in records[:100]:
            group = synthesize_group(rec, k, rng)
            for controller in CONTROLLERS.keys():
                _ = certitwin_runtime_decision(group, controller)

    print("Benchmarking...")
    for k in K_LIST:
        print(f"Running synthetic K={k} ...")
        k_start = time.perf_counter()

        for rep in range(REPETITIONS + WARMUP_REPETITIONS):
            record_this_rep = rep >= WARMUP_REPETITIONS

            for rec_i, rec in enumerate(records):
                group = synthesize_group(rec, k, rng)

                for controller in CONTROLLERS.keys():
                    result = certitwin_runtime_decision(group, controller)

                    if record_this_rep:
                        detail_rows.append(
                            {
                                "synthetic_k": k,
                                "repetition": rep - WARMUP_REPETITIONS,
                                "candidate_group_id": rec["candidate_group_id"],
                                "controller": controller,
                                **result,
                            }
                        )

        k_end = time.perf_counter()
        print(f"Synthetic K={k} done in {k_end - k_start:.2f}s")

    return pd.DataFrame(detail_rows)


def summarize(detail):
    rows = []

    for k, g in detail.groupby("synthetic_k"):
        rows.append(
            {
                "synthetic_k": int(k),
                "num_decisions": int(len(g)),
                "controllers": int(g["controller"].nunique()),
                "repetitions": int(g["repetition"].nunique()),
                "output_handling_ms_mean": float(g["output_handling_ms"].mean()),
                "certificate_filtering_ms_mean": float(g["certificate_filtering_ms"].mean()),
                "projection_ms_mean": float(g["projection_ms"].mean()),
                "total_ms_mean": float(g["total_ms"].mean()),
                "total_ms_median": float(g["total_ms"].median()),
                "total_ms_p95": float(g["total_ms"].quantile(0.95)),
                "total_ms_p99": float(g["total_ms"].quantile(0.99)),
                "latency_per_candidate_us": float(g["total_ms"].mean() * 1000.0 / k),
                "fallback_rate": float(g["fallback"].mean()),
                "safe_candidate_ratio": float(g["safe_candidate_ratio"].mean()),
                "mean_action_distance": float(g["action_distance"].mean()),
                "throughput_decisions_per_second": float(1000.0 / (g["total_ms"].mean() + EPS)),
            }
        )

    out = pd.DataFrame(rows).sort_values("synthetic_k")
    return out


def linear_fit(summary):
    x = summary["synthetic_k"].to_numpy(dtype=float)
    y = summary["total_ms_mean"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + EPS)

    fit = pd.DataFrame(
        [
            {
                "fit_target": "mean_total_latency_ms_vs_K",
                "slope_ms_per_candidate": slope,
                "intercept_ms": intercept,
                "r2": r2,
                "k_min": int(np.min(x)),
                "k_max": int(np.max(x)),
                "num_points": int(len(x)),
            }
        ]
    )

    return fit


def make_paper_table(summary):
    paper = summary[
        [
            "synthetic_k",
            "total_ms_mean",
            "total_ms_p95",
            "total_ms_p99",
            "latency_per_candidate_us",
            "throughput_decisions_per_second",
        ]
    ].copy()

    paper = paper.rename(
        columns={
            "synthetic_k": "Synthetic K",
            "total_ms_mean": "Mean latency (ms)",
            "total_ms_p95": "P95 latency (ms)",
            "total_ms_p99": "P99 latency (ms)",
            "latency_per_candidate_us": "Latency / candidate (us)",
            "throughput_decisions_per_second": "Throughput (decisions/s)",
        }
    )

    for c in paper.columns:
        if c != "Synthetic K":
            paper[c] = paper[c].astype(float).round(6)

    return paper


def main():
    df = load_candidates()
    records = make_group_records(df, MAX_GROUPS)

    print(f"Candidate groups used for benchmark: {len(records)}")
    print(f"Controllers: {list(CONTROLLERS.keys())}")
    print(f"K_LIST: {K_LIST}")
    print(f"Repetitions: {REPETITIONS}")

    detail = run_benchmark(records)
    summary = summarize(detail)
    fit = linear_fit(summary)
    paper = make_paper_table(summary)

    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")
    fit.to_csv(OUT_FIT, index=False, encoding="utf-8-sig")

    report = {
        "status": "completed",
        "input": str(INPUT_CANDIDATES),
        "outputs": {
            "detail": str(OUT_DETAIL),
            "summary": str(OUT_SUMMARY),
            "paper": str(OUT_PAPER),
            "linear_fit": str(OUT_FIT),
        },
        "candidate_groups_used": int(len(records)),
        "controllers": list(CONTROLLERS.keys()),
        "k_list": K_LIST,
        "repetitions": REPETITIONS,
        "random_seed": RANDOM_SEED,
        "linear_fit": fit.to_dict(orient="records")[0],
        "important_note": (
            "This is a synthetic candidate-size runtime benchmark. It expands each real "
            "candidate group to K candidates by deterministic tiling with tiny numerical jitter. "
            "It is used only to test certificate-filtering and projection scalability with respect "
            "to candidate-set size. It must not be interpreted as replayed-candidate performance "
            "for safety or utility. The main replayed-candidate performance remains based on the "
            "real K=5 candidate groups."
        ),
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTNSM-2.2B completed.")

    print("\nPaper table:")
    print(paper.to_string(index=False))

    print("\nLinear fit:")
    print(fit.to_string(index=False))

    print(f"\nSaved:\n{OUT_DETAIL}\n{OUT_SUMMARY}\n{OUT_PAPER}\n{OUT_FIT}\n{OUT_REPORT}")


if __name__ == "__main__":
    main()