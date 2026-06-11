#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_coloran_logged_sanity_validation_v4.py

ColO-RAN logged sanity validation (V4) for CertiTwin-RSM.

Purpose
-------
This script builds a small SECONDARY public logged O-RAN sanity validation
from the ColO-RAN rome_static_medium dataset.

The validation is intentionally modest: it is not a live near-RT RIC
deployment and not a replacement for the main commercial-twinning replay.
It checks whether the same qualitative runtime-governance behavior appears
on public logged O-RAN data when candidate actions are scheduling-policy
candidates (sched0/sched1/sched2), rather than PRB-allocation vectors.

Key fixes over V2
-----------------
1. Group-level random split instead of chronological split:
   This sanity validation is not meant to test long-horizon forecasting
   across tr IDs. The goal is to evaluate runtime governance under public
   logged candidate groups while avoiding train/calib/test row leakage.

2. Cleaner candidate-aware twin features:
   The digital-twin model uses context/action features such as sched_id,
   BS ID, tr/exp/window IDs, slice PRB, number of UEs, and requested PRBs.
   Logged outcomes such as throughput, buffer, error, and starvation are
   used only to construct outcome/risk labels, not as twin input features.

3. Rank-normalized finite-candidate logged risk proxies:
   eMBB risk is the relative deficit from the best eMBB throughput candidate
   within the same finite group. URLLC-like risk is a group-relative
   normalized burden from starvation, buffer pressure, and error metrics.

Outputs
-------
- coloran_candidate_group_dataset_v4.csv
- table_coloran_logged_sanity_validation_v4.csv
- coloran_logged_sanity_metadata_v4.txt
- coloran_logged_sanity_diagnostics_v4.csv

Dependencies
------------
pandas, numpy, scikit-learn
Install if needed:
    pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

SCHED_LABELS = {
    0: "RR-like",
    1: "WF-like",
    2: "PF-like",
}
EPS = 1e-9


def parse_ids(path: Path) -> Tuple[int, int, int, int]:
    parts = path.as_posix().split("/")
    sched_id = tr_id = exp_id = bs_id = None
    for p in parts:
        if re.fullmatch(r"sched\d+", p):
            sched_id = int(p.replace("sched", ""))
        elif re.fullmatch(r"tr\d+", p):
            tr_id = int(p.replace("tr", ""))
        elif re.fullmatch(r"exp\d+", p):
            exp_id = int(p.replace("exp", ""))
        elif re.fullmatch(r"bs\d+", p):
            bs_id = int(p.replace("bs", ""))
    if sched_id is None or tr_id is None or exp_id is None or bs_id is None:
        raise ValueError(f"Cannot parse sched/tr/exp/bs ids from {path}")
    return sched_id, tr_id, exp_id, bs_id


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep, cols = [], []
    for c in df.columns:
        cc = str(c).strip()
        if cc and not cc.lower().startswith("unnamed"):
            keep.append(c)
            cols.append(cc)
    out = df[keep].copy()
    out.columns = cols
    return out


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def mode_or_first(s: pd.Series, default=0):
    if len(s) == 0:
        return default
    try:
        m = s.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
    except Exception:
        pass
    return s.iloc[0]


def aggregate_one_bs_policy(bs_dir: Path, window_ms: int) -> pd.DataFrame:
    """Aggregate slice-level logged metrics for one sched/tr/exp/bs directory."""
    sched_id, tr_id, exp_id, bs_id = parse_ids(bs_dir)
    slices_dir = bs_dir / f"slices_bs{bs_id}"
    if not slices_dir.exists():
        return pd.DataFrame()

    frames = []
    for fp in sorted(slices_dir.glob("*_metrics.csv")):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df.empty:
            continue
        df = clean_columns(df)
        if "Timestamp" not in df.columns or "slice_id" not in df.columns:
            continue

        df["Timestamp"] = safe_numeric(df, "Timestamp")
        df["slice_id"] = safe_numeric(df, "slice_id").astype(int)
        df["sched_id"] = sched_id
        df["tr_id"] = tr_id
        df["exp_id"] = exp_id
        df["bs_id"] = bs_id

        t0 = df["Timestamp"].min()
        df["window_idx"] = np.floor((df["Timestamp"] - t0) / float(window_ms)).astype(int)

        keep = [
            "sched_id", "tr_id", "exp_id", "bs_id", "window_idx", "Timestamp",
            "slice_id", "num_ues", "slice_prb", "scheduling_policy",
            "dl_mcs", "dl_buffer [bytes]", "tx_brate downlink [Mbps]",
            "tx_errors downlink (%)", "dl_cqi",
            "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
            "rx_errors uplink (%)", "ul_sinr",
            "sum_requested_prbs", "sum_granted_prbs",
        ]
        for col in keep:
            if col not in df.columns:
                df[col] = 0.0
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    group_cols = ["sched_id", "tr_id", "exp_id", "bs_id", "window_idx", "slice_id"]
    agg = data.groupby(group_cols, as_index=False).agg({
        "Timestamp": "min",
        "num_ues": "max",
        "slice_prb": "mean",
        "scheduling_policy": mode_or_first,
        "dl_mcs": "mean",
        "dl_buffer [bytes]": "sum",
        "tx_brate downlink [Mbps]": "sum",
        "tx_errors downlink (%)": "mean",
        "dl_cqi": "mean",
        "ul_buffer [bytes]": "sum",
        "rx_brate uplink [Mbps]": "sum",
        "rx_errors uplink (%)": "mean",
        "ul_sinr": "mean",
        "sum_requested_prbs": "sum",
        "sum_granted_prbs": "sum",
    })

    rows = []
    for (sched, tr, exp, bs, win), g in agg.groupby(["sched_id", "tr_id", "exp_id", "bs_id", "window_idx"]):
        row = {
            "sched_id": int(sched),
            "sched_label": SCHED_LABELS.get(int(sched), f"sched{sched}"),
            "tr_id": int(tr),
            "exp_id": int(exp),
            "bs_id": int(bs),
            "window_idx": int(win),
        }
        total_dl = 0.0
        total_ul = 0.0
        total_ues = 0.0
        for sid in [0, 1, 2]:
            sg = g[g["slice_id"] == sid]
            prefix = f"s{sid}"
            if len(sg) == 0:
                vals = {
                    f"rate_dl_{prefix}": 0.0,
                    f"rate_ul_{prefix}": 0.0,
                    f"num_ues_{prefix}": 0.0,
                    f"slice_prb_{prefix}": 0.0,
                    f"dl_buffer_{prefix}": 0.0,
                    f"ul_buffer_{prefix}": 0.0,
                    f"tx_err_{prefix}": 0.0,
                    f"rx_err_{prefix}": 0.0,
                    f"dl_cqi_{prefix}": 0.0,
                    f"ul_sinr_{prefix}": 0.0,
                    f"requested_prb_{prefix}": 0.0,
                    f"granted_prb_{prefix}": 0.0,
                    f"starvation_{prefix}": 0.0,
                }
            else:
                rate_dl = float(sg["tx_brate downlink [Mbps]"].sum())
                rate_ul = float(sg["rx_brate uplink [Mbps]"].sum())
                req = float(sg["sum_requested_prbs"].sum())
                grt = float(sg["sum_granted_prbs"].sum())
                vals = {
                    f"rate_dl_{prefix}": rate_dl,
                    f"rate_ul_{prefix}": rate_ul,
                    f"num_ues_{prefix}": float(sg["num_ues"].max()),
                    f"slice_prb_{prefix}": float(sg["slice_prb"].mean()),
                    f"dl_buffer_{prefix}": float(sg["dl_buffer [bytes]"].sum()),
                    f"ul_buffer_{prefix}": float(sg["ul_buffer [bytes]"].sum()),
                    f"tx_err_{prefix}": float(sg["tx_errors downlink (%)"].mean()),
                    f"rx_err_{prefix}": float(sg["rx_errors uplink (%)"].mean()),
                    f"dl_cqi_{prefix}": float(sg["dl_cqi"].mean()),
                    f"ul_sinr_{prefix}": float(sg["ul_sinr"].mean()),
                    f"requested_prb_{prefix}": req,
                    f"granted_prb_{prefix}": grt,
                    f"starvation_{prefix}": max(req - grt, 0.0) / max(req, 1.0),
                }
                total_dl += rate_dl
                total_ul += rate_ul
                total_ues += vals[f"num_ues_{prefix}"]
            row.update(vals)
        row["total_dl_rate"] = total_dl
        row["total_ul_rate"] = total_ul
        row["total_ues"] = total_ues
        rows.append(row)

    return pd.DataFrame(rows)


def collect_candidate_rows(root: Path, max_tr: int, max_exp: int, max_windows: int, window_ms: int) -> pd.DataFrame:
    base = root / "rome_static_medium"
    rows = []
    for sched_dir in sorted(base.glob("sched*")):
        if not sched_dir.is_dir():
            continue
        for tr_dir in sorted(sched_dir.glob("tr*"), key=lambda p: int(p.name.replace("tr", ""))):
            tr_id = int(tr_dir.name.replace("tr", ""))
            if max_tr > 0 and tr_id >= max_tr:
                continue
            for exp_dir in sorted(tr_dir.glob("exp*"), key=lambda p: int(p.name.replace("exp", ""))):
                exp_id = int(exp_dir.name.replace("exp", ""))
                if max_exp > 0 and exp_id > max_exp:
                    continue
                for bs_dir in sorted(exp_dir.glob("bs*"), key=lambda p: int(p.name.replace("bs", ""))):
                    if not bs_dir.is_dir():
                        continue
                    df = aggregate_one_bs_policy(bs_dir, window_ms=window_ms)
                    if df.empty:
                        continue
                    if max_windows > 0:
                        df = df[df["window_idx"] < max_windows]
                    if not df.empty:
                        rows.append(df)
    if not rows:
        raise RuntimeError("No candidate rows collected. Check dataset path and parameters.")

    df = pd.concat(rows, ignore_index=True)
    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]

    # Keep only finite candidate groups that contain all three scheduling policies.
    counts = df.groupby(key_cols)["sched_id"].nunique().reset_index(name="n_sched")
    valid = counts[counts["n_sched"] >= 3][key_cols]
    df = df.merge(valid, on=key_cols, how="inner")
    return df


def add_group_relative_outcomes(df: pd.DataFrame, eps_embb: float, eps_urllc: float) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Construct rank-normalized finite-candidate logged risk proxies and utility.

    The proxy is intentionally defined at the candidate-group level. For each
    (tr, exp, bs, time-window) group, sched0/sched1/sched2 are the finite
    candidate actions. The best candidate for an objective gets risk 0, the
    middle candidate gets risk 0.5, and the worst candidate gets risk 1.
    This avoids treating an absolute throughput level as an operator SLA in
    this secondary public logged sanity validation.
    """
    df = df.copy()
    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]

    # eMBB risk: rank-normalized throughput deficit.
    # Best eMBB throughput in the group -> 0; middle -> 0.5; worst -> 1.
    n = df.groupby(key_cols)["sched_id"].transform("count").astype(float)
    rank_embb = df.groupby(key_cols)["rate_dl_s0"].rank(method="average", ascending=False)
    df["risk_embb"] = ((rank_embb - 1.0) / (n - 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["risk_embb"] = df["risk_embb"].clip(0.0, 1.0)

    # URLLC-like burden from slice-2 starvation, buffer, and error.
    # Lower burden is better.
    def robust_norm(s: pd.Series) -> pd.Series:
        q95 = float(s.quantile(0.95))
        if q95 <= EPS:
            q95 = max(float(s.max()), 1.0)
        return (s / (q95 + EPS)).clip(0.0, 2.0)

    url_buf = df["dl_buffer_s2"] + df["ul_buffer_s2"]
    url_err = df["tx_err_s2"] + df["rx_err_s2"]
    df["urllc_burden_raw"] = (
        0.50 * robust_norm(df["starvation_s2"])
        + 0.25 * robust_norm(url_buf)
        + 0.25 * robust_norm(url_err)
    )

    rank_urllc = df.groupby(key_cols)["urllc_burden_raw"].rank(method="average", ascending=True)
    df["risk_urllc_like"] = ((rank_urllc - 1.0) / (n - 1.0)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["risk_urllc_like"] = df["risk_urllc_like"].clip(0.0, 1.0)

    # Utility keeps magnitude information so retention remains meaningful.
    def group_norm(col: str) -> pd.Series:
        mx = df.groupby(key_cols)[col].transform("max")
        return np.where(mx > EPS, df[col] / (mx + EPS), 0.0)

    df["embb_score"] = group_norm("rate_dl_s0")
    mtc_score = group_norm("rate_dl_s1")
    url_score = group_norm("rate_dl_s2")

    df["utility_true"] = (
        0.55 * df["embb_score"]
        + 0.15 * mtc_score
        + 0.30 * url_score
        - 0.20 * df["risk_urllc_like"]
    )
    min_u = float(df["utility_true"].min())
    if min_u <= 0:
        df["utility_true"] = df["utility_true"] - min_u + 0.05

    df["true_safe"] = ((df["risk_embb"] <= eps_embb) & (df["risk_urllc_like"] <= eps_urllc)).astype(int)
    df["true_unsafe"] = 1 - df["true_safe"]

    meta = {
        "eps_embb": eps_embb,
        "eps_urllc_like": eps_urllc,
        "candidate_rows": int(len(df)),
        "groups": int(df[key_cols].drop_duplicates().shape[0]),
        "true_safe_candidate_ratio": float(df["true_safe"].mean()),
        "groups_with_true_safe_ratio": float(df.groupby(key_cols)["true_safe"].max().mean()),
        "proxy_note": "Rank-normalized within each finite candidate group: best=0, middle=0.5, worst=1.",
    }
    return df, meta


def random_group_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split by candidate group to prevent row-level leakage."""
    df = df.copy()
    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]
    groups = df[key_cols].drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    order = np.arange(len(groups))
    rng.shuffle(order)

    n = len(groups)
    n_train = int(0.60 * n)
    n_calib = int(0.20 * n)

    groups["split"] = "test"
    groups.loc[order[:n_train], "split"] = "train"
    groups.loc[order[n_train:n_train + n_calib], "split"] = "calib"

    return df.merge(groups, on=key_cols, how="left")


def build_twin_features(df: pd.DataFrame) -> list[str]:
    """Use context/action features only; avoid outcome leakage."""
    features = [
        "sched_id", "tr_id", "exp_id", "bs_id", "window_idx",
        "slice_prb_s0", "slice_prb_s1", "slice_prb_s2",
        "num_ues_s0", "num_ues_s1", "num_ues_s2",
        "total_ues",
        "requested_prb_s0", "requested_prb_s1", "requested_prb_s2",
    ]
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    return features


def fit_twin_and_certify(
    df: pd.DataFrame,
    seed: int,
    eps_embb: float,
    eps_urllc: float,
    q_quantile: float,
    q_cap: float,
    cert_relax: float,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    features = build_twin_features(df)

    # Additional prediction target embb_score helps define a throughput-greedy raw controller
    # without using true outcome at evaluation time.
    targets = ["risk_embb", "risk_urllc_like", "utility_true", "embb_score"]

    train = df[df["split"] == "train"].copy()
    calib = df[df["split"] == "calib"].copy()
    if len(train) < 50 or len(calib) < 20:
        raise RuntimeError("Too few train/calib rows. Increase max-tr/max-exp/max-windows.")

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=180,
            max_depth=11,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    )
    model.fit(train[features], train[targets])

    pred = model.predict(df[features])
    df["pred_risk_embb"] = np.clip(pred[:, 0], 0.0, 1.0)
    df["pred_risk_urllc_like"] = np.clip(pred[:, 1], 0.0, 1.0)
    df["pred_utility"] = np.maximum(pred[:, 2], 0.0)
    df["pred_embb_score"] = np.clip(pred[:, 3], 0.0, 1.0)

    calib_pred = model.predict(calib[features])
    pos_embb = np.maximum(calib["risk_embb"].values - calib_pred[:, 0], 0.0)
    pos_urllc = np.maximum(calib["risk_urllc_like"].values - calib_pred[:, 1], 0.0)

    q_embb_raw = float(np.quantile(pos_embb, q_quantile))
    q_urllc_raw = float(np.quantile(pos_urllc, q_quantile))
    q_embb = min(q_embb_raw, q_cap)
    q_urllc = min(q_urllc_raw, q_cap)

    df["cert_risk_embb"] = df["pred_risk_embb"] + q_embb
    df["cert_risk_urllc_like"] = df["pred_risk_urllc_like"] + q_urllc
    df["cert_safe"] = (
        (df["cert_risk_embb"] <= eps_embb + cert_relax)
        & (df["cert_risk_urllc_like"] <= eps_urllc + cert_relax)
    ).astype(int)
    df["cert_score"] = 0.5 * df["cert_risk_embb"] + 0.5 * df["cert_risk_urllc_like"]

    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]

    def mae(split: str, target: str, pred_col: str) -> float:
        d = df[df["split"] == split]
        if len(d) == 0:
            return float("nan")
        return float(mean_absolute_error(d[target], d[pred_col]))

    meta = {
        "feature_set": ";".join(features),
        "q_quantile": q_quantile,
        "q90_embb_residual_raw": q_embb_raw,
        "q90_urllc_like_residual_raw": q_urllc_raw,
        "q_embb_residual_used": q_embb,
        "q_urllc_like_residual_used": q_urllc,
        "q_cap": q_cap,
        "cert_relax": cert_relax,
        "train_mae_risk_embb": mae("train", "risk_embb", "pred_risk_embb"),
        "train_mae_risk_urllc_like": mae("train", "risk_urllc_like", "pred_risk_urllc_like"),
        "calib_mae_risk_embb": mae("calib", "risk_embb", "pred_risk_embb"),
        "calib_mae_risk_urllc_like": mae("calib", "risk_urllc_like", "pred_risk_urllc_like"),
        "test_mae_risk_embb": mae("test", "risk_embb", "pred_risk_embb"),
        "test_mae_risk_urllc_like": mae("test", "risk_urllc_like", "pred_risk_urllc_like"),
        "cert_safe_candidate_ratio": float(df["cert_safe"].mean()),
        "groups_with_cert_safe_ratio": float(df.groupby(key_cols)["cert_safe"].max().mean()),
    }
    return df, meta


def choose_raw(group: pd.DataFrame, controller: str) -> pd.Series:
    g = group.copy()

    if controller == "PF-like":
        cand = g[g["sched_id"] == 2]
        if len(cand) > 0:
            return cand.iloc[0]
        return g.sort_values("pred_utility", ascending=False).iloc[0]

    if controller == "Throughput-greedy":
        # Aggressive raw controller: maximize predicted eMBB throughput score.
        return g.sort_values(["pred_embb_score", "pred_utility"], ascending=False).iloc[0]

    if controller == "SLA-aware":
        # Conservative raw controller: prefer low predicted risk, then utility.
        return g.sort_values(
            ["pred_risk_urllc_like", "pred_risk_embb", "pred_utility"],
            ascending=[True, True, False],
        ).iloc[0]

    raise ValueError(f"Unknown controller: {controller}")


def shield_action(group: pd.DataFrame, raw: pd.Series, lambda_r: float, lambda_d: float) -> tuple[pd.Series, int]:
    g = group.copy()

    if int(raw["cert_safe"]) == 1:
        return raw, 0

    safe = g[g["cert_safe"] == 1].copy()
    if len(safe) == 0:
        return g.sort_values("cert_score").iloc[0], 1

    safe["policy_distance"] = (safe["sched_id"].astype(float) - float(raw["sched_id"])).abs()
    safe["shield_score"] = (
        safe["pred_utility"]
        - lambda_r * safe["cert_score"]
        - lambda_d * safe["policy_distance"]
    )
    return safe.sort_values("shield_score", ascending=False).iloc[0], 0


def evaluate(df: pd.DataFrame, lambda_r: float, lambda_d: float) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]
    controllers = ["PF-like", "Throughput-greedy", "SLA-aware"]
    rows = []

    for ctrl in controllers:
        raw_unsafe, shield_unsafe = [], []
        utility_ret, changed, pass_through, fallback, dist = [], [], [], [], []

        for _, g in test.groupby(key_cols, sort=False):
            if g["sched_id"].nunique() < 3:
                continue
            raw = choose_raw(g, ctrl)
            exe, fb = shield_action(g, raw, lambda_r=lambda_r, lambda_d=lambda_d)

            raw_unsafe.append(float(raw["true_unsafe"]))
            shield_unsafe.append(float(exe["true_unsafe"]))
            utility_ret.append(float(exe["utility_true"]) / max(float(raw["utility_true"]), EPS))

            ch = int(raw["sched_id"]) != int(exe["sched_id"])
            changed.append(float(ch))
            pass_through.append(1.0 - float(ch))
            fallback.append(float(fb))
            dist.append(abs(float(raw["sched_id"]) - float(exe["sched_id"])))

        if not raw_unsafe:
            continue

        raw_rate = 100.0 * float(np.mean(raw_unsafe))
        shield_rate = 100.0 * float(np.mean(shield_unsafe))
        reduction = np.nan if raw_rate <= 1e-12 else 100.0 * (raw_rate - shield_rate) / raw_rate

        rows.append({
            "Scenario": "ColO-RAN logged",
            "Raw controller": ctrl,
            "Raw unsafe (%)": raw_rate,
            "Shielded unsafe (%)": shield_rate,
            "Unsafe reduction (%)": reduction,
            "Utility retention (%)": 100.0 * float(np.mean(utility_ret)),
            "Action changed (%)": 100.0 * float(np.mean(changed)),
            "Pass-through (%)": 100.0 * float(np.mean(pass_through)),
            "Mean policy distance": float(np.mean(dist)),
            "Fallback (%)": 100.0 * float(np.mean(fallback)),
            "Evaluated groups": int(len(raw_unsafe)),
        })

    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["tr_id", "exp_id", "bs_id", "window_idx"]
    rows = []
    for split, d in df.groupby("split"):
        rows.append({
            "split": split,
            "candidate_rows": int(len(d)),
            "groups": int(d[key_cols].drop_duplicates().shape[0]),
            "true_safe_candidate_ratio": float(d["true_safe"].mean()),
            "cert_safe_candidate_ratio": float(d["cert_safe"].mean()),
            "groups_with_true_safe_ratio": float(d.groupby(key_cols)["true_safe"].max().mean()),
            "groups_with_cert_safe_ratio": float(d.groupby(key_cols)["cert_safe"].max().mean()),
            "mean_risk_embb": float(d["risk_embb"].mean()),
            "mean_risk_urllc_like": float(d["risk_urllc_like"].mean()),
            "mean_cert_risk_embb": float(d["cert_risk_embb"].mean()),
            "mean_cert_risk_urllc_like": float(d["cert_risk_urllc_like"].mean()),
        })
    all_d = df
    rows.append({
        "split": "all",
        "candidate_rows": int(len(all_d)),
        "groups": int(all_d[key_cols].drop_duplicates().shape[0]),
        "true_safe_candidate_ratio": float(all_d["true_safe"].mean()),
        "cert_safe_candidate_ratio": float(all_d["cert_safe"].mean()),
        "groups_with_true_safe_ratio": float(all_d.groupby(key_cols)["true_safe"].max().mean()),
        "groups_with_cert_safe_ratio": float(all_d.groupby(key_cols)["cert_safe"].max().mean()),
        "mean_risk_embb": float(all_d["risk_embb"].mean()),
        "mean_risk_urllc_like": float(all_d["risk_urllc_like"].mean()),
        "mean_cert_risk_embb": float(all_d["cert_risk_embb"].mean()),
        "mean_cert_risk_urllc_like": float(all_d["cert_risk_urllc_like"].mean()),
    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to colosseum-oran-coloran-dataset root.")
    parser.add_argument("--out-dir", type=str, default="coloran_logged_sanity_v4_outputs")
    parser.add_argument("--max-tr", type=int, default=28,
                        help="Use tr0...tr(max-tr-1). For debug, set 3.")
    parser.add_argument("--max-exp", type=int, default=5,
                        help="Use exp1...exp(max-exp). For debug, set 1.")
    parser.add_argument("--max-windows", type=int, default=60,
                        help="Max windows per sched/tr/exp/bs. Use 0 for all.")
    parser.add_argument("--window-ms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--eps-embb", type=float, default=0.75,
                        help="Rank-normalized eMBB risk threshold. Default 0.75 means the worst policy candidate is unsafe.")
    parser.add_argument("--eps-urllc", type=float, default=0.75,
                        help="Rank-normalized URLLC-like risk threshold. Default 0.75 means the worst policy candidate is unsafe.")
    parser.add_argument("--q-quantile", type=float, default=0.80,
                        help="Calibration quantile for logged sanity certificate.")
    parser.add_argument("--q-cap", type=float, default=0.15,
                        help="Residual cap for logged proxy certificate.")
    parser.add_argument("--cert-relax", type=float, default=0.05,
                        help="Small relaxation for logged proxy sanity validation.")
    parser.add_argument("--lambda-r", type=float, default=0.35)
    parser.add_argument("--lambda-d", type=float, default=0.08)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not (root / "rome_static_medium").exists():
        raise FileNotFoundError(f"Cannot find rome_static_medium under dataset root: {root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Collecting ColO-RAN logged candidate rows...")
    df = collect_candidate_rows(
        root=root,
        max_tr=args.max_tr,
        max_exp=args.max_exp,
        max_windows=args.max_windows,
        window_ms=args.window_ms,
    )
    print(f"  collected candidate rows: {len(df)}")

    print("[2/6] Building group-relative logged risk proxies...")
    df, proxy_meta = add_group_relative_outcomes(df, eps_embb=args.eps_embb, eps_urllc=args.eps_urllc)

    print("[3/6] Random group split...")
    df = random_group_split(df, seed=args.seed)

    print("[4/6] Fitting candidate-aware twin and calibrated certificate...")
    df, twin_meta = fit_twin_and_certify(
        df,
        seed=args.seed,
        eps_embb=args.eps_embb,
        eps_urllc=args.eps_urllc,
        q_quantile=args.q_quantile,
        q_cap=args.q_cap,
        cert_relax=args.cert_relax,
    )

    print("[5/6] Evaluating runtime governance...")
    table = evaluate(df, lambda_r=args.lambda_r, lambda_d=args.lambda_d)
    diag = diagnostics(df)

    print("[6/6] Saving outputs...")
    dataset_path = out_dir / "coloran_candidate_group_dataset_v4.csv"
    table_path = out_dir / "table_coloran_logged_sanity_validation_v4.csv"
    diag_path = out_dir / "coloran_logged_sanity_diagnostics_v4.csv"
    meta_path = out_dir / "coloran_logged_sanity_metadata_v4.txt"

    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    diag.to_csv(diag_path, index=False, encoding="utf-8-sig")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("ColO-RAN logged sanity validation V4 metadata\n")
        f.write("============================================\n\n")
        f.write("This is a public logged O-RAN sanity validation, not a live near-RT RIC deployment.\n")
        f.write("Candidate action semantics: scheduling-policy candidates (sched0/sched1/sched2).\n")
        f.write("Split protocol: group-level random train/calib/test split; no row-level leakage across the same candidate group.\n")
        f.write("Twin features: context/action features only; logged throughput/buffer/error/starvation fields are used as outcomes, not as input features.\n")
        f.write("Risk proxies: rank-normalized finite-candidate eMBB throughput-deficit risk and URLLC-like burden risk.\n")
        f.write("Slice mapping: slice 0=eMBB, slice 1=MTC, slice 2=URLLC, following the ColO-RAN README.\n\n")

        f.write("Run parameters:\n")
        for k, v in vars(args).items():
            f.write(f"- {k}: {v}\n")

        f.write("\nProxy metadata:\n")
        for k, v in proxy_meta.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nTwin/certificate metadata:\n")
        for k, v in twin_meta.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nOutputs:\n")
        f.write(f"- {dataset_path}\n")
        f.write(f"- {table_path}\n")
        f.write(f"- {diag_path}\n")

    print("\n=== Paper-style ColO-RAN logged sanity table ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(table.round(4).to_string(index=False))

    print("\n=== Diagnostics ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(diag.round(4).to_string(index=False))

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print(f"- {dataset_path.name}")
    print(f"- {table_path.name}")
    print(f"- {diag_path.name}")
    print(f"- {meta_path.name}")


if __name__ == "__main__":
    main()
