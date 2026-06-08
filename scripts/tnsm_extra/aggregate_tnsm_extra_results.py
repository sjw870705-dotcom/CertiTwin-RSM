from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "tnsm_extra"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs from previous steps
IN_RULE = RESULT_DIR / "table_tnsm_2_1b_controller_rule_shield_paper.csv"
IN_RUNTIME_REAL = RESULT_DIR / "table_tnsm_2_2_runtime_candidate_size_paper.csv"
IN_RUNTIME_SYN = RESULT_DIR / "table_tnsm_2_2b_synthetic_runtime_paper.csv"
IN_RUNTIME_FIT = RESULT_DIR / "table_tnsm_2_2b_linear_fit.csv"
IN_MULTI = RESULT_DIR / "table_tnsm_2_3_multiseed_paper.csv"

# Outputs
OUT_TABLE_IV = RESULT_DIR / "paper_table_IV_rule_based_fixed_ucb_shields.csv"
OUT_TABLE_V = RESULT_DIR / "paper_table_V_runtime_overhead_scalability.csv"
OUT_APP_MULTI = RESULT_DIR / "appendix_table_multiseed_stability.csv"
OUT_FINDINGS = RESULT_DIR / "tnsm_extra_key_findings.txt"
OUT_REPORT = RESULT_DIR / "tnsm_2_4_aggregate_report.json"


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}\n"
            f"Please confirm the corresponding previous step has completed."
        )


def fmt(x, digits=4):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def load_inputs():
    for p in [IN_RULE, IN_RUNTIME_REAL, IN_RUNTIME_SYN, IN_RUNTIME_FIT, IN_MULTI]:
        check_file(p)

    rule = pd.read_csv(IN_RULE)
    runtime_real = pd.read_csv(IN_RUNTIME_REAL)
    runtime_syn = pd.read_csv(IN_RUNTIME_SYN)
    runtime_fit = pd.read_csv(IN_RUNTIME_FIT)
    multi = pd.read_csv(IN_MULTI)

    return rule, runtime_real, runtime_syn, runtime_fit, multi


def build_table_iv(rule):
    """
    Table IV: Rule-based and fixed-UCB runtime shield comparison.
    Keep aggregate paper-level rows only.
    """
    df = rule.copy()

    # Normalize column names from previous script.
    rename = {
        "Method": "Shielding method",
        "Controllers": "Controllers",
        "Max shielded unsafe (%)": "Max unsafe (%)",
        "Mean shielded unsafe (%)": "Mean unsafe (%)",
        "Min utility retention (%)": "Min utility retention (%)",
        "Mean action distance": "Mean action distance",
        "Max fallback (%)": "Max fallback (%)",
        "Max false-safe (%)": "Max false-safe (%)",
    }
    df = df.rename(columns=rename)

    keep = [
        "Shielding method",
        "Controllers",
        "Max unsafe (%)",
        "Mean unsafe (%)",
        "Min utility retention (%)",
        "Mean action distance",
        "Max fallback (%)",
        "Max false-safe (%)",
    ]
    df = df[keep].copy()

    method_order = {
        "Full-CertiTwin-q90": 0,
        "SLA-Rule-Shield": 1,
        "Fixed-UCB-1sigma-Shield": 2,
        "Fixed-UCB-1.64sigma-Shield": 3,
    }
    df["_order"] = df["Shielding method"].map(method_order).fillna(99)
    df = df.sort_values("_order").drop(columns=["_order"])

    for c in df.columns:
        if c not in ["Shielding method", "Controllers"]:
            df[c] = df[c].astype(float).round(4)

    return df


def build_table_v(runtime_real, runtime_syn, runtime_fit):
    """
    Table V: Runtime overhead and candidate-size scalability.
    Panel A: real replayed K=5 overhead.
    Panel B: synthetic K-scaling micro-benchmark.
    """
    # Panel A: use the most conservative row from real runtime results.
    # Since effective K is fixed at 5, choose max mean/P95/P99 over target K rows.
    real = runtime_real.copy()

    panel_a = pd.DataFrame([
        {
            "Panel": "A: Replayed finite-candidate setting",
            "Setting": "Real replayed candidate groups",
            "K": int(round(float(real["Mean effective K"].max()))),
            "Mean latency (ms)": float(real["Mean total latency (ms)"].max()),
            "P95 latency (ms)": float(real["P95 latency (ms)"].max()),
            "P99 latency (ms)": float(real["P99 latency (ms)"].max()),
            "Shielded unsafe (%)": float(real["Shielded unsafe (%)"].max()),
            "Utility retention (%)": float(real["Utility retention (%)"].min()),
            "Fallback (%)": float(real["Fallback (%)"].max()),
            "Note": "End-to-end Python implementation with table-based output handling",
        }
    ])

    # Panel B: keep representative synthetic K values.
    syn = runtime_syn.copy()
    if "Synthetic K" not in syn.columns:
        raise RuntimeError("Synthetic runtime table does not contain 'Synthetic K' column.")

    selected_k = [5, 20, 100, 500, 1000]
    syn = syn[syn["Synthetic K"].isin(selected_k)].copy()

    panel_b_rows = []
    for _, r in syn.iterrows():
        panel_b_rows.append(
            {
                "Panel": "B: Synthetic K-scaling micro-benchmark",
                "Setting": "Vectorized certificate/projection core",
                "K": int(r["Synthetic K"]),
                "Mean latency (ms)": float(r["Mean latency (ms)"]),
                "P95 latency (ms)": float(r["P95 latency (ms)"]),
                "P99 latency (ms)": float(r["P99 latency (ms)"]),
                "Shielded unsafe (%)": "",
                "Utility retention (%)": "",
                "Fallback (%)": "",
                "Note": "Runtime scaling only; not used for safety/utility comparison",
            }
        )

    panel_b = pd.DataFrame(panel_b_rows)

    out = pd.concat([panel_a, panel_b], ignore_index=True)

    for c in ["Mean latency (ms)", "P95 latency (ms)", "P99 latency (ms)"]:
        out[c] = out[c].apply(lambda x: fmt(x, 4))

    for c in ["Shielded unsafe (%)", "Utility retention (%)", "Fallback (%)"]:
        out[c] = out[c].apply(lambda x: "" if x == "" else fmt(x, 4))

    return out


def build_appendix_multiseed(multi):
    """
    Appendix table: multi-seed stability.
    Keep the complete paper table from Step 2.3.
    """
    df = multi.copy()

    # Prefer K=5 first, then K=3 stress test.
    if "K setting" in df.columns:
        df["_order_k"] = df["K setting"].map({5: 0, 3: 1}).fillna(99)
        controller_order = {"PF": 0, "Greedy-SLA": 1, "DT-Top1": 2}
        df["_order_c"] = df["Controller"].map(controller_order).fillna(99)
        df = df.sort_values(["_order_k", "_order_c"]).drop(columns=["_order_k", "_order_c"])

    return df


def make_findings(table_iv, table_v, multi, runtime_fit):
    """
    Generate paper-ready textual conclusions.
    """
    # Extract Table IV values
    rule_row = table_iv[table_iv["Shielding method"] == "SLA-Rule-Shield"].iloc[0]
    full_row = table_iv[table_iv["Shielding method"] == "Full-CertiTwin-q90"].iloc[0]
    ucb1_row = table_iv[table_iv["Shielding method"] == "Fixed-UCB-1sigma-Shield"].iloc[0]
    ucb164_row = table_iv[table_iv["Shielding method"] == "Fixed-UCB-1.64sigma-Shield"].iloc[0]

    # Runtime real
    real = table_v[table_v["Panel"].str.startswith("A")].iloc[0]

    # Fit
    fit = runtime_fit.iloc[0]
    slope = float(fit["slope_ms_per_candidate"])
    intercept = float(fit["intercept_ms"])
    r2 = float(fit["r2"])

    # Synthetic K=1000
    syn1000 = table_v[(table_v["Panel"].str.startswith("B")) & (table_v["K"] == 1000)].iloc[0]

    # Multi-seed K=5 key rows
    multi_k5 = multi[multi["K setting"].astype(int) == 5].copy()
    greedy = multi_k5[multi_k5["Controller"] == "Greedy-SLA"].iloc[0]
    dt = multi_k5[multi_k5["Controller"] == "DT-Top1"].iloc[0]
    pf = multi_k5[multi_k5["Controller"] == "PF"].iloc[0]

    text = f"""TNSM extra results summary

1. Rule-based and fixed-UCB runtime shields
- Full-CertiTwin-q90 achieves max shielded unsafe = {full_row['Max unsafe (%)']:.4f}% and max false-safe = {full_row['Max false-safe (%)']:.4f}% across the reconstructable controller-level candidates.
- SLA-Rule-Shield leaves max shielded unsafe = {rule_row['Max unsafe (%)']:.4f}% and max false-safe = {rule_row['Max false-safe (%)']:.4f}%, showing that mean-risk rules do not sufficiently control false-safe execution.
- Fixed-UCB-1sigma and Fixed-UCB-1.64sigma achieve max shielded unsafe = {ucb1_row['Max unsafe (%)']:.4f}% and {ucb164_row['Max unsafe (%)']:.4f}%, respectively, in this dataset. These variants are strong conservative shields, but they rely on manually chosen uncertainty multipliers rather than calibration-split residual adaptation.
- Recommended paper interpretation: SLA-Rule-Shield demonstrates the limitation of predicted-mean safety rules; Fixed-UCB demonstrates a strong but hand-tuned conservative alternative; CertiTwin-RSM provides a calibrated, data-adaptive certificate suitable for runtime SLA assurance.

2. Runtime overhead in the real replayed candidate setting
- The real replayed finite-candidate setting has effective K = {real['K']}.
- End-to-end Python implementation latency is mean = {real['Mean latency (ms)']} ms, P95 = {real['P95 latency (ms)']} ms, and P99 = {real['P99 latency (ms)']} ms.
- Shielded unsafe = {real['Shielded unsafe (%)']}%, utility retention = {real['Utility retention (%)']}%, and fallback = {real['Fallback (%)']}%.
- Recommended paper interpretation: under the actual replayed candidate construction, CertiTwin-RSM adds only millisecond-level execution-time overhead.

3. Synthetic candidate-size runtime scalability
- The vectorized certificate-filtering and projection micro-benchmark reaches K=1000 with mean latency = {syn1000['Mean latency (ms)']} ms and P99 latency = {syn1000['P99 latency (ms)']} ms.
- Linear fit of mean latency versus K gives slope = {slope:.8f} ms/candidate, intercept = {intercept:.6f} ms, and R^2 = {r2:.4f}.
- Recommended paper interpretation: the algorithmic core of finite-candidate certificate filtering and projection scales approximately linearly with candidate-set size. This synthetic benchmark is used only for runtime scaling, not for replayed safety or utility comparison.

4. Multi-seed stability
- Under real K=5, PF remains safe without intervention: raw/shielded unsafe = {pf['Raw unsafe (%)']} / {pf['Shielded unsafe (%)']}.
- Under real K=5, Greedy-SLA is consistently corrected: raw unsafe = {greedy['Raw unsafe (%)']}, shielded unsafe = {greedy['Shielded unsafe (%)']}, utility retention = {greedy['Utility retention (%)']}, fallback = {greedy['Fallback (%)']}.
- Under real K=5, DT-Top1 is aggressively unsafe before shielding but stable after shielding: raw unsafe = {dt['Raw unsafe (%)']}, shielded unsafe = {dt['Shielded unsafe (%)']}, utility retention = {dt['Utility retention (%)']}, fallback = {dt['Fallback (%)']}.
- Recommended paper interpretation: evaluation-level multi-seed analysis confirms that the runtime shielding behavior is not caused by a particular candidate ordering or tie-breaking choice.

5. How to place these results in the paper
- Main text Table IV: use paper_table_IV_rule_based_fixed_ucb_shields.csv.
- Main text Table V: use paper_table_V_runtime_overhead_scalability.csv.
- Appendix table: use appendix_table_multiseed_stability.csv.
- In the main text, state explicitly that the synthetic K-scaling benchmark is used only for runtime overhead analysis and does not change the replayed-candidate safety/utility evaluation.
"""
    return text


def main():
    rule, runtime_real, runtime_syn, runtime_fit, multi = load_inputs()

    table_iv = build_table_iv(rule)
    table_v = build_table_v(runtime_real, runtime_syn, runtime_fit)
    app_multi = build_appendix_multiseed(multi)
    findings = make_findings(table_iv, table_v, app_multi, runtime_fit)

    table_iv.to_csv(OUT_TABLE_IV, index=False, encoding="utf-8-sig")
    table_v.to_csv(OUT_TABLE_V, index=False, encoding="utf-8-sig")
    app_multi.to_csv(OUT_APP_MULTI, index=False, encoding="utf-8-sig")
    OUT_FINDINGS.write_text(findings, encoding="utf-8")

    report = {
        "status": "completed",
        "inputs": {
            "rule_shields": str(IN_RULE),
            "runtime_real": str(IN_RUNTIME_REAL),
            "runtime_synthetic": str(IN_RUNTIME_SYN),
            "runtime_fit": str(IN_RUNTIME_FIT),
            "multiseed": str(IN_MULTI),
        },
        "outputs": {
            "table_iv": str(OUT_TABLE_IV),
            "table_v": str(OUT_TABLE_V),
            "appendix_multiseed": str(OUT_APP_MULTI),
            "key_findings": str(OUT_FINDINGS),
        },
    }

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nTNSM-2.4 completed.")

    print("\n=== Paper Table IV ===")
    print(table_iv.to_string(index=False))

    print("\n=== Paper Table V ===")
    print(table_v.to_string(index=False))

    print("\n=== Appendix Multi-seed Table ===")
    print(app_multi.to_string(index=False))

    print(f"\nSaved:\n{OUT_TABLE_IV}\n{OUT_TABLE_V}\n{OUT_APP_MULTI}\n{OUT_FINDINGS}\n{OUT_REPORT}")


if __name__ == "__main__":
    main()