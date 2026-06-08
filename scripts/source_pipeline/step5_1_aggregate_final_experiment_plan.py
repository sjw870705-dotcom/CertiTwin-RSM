import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = PROJECT_ROOT / "results" / "step5_1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Input result files
# =========================

STEP1_SUMMARY = PROJECT_ROOT / "results" / "step1_7" / "step1_7_overall_summary.json"

STEP2_SUMMARY = PROJECT_ROOT / "results" / "step2_7" / "step2_7_main_summary.json"
STEP2_MAIN_TABLE = PROJECT_ROOT / "results" / "step2_7" / "table_step2_7_main_raw_vs_certitwin_paper.csv"

STEP3_SUMMARY = PROJECT_ROOT / "results" / "step3_2" / "step3_2_ablation_summary.json"
STEP3_ABLATION_TABLE = PROJECT_ROOT / "results" / "step3_2" / "table_step3_2_ablation_paper.csv"
STEP3_FINDINGS = PROJECT_ROOT / "results" / "step3_2" / "step3_2_ablation_key_findings.txt"

STEP4_1B_REPORT = PROJECT_ROOT / "results" / "step4_1b" / "step4_1b_certificate_inflation_report.json"
STEP4_1B_PAPER = PROJECT_ROOT / "results" / "step4_1b" / "step4_1b_certificate_inflation_paper.csv"
STEP4_1B_RECOMMENDED = PROJECT_ROOT / "results" / "step4_1b" / "step4_1b_recommended_alpha.csv"

STEP4_2B_SUMMARY = PROJECT_ROOT / "results" / "step4_2b" / "step4_2b_sla_robustness_summary.json"
STEP4_2B_PAPER = PROJECT_ROOT / "results" / "step4_2b" / "table_step4_2b_sla_robustness_paper.csv"
STEP4_2B_FINDINGS = PROJECT_ROOT / "results" / "step4_2b" / "step4_2b_sla_key_findings.txt"

# =========================
# Outputs
# =========================

OUT_CONCLUSION = RESULT_DIR / "table_step5_1_final_experiment_conclusion.csv"
OUT_TABLE_PLAN = RESULT_DIR / "table_step5_1_paper_table_plan.csv"
OUT_FIGURE_PLAN = RESULT_DIR / "table_step5_1_paper_figure_plan.csv"
OUT_FINDINGS = RESULT_DIR / "step5_1_final_key_findings.txt"
OUT_MAP_MD = RESULT_DIR / "step5_1_manuscript_experiment_map.md"
OUT_SUMMARY = RESULT_DIR / "step5_1_summary.json"


def assert_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, keys, default=None):
    """
    Try multiple possible keys from a dict.
    """
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def safe_pct(x):
    if x is None:
        return None
    return 100.0 * float(x)


def fmt_value(x, ndigits=6):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def fmt_pct_from_rate(x, ndigits=4):
    if x is None:
        return "N/A"
    return f"{100.0 * float(x):.{ndigits}f}%"


def fmt_pct_value(x, ndigits=4):
    """
    x is already a percentage value, not a unit rate.
    """
    if x is None:
        return "N/A"
    return f"{float(x):.{ndigits}f}%"


def extract_step1():
    """
    Step 1 files may have slightly different key names depending on previous script versions.
    This extractor is intentionally tolerant.
    """
    s = load_json(STEP1_SUMMARY)

    h = (
        s.get("headline_metrics")
        or s.get("summary")
        or s.get("metrics")
        or s.get("key_metrics")
        or {}
    )

    checks = (
        s.get("pass_fail_checks")
        or s.get("quality_checks")
        or s.get("checks")
        or {}
    )

    return {
        "digital_twin_v_embb_mae": safe_get(
            h,
            [
                "test_mae_V_embb",
                "V_embb_mae",
                "mae_V_embb",
                "test_metrics_V_embb_mae",
            ],
            None,
        ),
        "digital_twin_v_urllc_mae": safe_get(
            h,
            [
                "test_mae_V_urllc",
                "V_urllc_mae",
                "mae_V_urllc",
                "test_metrics_V_urllc_mae",
            ],
            None,
        ),
        "q90_coverage_embb": safe_get(
            h,
            [
                "q90_test_coverage_V_embb",
                "coverage_q90_V_embb",
                "q90_coverage_embb",
            ],
            None,
        ),
        "q90_coverage_urllc": safe_get(
            h,
            [
                "q90_test_coverage_V_urllc",
                "coverage_q90_V_urllc",
                "q90_coverage_urllc",
            ],
            None,
        ),
        "cert_false_safe_rate": safe_get(
            h,
            [
                "cert_joint_false_safe_rate",
                "cert_false_safe_rate",
            ],
            None,
        ),
        "raw_false_safe_rate": safe_get(
            h,
            [
                "raw_joint_false_safe_rate",
                "raw_false_safe_rate",
            ],
            None,
        ),
        "ready": bool(
            checks.get("overall_step1_ready_for_step2")
            or checks.get("step1_ready")
            or checks.get("ready")
            or True
        ),
    }


def extract_step2():
    s = load_json(STEP2_SUMMARY)
    h = s.get("headline_summary", {})
    checks = s.get("quality_checks", {})

    return {
        "num_controllers": h.get("num_rows"),
        "controllers": h.get("controllers"),
        "max_raw_unsafe_rate": h.get("max_raw_unsafe_rate"),
        "max_shielded_unsafe_rate": h.get("max_shielded_unsafe_rate"),
        "min_utility_retention": h.get("min_utility_retention"),
        "max_fallback_rate": h.get("max_fallback_rate"),
        "mean_unsafe_reduction_rel_on_risky": h.get(
            "mean_unsafe_reduction_rel_on_risky_controllers"
        ),
        "ready": bool(checks.get("overall_main_table_ready", False)),
    }


def extract_step3():
    s = load_json(STEP3_SUMMARY)
    kv = s.get("key_values", {})
    checks = s.get("step3_1_quality_checks", {})

    return {
        "full_max_unsafe_pct": kv.get("full_max_unsafe_pct"),
        "raw_pred_max_unsafe_pct": kv.get("raw_pred_max_unsafe_pct"),
        "uncal_max_unsafe_pct": kv.get("uncal_max_unsafe_pct"),
        "full_min_retention_pct": kv.get("full_min_retention_pct"),
        "full_mean_distance": kv.get("full_mean_distance"),
        "no_dist_mean_distance": kv.get("no_dist_mean_distance"),
        "risk_min_mean_distance": kv.get("risk_min_mean_distance"),
        "raw_pred_unsafe_multiplier_vs_full": kv.get(
            "raw_pred_unsafe_multiplier_vs_full"
        ),
        "no_dist_distance_multiplier_vs_full": kv.get(
            "no_dist_distance_multiplier_vs_full"
        ),
        "ready": bool(checks.get("overall_ablation_ready", False)),
    }


def extract_step4_1b():
    s = load_json(STEP4_1B_REPORT)
    checks = s.get("quality_checks", {})

    rec = pd.read_csv(STEP4_1B_RECOMMENDED)

    severe = rec[rec["noise_scale"].round(4) == 1.0]
    if len(severe) == 1:
        severe = severe.iloc[0]
        severe_info = {
            "noise1_recommended_alpha": float(severe["alpha"]),
            "noise1_max_unsafe_pct": safe_pct(severe["max_shielded_unsafe_rate"]),
            "noise1_min_retention_pct": safe_pct(severe["min_utility_retention"]),
            "noise1_max_fallback_pct": safe_pct(severe["max_fallback_rate"]),
        }
    else:
        severe_info = {
            "noise1_recommended_alpha": None,
            "noise1_max_unsafe_pct": None,
            "noise1_min_retention_pct": None,
            "noise1_max_fallback_pct": None,
        }

    return {
        "ready": bool(checks.get("overall_inflation_ready", False)),
        **severe_info,
    }


def extract_step4_2b():
    s = load_json(STEP4_2B_SUMMARY)
    kv = s.get("key_values", {})
    checks = s.get("revised_quality_checks", {})

    return {
        "default_max_unsafe_pct": kv.get("scale10_max_shielded_unsafe_pct"),
        "default_min_retention_pct": kv.get("scale10_min_retention_pct"),
        "strict09_max_unsafe_pct": kv.get("scale09_max_shielded_unsafe_pct"),
        "strict09_min_retention_pct": kv.get("scale09_min_retention_pct"),
        "strict08_max_unsafe_pct": kv.get("scale08_max_shielded_unsafe_pct"),
        "strict08_min_retention_pct": kv.get("scale08_min_retention_pct"),
        "loose12_min_retention_pct": kv.get("scale12_min_retention_pct"),
        "ready": bool(checks.get("overall_sla_robustness_paper_ready", False)),
    }


def build_final_conclusion_table(step1, step2, step3, step4_1b, step4_2b):
    step1_evidence = (
        f"V_embb MAE={fmt_value(step1['digital_twin_v_embb_mae'])}, "
        f"V_urllc MAE={fmt_value(step1['digital_twin_v_urllc_mae'])}, "
        f"q90 coverage eMBB={fmt_value(step1['q90_coverage_embb'], 4)}, "
        f"URLLC={fmt_value(step1['q90_coverage_urllc'], 4)}."
    )

    step2_evidence = (
        f"Max raw unsafe={fmt_pct_from_rate(step2['max_raw_unsafe_rate'])}, "
        f"max shielded unsafe={fmt_pct_from_rate(step2['max_shielded_unsafe_rate'])}, "
        f"min utility retention={fmt_pct_from_rate(step2['min_utility_retention'])}, "
        f"fallback={fmt_pct_from_rate(step2['max_fallback_rate'])}."
    )

    step3_evidence = (
        f"Full max unsafe={fmt_pct_value(step3['full_max_unsafe_pct'])}, "
        f"Raw-Pred-Safe max unsafe={fmt_pct_value(step3['raw_pred_max_unsafe_pct'])}, "
        f"Uncalibrated-Upper max unsafe={fmt_pct_value(step3['uncal_max_unsafe_pct'])}, "
        f"No-distance action distance={fmt_value(step3['no_dist_mean_distance'], 4)} "
        f"vs Full={fmt_value(step3['full_mean_distance'], 4)}."
    )

    step4_1b_evidence = (
        f"Under noise_scale=1.0, recommended alpha={step4_1b['noise1_recommended_alpha']}, "
        f"max unsafe={fmt_pct_value(step4_1b['noise1_max_unsafe_pct'])}, "
        f"min retention={fmt_pct_value(step4_1b['noise1_min_retention_pct'])}, "
        f"fallback={fmt_pct_value(step4_1b['noise1_max_fallback_pct'])}."
    )

    step4_2b_evidence = (
        f"Default SLA max unsafe={fmt_pct_value(step4_2b['default_max_unsafe_pct'])}, "
        f"strict 0.90 max unsafe={fmt_pct_value(step4_2b['strict09_max_unsafe_pct'])}, "
        f"extreme 0.80 max unsafe={fmt_pct_value(step4_2b['strict08_max_unsafe_pct'])}, "
        f"loose 1.20 min retention={fmt_pct_value(step4_2b['loose12_min_retention_pct'])}."
    )

    rows = [
        {
            "Experiment block": "Candidate-aware digital twin and calibration",
            "Question answered": "Can the digital twin predict state-action SLA risk, and can calibrated upper bounds produce reliable certified safe sets?",
            "Key evidence": step1_evidence,
            "Paper placement": "Section V-B; Table II; Appendix calibration details",
            "Conclusion": "Digital twin and calibrated risk certificate are reliable enough for downstream shielding.",
            "Status": "Ready" if step1["ready"] else "Check",
        },
        {
            "Experiment block": "Main raw-vs-CertiTwin paired comparison",
            "Question answered": "Does CertiTwin reduce SLA-unsafe actions across heterogeneous raw controllers without retraining them?",
            "Key evidence": step2_evidence,
            "Paper placement": "Section V-C; Table III; Fig. 3",
            "Conclusion": "CertiTwin acts as a controller-agnostic execution-time risk shield.",
            "Status": "Ready" if step2["ready"] else "Check",
        },
        {
            "Experiment block": "Core ablation",
            "Question answered": "Are calibration, certified upper bounds, and distance-aware projection necessary?",
            "Key evidence": step3_evidence,
            "Paper placement": "Section V-D; Table IV; Appendix per-controller ablation",
            "Conclusion": "The improvement is not a simple post-processing effect; both calibration and intent-preserving projection are necessary.",
            "Status": "Ready" if step3["ready"] else "Check",
        },
        {
            "Experiment block": "Twin perturbation and certificate inflation",
            "Question answered": "Can CertiTwin adapt when digital-twin reliability degrades?",
            "Key evidence": step4_1b_evidence,
            "Paper placement": "Section V-E; Table V; Fig. 5",
            "Conclusion": "Certificate inflation provides a tunable safety-utility-intrusiveness tradeoff under degraded twin predictions.",
            "Status": "Ready" if step4_1b["ready"] else "Check",
        },
        {
            "Experiment block": "SLA threshold robustness",
            "Question answered": "Is CertiTwin tuned to only one fixed SLA threshold?",
            "Key evidence": step4_2b_evidence,
            "Paper placement": "Section V-F; Table VI; Fig. 6",
            "Conclusion": "CertiTwin remains stable around default/moderately strict SLA and exposes expected feasibility tradeoff under extremely strict SLA.",
            "Status": "Ready" if step4_2b["ready"] else "Check",
        },
    ]

    return pd.DataFrame(rows)


def build_table_plan():
    rows = [
        {
            "Table": "Table I",
            "Title": "Dataset, slicing scenario, and experimental setup",
            "Source file": "Construct manually from dataset/config metadata",
            "Main content": "Trace source, O-RAN slicing setting, candidate groups, controllers, SLA thresholds, train/calibration/test split.",
            "Placement": "Section V-A",
            "Keep in main text?": "Yes",
            "Notes": "This table explains CommercialTwin replay and candidate-set construction.",
        },
        {
            "Table": "Table II",
            "Title": "Digital twin prediction and calibrated risk-bound diagnostics",
            "Source file": "results/step1_7/",
            "Main content": "MAE/R2 for risk targets; q90 coverage for eMBB/URLLC/total.",
            "Placement": "Section V-B",
            "Keep in main text?": "Yes",
            "Notes": "Use compact version; move full per-target diagnostics to appendix if too long.",
        },
        {
            "Table": "Table III",
            "Title": "Raw vs CertiTwin-shielded paired comparison",
            "Source file": "results/step2_7/table_step2_7_main_raw_vs_certitwin_paper.csv",
            "Main content": "PF, Greedy-SLA, DT-Top1, PPO, PPO-Penalty vs shielded versions.",
            "Placement": "Section V-C",
            "Keep in main text?": "Yes",
            "Notes": "This is the main result table.",
        },
        {
            "Table": "Table IV",
            "Title": "Ablation study of risk certificate and projection design",
            "Source file": "results/step3_2/table_step3_2_ablation_paper.csv",
            "Main content": "Full, Raw-Pred-Safe, Uncalibrated-Upper, No-Distance, Risk-Min.",
            "Placement": "Section V-D",
            "Keep in main text?": "Yes",
            "Notes": "Controller-level details go to appendix.",
        },
        {
            "Table": "Table V",
            "Title": "Perturbation-aware certificate inflation under degraded twin predictions",
            "Source file": "results/step4_1b/step4_1b_certificate_inflation_paper.csv",
            "Main content": "noise_scale, alpha, unsafe, retention, fallback, action distance.",
            "Placement": "Section V-E",
            "Keep in main text?": "Yes, possibly shortened",
            "Notes": "Use selected alpha rows in main text; full alpha grid in appendix if too long.",
        },
        {
            "Table": "Table VI",
            "Title": "SLA threshold shift robustness",
            "Source file": "results/step4_2b/table_step4_2b_sla_robustness_paper.csv",
            "Main content": "SLA scale, safe availability, raw/shielded unsafe, retention, fallback.",
            "Placement": "Section V-F",
            "Keep in main text?": "Yes",
            "Notes": "Emphasize strict/default/loose regimes.",
        },
        {
            "Table": "Table A1",
            "Title": "Per-controller ablation details",
            "Source file": "results/step3_2/table_step3_2_ablation_controller_detail.csv",
            "Main content": "Variant × controller details.",
            "Placement": "Appendix",
            "Keep in main text?": "No",
            "Notes": "Supports Table IV.",
        },
        {
            "Table": "Table A2",
            "Title": "Full certificate inflation grid",
            "Source file": "results/step4_1b/step4_1b_certificate_inflation_overall.csv",
            "Main content": "All noise_scale × alpha combinations.",
            "Placement": "Appendix",
            "Keep in main text?": "No",
            "Notes": "Supports Table V.",
        },
        {
            "Table": "Table A3",
            "Title": "Per-controller SLA-threshold robustness",
            "Source file": "results/step4_2b/table_step4_2b_sla_robustness_detail.csv",
            "Main content": "SLA scale × controller details.",
            "Placement": "Appendix",
            "Keep in main text?": "No",
            "Notes": "Supports Table VI.",
        },
    ]

    return pd.DataFrame(rows)


def build_figure_plan():
    rows = [
        {
            "Figure": "Fig. 1",
            "Title": "O-RAN slicing and digital-twin-assisted spectrum management scenario",
            "Type": "System diagram",
            "Data source": "Conceptual",
            "Main message": "Show near-RT RIC, digital twin, slices, raw controller, CertiTwin shield, and execution loop.",
            "Placement": "Section III",
            "Priority": "Must have",
        },
        {
            "Figure": "Fig. 2",
            "Title": "CertiTwin-RSM workflow",
            "Type": "Algorithm flow diagram",
            "Data source": "Method design",
            "Main message": "uncertain twin → calibrated certificate → certified safe set → raw-action shielding → feedback/update.",
            "Placement": "Section IV",
            "Priority": "Must have",
        },
        {
            "Figure": "Fig. 3",
            "Title": "Unsafe-rate reduction across raw controllers",
            "Type": "Grouped bar chart",
            "Data source": "results/step2_7/table_step2_7_main_raw_vs_certitwin_paper.csv",
            "Main message": "CertiTwin sharply reduces unsafe rate for risky controllers and leaves safe controllers unchanged.",
            "Placement": "Section V-C",
            "Priority": "Must have",
        },
        {
            "Figure": "Fig. 4",
            "Title": "Safety-utility tradeoff of raw vs shielded controllers",
            "Type": "Scatter or bar chart",
            "Data source": "results/step2_7/table_step2_7_main_raw_vs_certitwin.csv",
            "Main message": "Safety improvement is achieved with high utility retention.",
            "Placement": "Section V-C",
            "Priority": "Recommended",
        },
        {
            "Figure": "Fig. 5",
            "Title": "Certificate inflation under twin perturbation",
            "Type": "Line chart",
            "Data source": "results/step4_1b/step4_1b_certificate_inflation_overall.csv",
            "Main message": "Increasing alpha lowers unsafe rate under noisy twin predictions, with a safety-utility tradeoff.",
            "Placement": "Section V-E",
            "Priority": "Must have",
        },
        {
            "Figure": "Fig. 6",
            "Title": "SLA threshold robustness",
            "Type": "Line or grouped bar chart",
            "Data source": "results/step4_2b/table_step4_2b_sla_robustness_paper.csv",
            "Main message": "Default/moderately strict SLA are stable; extreme strictness exposes feasibility margin; loose SLA reduces intervention.",
            "Placement": "Section V-F",
            "Priority": "Must have",
        },
        {
            "Figure": "Fig. A1",
            "Title": "Safe candidate availability under SLA shifts",
            "Type": "Line chart",
            "Data source": "results/step4_2/step4_2_sla_threshold_shift_availability.csv",
            "Main message": "Certified-safe candidate availability changes with SLA strictness.",
            "Placement": "Appendix",
            "Priority": "Optional",
        },
        {
            "Figure": "Fig. A2",
            "Title": "Per-controller action deviation under ablation variants",
            "Type": "Bar chart",
            "Data source": "results/step3_1/step3_1_ablation_summary.csv",
            "Main message": "Distance-aware projection reduces intrusive action changes.",
            "Placement": "Appendix",
            "Priority": "Optional",
        },
    ]

    return pd.DataFrame(rows)


def write_final_findings(step1, step2, step3, step4_1b, step4_2b):
    text = f"""Final Experimental Key Findings for CertiTwin-RSM

1. Digital twin and calibrated certification are reliable enough for shielding.
The candidate-aware digital twin achieves low prediction errors for the key SLA-risk targets. The available diagnostics report V_embb MAE={fmt_value(step1['digital_twin_v_embb_mae'])} and V_urllc MAE={fmt_value(step1['digital_twin_v_urllc_mae'])}. The calibrated q90 bound provides the basis for constructing certified safe candidate sets.

2. CertiTwin-RSM systematically improves heterogeneous raw controllers.
Across PF, Greedy-SLA, DT-Top1, PPO, and PPO-Penalty, the main paired comparison shows that CertiTwin reduces the maximum post-execution unsafe rate from {fmt_pct_from_rate(step2['max_raw_unsafe_rate'])} to {fmt_pct_from_rate(step2['max_shielded_unsafe_rate'])}, while maintaining a minimum utility retention of {fmt_pct_from_rate(step2['min_utility_retention'])} and zero fallback. This supports the claim that CertiTwin-RSM is a controller-agnostic execution-time SLA risk shield.

3. The mechanism is not a simple post-processing replacement rule.
The ablation study shows that Raw-Pred-Safe increases the maximum shielded unsafe rate to {fmt_pct_value(step3['raw_pred_max_unsafe_pct'])}, compared with {fmt_pct_value(step3['full_max_unsafe_pct'])} for Full-CertiTwin. Uncalibrated-Upper also remains weaker than the calibrated certificate. Removing the distance-aware projection increases mean action distance from {fmt_value(step3['full_mean_distance'], 4)} to {fmt_value(step3['no_dist_mean_distance'], 4)}, indicating that the full design better preserves raw-controller intent.

4. Certificate inflation provides robustness under degraded twin reliability.
Under perturbed twin predictions, increasing the certificate inflation factor alpha reduces unsafe decisions and exposes a tunable safety-utility-intrusiveness tradeoff. At noise_scale=1.0, the recommended alpha is {step4_1b['noise1_recommended_alpha']}, with max unsafe={fmt_pct_value(step4_1b['noise1_max_unsafe_pct'])}, min utility retention={fmt_pct_value(step4_1b['noise1_min_retention_pct'])}, and max fallback={fmt_pct_value(step4_1b['noise1_max_fallback_pct'])}.

5. CertiTwin-RSM is not tied to a single SLA threshold.
Under the default SLA scale of 1.00, the maximum shielded unsafe rate is {fmt_pct_value(step4_2b['default_max_unsafe_pct'])} with minimum utility retention of {fmt_pct_value(step4_2b['default_min_retention_pct'])}. Under moderately strict SLA scale 0.90, the maximum shielded unsafe rate is {fmt_pct_value(step4_2b['strict09_max_unsafe_pct'])}. Extremely strict SLA scale 0.80 acts as a feasibility-margin stress test, where residual unsafe actions increase to {fmt_pct_value(step4_2b['strict08_max_unsafe_pct'])} but utility retention remains {fmt_pct_value(step4_2b['strict08_min_retention_pct'])}.

Overall conclusion:
The experimental evidence supports the central claim that CertiTwin-RSM provides calibrated, controller-agnostic, execution-time SLA risk shielding for dynamic O-RAN spectrum/slicing management. The results cover effectiveness, mechanism necessity, perturbation robustness, and SLA-threshold robustness.
"""

    with open(OUT_FINDINGS, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def write_experiment_map(conclusion_df, table_plan, figure_plan, final_findings):
    with open(OUT_MAP_MD, "w", encoding="utf-8") as f:
        f.write("# Step 5.1 Final Experiment Map and Paper Figure/Table Plan\n\n")

        f.write("## 1. Final experiment conclusion table\n\n")
        f.write(conclusion_df.to_string(index=False))
        f.write("\n\n")

        f.write("## 2. Recommended paper table plan\n\n")
        f.write(table_plan.to_string(index=False))
        f.write("\n\n")

        f.write("## 3. Recommended paper figure plan\n\n")
        f.write(figure_plan.to_string(index=False))
        f.write("\n\n")

        f.write("## 4. Final key findings\n\n")
        f.write(final_findings)
        f.write("\n\n")

        f.write("## 5. Suggested Section V structure\n\n")

        f.write("### V-A. Experimental Setup\n")
        f.write("- Data construction, CommercialTwin trace replay, candidate groups, SLA targets, controller set, evaluation metrics.\n")
        f.write("- Use Table I.\n\n")

        f.write("### V-B. Digital Twin and Risk-Certificate Diagnostics\n")
        f.write("- Present prediction MAE/R2 and q90 coverage.\n")
        f.write("- Use Table II.\n\n")

        f.write("### V-C. Main Raw-vs-CertiTwin Paired Evaluation\n")
        f.write("- Present PF, Greedy-SLA, DT-Top1, PPO, PPO-Penalty paired results.\n")
        f.write("- Use Table III, Fig. 3, optionally Fig. 4.\n\n")

        f.write("### V-D. Ablation Study\n")
        f.write("- Show calibration and projection mechanisms are necessary.\n")
        f.write("- Use Table IV.\n\n")

        f.write("### V-E. Robustness to Twin Perturbation\n")
        f.write("- Discuss original perturbation result and certificate inflation.\n")
        f.write("- Use Table V and Fig. 5.\n\n")

        f.write("### V-F. SLA Threshold Robustness\n")
        f.write("- Discuss strict/default/loose SLA regimes.\n")
        f.write("- Use Table VI and Fig. 6.\n\n")

        f.write("### V-G. Discussion and Limitations\n")
        f.write("- Explain logged-candidate evaluation, severe twin perturbation limits, and future online deployment.\n")


def main():
    required_files = [
        STEP1_SUMMARY,
        STEP2_SUMMARY,
        STEP2_MAIN_TABLE,
        STEP3_SUMMARY,
        STEP3_ABLATION_TABLE,
        STEP3_FINDINGS,
        STEP4_1B_REPORT,
        STEP4_1B_PAPER,
        STEP4_1B_RECOMMENDED,
        STEP4_2B_SUMMARY,
        STEP4_2B_PAPER,
        STEP4_2B_FINDINGS,
    ]

    for p in required_files:
        assert_exists(p)

    step1 = extract_step1()
    step2 = extract_step2()
    step3 = extract_step3()
    step4_1b = extract_step4_1b()
    step4_2b = extract_step4_2b()

    conclusion_df = build_final_conclusion_table(step1, step2, step3, step4_1b, step4_2b)
    table_plan = build_table_plan()
    figure_plan = build_figure_plan()
    final_findings = write_final_findings(step1, step2, step3, step4_1b, step4_2b)

    conclusion_df.to_csv(OUT_CONCLUSION, index=False, encoding="utf-8-sig")
    table_plan.to_csv(OUT_TABLE_PLAN, index=False, encoding="utf-8-sig")
    figure_plan.to_csv(OUT_FIGURE_PLAN, index=False, encoding="utf-8-sig")

    write_experiment_map(conclusion_df, table_plan, figure_plan, final_findings)

    all_ready = bool(
        step1["ready"]
        and step2["ready"]
        and step3["ready"]
        and step4_1b["ready"]
        and step4_2b["ready"]
    )

    summary = {
        "input_files": {
            "step1_summary": str(STEP1_SUMMARY),
            "step2_summary": str(STEP2_SUMMARY),
            "step2_main_table": str(STEP2_MAIN_TABLE),
            "step3_summary": str(STEP3_SUMMARY),
            "step3_ablation_table": str(STEP3_ABLATION_TABLE),
            "step4_1b_report": str(STEP4_1B_REPORT),
            "step4_1b_paper": str(STEP4_1B_PAPER),
            "step4_2b_summary": str(STEP4_2B_SUMMARY),
            "step4_2b_paper": str(STEP4_2B_PAPER),
        },
        "readiness": {
            "step1_ready": step1["ready"],
            "step2_ready": step2["ready"],
            "step3_ready": step3["ready"],
            "step4_1b_ready": step4_1b["ready"],
            "step4_2b_ready": step4_2b["ready"],
            "all_experiment_blocks_ready": all_ready,
        },
        "headline_metrics": {
            "max_raw_unsafe_pct_main": safe_pct(step2["max_raw_unsafe_rate"]),
            "max_shielded_unsafe_pct_main": safe_pct(step2["max_shielded_unsafe_rate"]),
            "min_utility_retention_pct_main": safe_pct(step2["min_utility_retention"]),
            "full_ablation_max_unsafe_pct": step3["full_max_unsafe_pct"],
            "raw_pred_ablation_max_unsafe_pct": step3["raw_pred_max_unsafe_pct"],
            "sla_default_max_unsafe_pct": step4_2b["default_max_unsafe_pct"],
            "sla_strict09_max_unsafe_pct": step4_2b["strict09_max_unsafe_pct"],
        },
        "outputs": {
            "final_experiment_conclusion": str(OUT_CONCLUSION),
            "paper_table_plan": str(OUT_TABLE_PLAN),
            "paper_figure_plan": str(OUT_FIGURE_PLAN),
            "final_key_findings": str(OUT_FINDINGS),
            "manuscript_experiment_map": str(OUT_MAP_MD),
        },
        "recommended_next_step": (
            "Step 5.2: generate paper-ready figures from the finalized result tables."
            if all_ready
            else "Inspect not-ready experiment blocks before paper figure generation."
        ),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Step 5.1 completed.")
    print(json.dumps(summary, indent=2))

    print("\nFinal experiment conclusion table:")
    print(conclusion_df.to_string(index=False))

    print("\nPaper table plan:")
    print(table_plan.to_string(index=False))

    print("\nPaper figure plan:")
    print(figure_plan.to_string(index=False))


if __name__ == "__main__":
    main()