import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STEP3_1_DIR = PROJECT_ROOT / "results" / "step3_1"
RESULT_DIR = PROJECT_ROOT / "results" / "step3_2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_OVERALL = STEP3_1_DIR / "step3_1_ablation_overall.csv"
INPUT_SUMMARY = STEP3_1_DIR / "step3_1_ablation_summary.csv"
INPUT_REPORT = STEP3_1_DIR / "step3_1_ablation_report.json"

OUT_PAPER = RESULT_DIR / "table_step3_2_ablation_paper.csv"
OUT_DETAIL = RESULT_DIR / "table_step3_2_ablation_controller_detail.csv"
OUT_FINDINGS = RESULT_DIR / "step3_2_ablation_key_findings.txt"
OUT_JSON = RESULT_DIR / "step3_2_ablation_summary.json"
OUT_MD = RESULT_DIR / "step3_2_report.md"


VARIANT_ORDER = {
    "Full-CertiTwin": 1,
    "Raw-Pred-Safe": 2,
    "Uncalibrated-Upper": 3,
    "No-Distance-Projection": 4,
    "Risk-Min-Projection": 5,
}

CONTROLLER_ORDER = {
    "PF": 1,
    "Greedy-SLA": 2,
    "DT-Top1": 3,
    "PPO": 4,
    "PPO-Penalty": 5,
}


VARIANT_DESCRIPTIONS = {
    "Full-CertiTwin": "Calibrated q90 upper bound + certified safe set + distance-aware projection",
    "Raw-Pred-Safe": "Predicted mean safety only, without calibration",
    "Uncalibrated-Upper": "Mean + 1.0 sigma upper bound, without calibration quantile",
    "No-Distance-Projection": "Certified safe set, but removes distance-to-raw-action term",
    "Risk-Min-Projection": "Certified safe set, selects the minimum upper-risk action",
}


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def pct(x):
    return 100.0 * float(x)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_paper_table(overall):
    table = overall.copy()

    table["Variant description"] = table["variant"].map(VARIANT_DESCRIPTIONS)

    table["Max shielded unsafe (%)"] = table["max_shielded_unsafe_rate"].apply(pct)
    table["Mean shielded unsafe (%)"] = table["mean_shielded_unsafe_rate"].apply(pct)
    table["Min utility retention (%)"] = table["min_utility_retention"].apply(pct)
    table["Mean utility retention (%)"] = table["mean_utility_retention"].apply(pct)
    table["Mean action distance"] = table["mean_action_distance"]
    table["Max fallback (%)"] = table["max_fallback_rate"].apply(pct)
    table["Mean unsafe reduction on risky controllers (%)"] = table[
        "mean_unsafe_reduction_rel_on_risky"
    ].apply(pct)

    table["variant_order"] = table["variant"].map(VARIANT_ORDER).fillna(99)

    table = table.sort_values("variant_order").reset_index(drop=True)

    paper = table[
        [
            "variant",
            "Variant description",
            "Max shielded unsafe (%)",
            "Mean shielded unsafe (%)",
            "Mean unsafe reduction on risky controllers (%)",
            "Min utility retention (%)",
            "Mean utility retention (%)",
            "Mean action distance",
            "Max fallback (%)",
        ]
    ].copy()

    paper = paper.rename(columns={"variant": "Variant"})

    numeric_cols = [
        c for c in paper.columns
        if c not in ["Variant", "Variant description"]
    ]

    for c in numeric_cols:
        paper[c] = paper[c].astype(float).round(4)

    return paper


def make_controller_detail_table(summary):
    detail = summary.copy()

    detail["variant_order"] = detail["variant"].map(VARIANT_ORDER).fillna(99)
    detail["controller_order"] = detail["controller"].map(CONTROLLER_ORDER).fillna(99)

    detail = detail.sort_values(["variant_order", "controller_order"]).reset_index(drop=True)

    detail["Raw unsafe (%)"] = detail["raw_true_unsafe_rate"].apply(pct)
    detail["Shielded unsafe (%)"] = detail["shielded_true_unsafe_rate"].apply(pct)
    detail["Unsafe reduction (%)"] = detail["unsafe_reduction_rel"].apply(pct)
    detail["Utility retention (%)"] = detail["utility_retention"].apply(pct)
    detail["Action changed (%)"] = detail["action_changed_rate"].apply(pct)
    detail["Pass-through (%)"] = detail["pass_through_rate"].apply(pct)
    detail["Fallback (%)"] = detail["fallback_rate"].apply(pct)
    detail["Mean action distance"] = detail["mean_action_distance"]

    out = detail[
        [
            "variant",
            "controller",
            "Raw unsafe (%)",
            "Shielded unsafe (%)",
            "Unsafe reduction (%)",
            "Utility retention (%)",
            "Action changed (%)",
            "Pass-through (%)",
            "Mean action distance",
            "Fallback (%)",
        ]
    ].copy()

    out = out.rename(
        columns={
            "variant": "Variant",
            "controller": "Controller",
        }
    )

    for c in out.columns:
        if c not in ["Variant", "Controller"]:
            out[c] = out[c].astype(float).round(4)

    return out


def extract_key_values(overall):
    def row_of(variant):
        row = overall[overall["variant"] == variant]
        if len(row) != 1:
            raise RuntimeError(f"Cannot find unique row for variant: {variant}")
        return row.iloc[0]

    full = row_of("Full-CertiTwin")
    raw_pred = row_of("Raw-Pred-Safe")
    uncal = row_of("Uncalibrated-Upper")
    no_dist = row_of("No-Distance-Projection")
    risk_min = row_of("Risk-Min-Projection")

    values = {
        "full_max_unsafe_pct": pct(full["max_shielded_unsafe_rate"]),
        "full_mean_unsafe_pct": pct(full["mean_shielded_unsafe_rate"]),
        "full_min_retention_pct": pct(full["min_utility_retention"]),
        "full_mean_retention_pct": pct(full["mean_utility_retention"]),
        "full_mean_distance": float(full["mean_action_distance"]),
        "full_max_fallback_pct": pct(full["max_fallback_rate"]),

        "raw_pred_max_unsafe_pct": pct(raw_pred["max_shielded_unsafe_rate"]),
        "raw_pred_min_retention_pct": pct(raw_pred["min_utility_retention"]),
        "raw_pred_mean_distance": float(raw_pred["mean_action_distance"]),

        "uncal_max_unsafe_pct": pct(uncal["max_shielded_unsafe_rate"]),
        "uncal_min_retention_pct": pct(uncal["min_utility_retention"]),
        "uncal_mean_distance": float(uncal["mean_action_distance"]),

        "no_dist_max_unsafe_pct": pct(no_dist["max_shielded_unsafe_rate"]),
        "no_dist_min_retention_pct": pct(no_dist["min_utility_retention"]),
        "no_dist_mean_distance": float(no_dist["mean_action_distance"]),

        "risk_min_max_unsafe_pct": pct(risk_min["max_shielded_unsafe_rate"]),
        "risk_min_min_retention_pct": pct(risk_min["min_utility_retention"]),
        "risk_min_mean_distance": float(risk_min["mean_action_distance"]),
    }

    values["raw_pred_unsafe_multiplier_vs_full"] = (
        values["raw_pred_max_unsafe_pct"] / max(values["full_max_unsafe_pct"], 1e-12)
    )

    values["uncal_unsafe_multiplier_vs_full"] = (
        values["uncal_max_unsafe_pct"] / max(values["full_max_unsafe_pct"], 1e-12)
    )

    values["no_dist_distance_multiplier_vs_full"] = (
        values["no_dist_mean_distance"] / max(values["full_mean_distance"], 1e-12)
    )

    values["risk_min_distance_multiplier_vs_full"] = (
        values["risk_min_mean_distance"] / max(values["full_mean_distance"], 1e-12)
    )

    return values


def write_key_findings(values):
    text = f"""Step 3.2 Ablation Key Findings

1. Calibration is necessary.
Full-CertiTwin achieves a maximum shielded unsafe rate of {values['full_max_unsafe_pct']:.4f}%, whereas Raw-Pred-Safe reaches {values['raw_pred_max_unsafe_pct']:.4f}%. This means that using predicted mean safety alone leaves substantially more unsafe actions after shielding. The uncalibrated mean-plus-sigma variant reduces the maximum unsafe rate to {values['uncal_max_unsafe_pct']:.4f}%, but it still remains higher than the calibrated q90 upper-bound version.

2. The complete CertiTwin design provides the most reliable safety.
Full-CertiTwin keeps the mean shielded unsafe rate at {values['full_mean_unsafe_pct']:.4f}% with zero fallback, while maintaining a minimum utility retention of {values['full_min_retention_pct']:.4f}%. This confirms that the calibrated certified safe set is not overly conservative.

3. Distance-aware projection preserves raw-controller intent.
No-Distance-Projection and Risk-Min-Projection achieve similarly low unsafe rates, but their mean action distances are {values['no_dist_mean_distance']:.4f} and {values['risk_min_mean_distance']:.4f}, respectively, compared with {values['full_mean_distance']:.4f} for Full-CertiTwin. Thus, removing the distance term makes the shield more intrusive even when safety is maintained.

4. Risk-only projection is not the preferred operating point.
Risk-Min-Projection can keep actions safe in this logged-candidate setting, but it changes actions much more aggressively. Therefore, Full-CertiTwin provides a better safety-intent tradeoff: it enforces certified SLA safety while staying closer to the raw controller decision.

Recommended paper placement:
- Main text: use table_step3_2_ablation_paper.csv.
- Appendix: use table_step3_2_ablation_controller_detail.csv.
- Main explanation: emphasize calibration, certified upper bound, and distance-aware projection.
"""
    with open(OUT_FINDINGS, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def make_report_md(paper_table, detail_table, values, report_json):
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Step 3.2 Paper-Ready Ablation Tables and Interpretation\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "This step converts the Step 3.1 shield-variant ablation into paper-ready tables. "
            "The main text should use the variant-level table, while the appendix can report "
            "controller-level details.\n\n"
        )

        f.write("## Main text ablation table\n\n")
        f.write(paper_table.to_string(index=False))
        f.write("\n\n")

        f.write("## Key findings\n\n")
        f.write(f"- Full-CertiTwin max shielded unsafe rate: {values['full_max_unsafe_pct']:.4f}%\n")
        f.write(f"- Raw-Pred-Safe max shielded unsafe rate: {values['raw_pred_max_unsafe_pct']:.4f}%\n")
        f.write(f"- Uncalibrated-Upper max shielded unsafe rate: {values['uncal_max_unsafe_pct']:.4f}%\n")
        f.write(f"- Full-CertiTwin min utility retention: {values['full_min_retention_pct']:.4f}%\n")
        f.write(f"- Full-CertiTwin mean action distance: {values['full_mean_distance']:.4f}\n")
        f.write(f"- No-Distance-Projection mean action distance: {values['no_dist_mean_distance']:.4f}\n")
        f.write(f"- Risk-Min-Projection mean action distance: {values['risk_min_mean_distance']:.4f}\n\n")

        f.write("## Suggested manuscript interpretation\n\n")
        f.write(
            "The ablation confirms that the performance of CertiTwin-RSM is not due to a simple "
            "post-processing replacement rule. Removing calibration and using raw predicted safety "
            "leaves more unsafe actions after shielding. Replacing calibrated q90 upper bounds with "
            "a fixed mean-plus-sigma rule improves over raw prediction but remains less reliable than "
            "the calibrated certificate. Removing the distance-aware projection preserves safety but "
            "substantially increases action deviation, indicating a more intrusive shield. Therefore, "
            "the complete design provides the most balanced safety–utility–intent tradeoff.\n\n"
        )

        f.write("## Appendix detail table preview\n\n")
        f.write(detail_table.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("## Output files\n\n")
        for k, v in report_json["outputs"].items():
            f.write(f"- {k}: `{v}`\n")


def main():
    for p in [INPUT_OVERALL, INPUT_SUMMARY, INPUT_REPORT]:
        assert_exists(p)

    overall = pd.read_csv(INPUT_OVERALL)
    summary = pd.read_csv(INPUT_SUMMARY)
    step3_1_report = load_json(INPUT_REPORT)

    paper_table = make_paper_table(overall)
    detail_table = make_controller_detail_table(summary)
    values = extract_key_values(overall)
    findings_text = write_key_findings(values)

    paper_table.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")
    detail_table.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    report_json = {
        "input_files": {
            "step3_1_overall": str(INPUT_OVERALL),
            "step3_1_summary": str(INPUT_SUMMARY),
            "step3_1_report": str(INPUT_REPORT),
        },
        "step3_1_quality_checks": step3_1_report.get("overall_quality_checks", {}),
        "key_values": values,
        "outputs": {
            "paper_ablation_table": str(OUT_PAPER),
            "controller_detail_table": str(OUT_DETAIL),
            "key_findings": str(OUT_FINDINGS),
            "report_md": str(OUT_MD),
        },
        "recommended_placement": {
            "main_text": "table_step3_2_ablation_paper.csv",
            "appendix": "table_step3_2_ablation_controller_detail.csv",
            "main_paragraph": "step3_2_ablation_key_findings.txt",
        },
        "important_note": (
            "The ablation table should be interpreted as mechanism validation. "
            "Full-CertiTwin is not merely a safe-action replacement rule; calibration and distance-aware projection are necessary."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    make_report_md(paper_table, detail_table, values, report_json)

    print("Step 3.2 completed.")
    print(json.dumps(report_json, indent=2))
    print("\nPaper ablation table:")
    print(paper_table.to_string(index=False))
    print("\nKey findings:")
    print(findings_text)


if __name__ == "__main__":
    main()