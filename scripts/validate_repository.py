from pathlib import Path
import csv
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    'README.md', 'data_manifest.md', 'docs/result_table_mapping.md',
    'docs/repository_supplement.md', 'reproducibility_checklist.md',
    'requirements.txt', 'environment.yml',
    'results/paper_tables/table_III_digital_twin_calibration_diagnostics.csv',
    'results/paper_tables/table_IV_main_raw_vs_certitwin.csv',
    'results/paper_tables/table_V_rule_ucb_shield_comparison.csv',
    'results/paper_tables/table_VI_ablation_certification_projection.csv',
    'results/paper_tables/table_VII_runtime_overhead_scalability.csv',
    'results/paper_tables/table_VIII_sla_threshold_robustness.csv',
    'results/supplement_tables/candidate_field_manifest.csv',
    'results/supplement_tables/evaluation_level_stability_details.csv',
    'figures/supplement/figC1_synthetic_candidate_runtime_scaling.png',
    'scripts/paper_figures/plot_fig3_unsafe_reduction.py',
    'scripts/paper_figures/plot_fig4_action_distance_ablation.py',
    'scripts/paper_figures/plot_fig5_certificate_inflation.py',
    'scripts/paper_figures/plot_fig6_sla_threshold_robustness.py',
    'scripts/paper_figures/plot_figC1_runtime_scaling.py',
]


def csv_has_rows(path: Path) -> bool:
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f))
        return len(rows) >= 2
    except Exception:
        return False


def main():
    missing = []
    empty_csv = []
    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif p.suffix.lower() == '.csv' and not csv_has_rows(p):
            empty_csv.append(rel)

    print(f'Repository root: {ROOT}')
    print(f'Required files checked: {len(REQUIRED_FILES)}')
    print(f'Missing files: {len(missing)}')
    for rel in missing:
        print(f'  MISSING: {rel}')
    print(f'CSV files without data rows: {len(empty_csv)}')
    for rel in empty_csv:
        print(f'  EMPTY_OR_INVALID_CSV: {rel}')

    if missing or empty_csv:
        sys.exit(1)
    print('Basic repository validation passed.')

if __name__ == '__main__':
    main()
