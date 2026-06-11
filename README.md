# CertiTwin-RSM Reproducibility Repository

This repository contains the experiment-reproducibility materials for:

**CertiTwin-RSM: Controller-Agnostic Runtime Risk Shielding for SLA-Assured O-RAN Service Management**

The repository supports the page-limited TNSM resubmission.  It is designed to let reviewers inspect the reported paper-level results, regenerate the plotted figures, and access the appendix-level materials that were moved out of the manuscript to satisfy the 16-page limit.

> Scope note: this is an experiment-reproducibility repository.  It intentionally does **not** include the manuscript source, manuscript PDF, cover letter, response letter, or peer-review material.

## Repository contents

```text
configs/                    Configuration templates and SLA thresholds
data/processed_samples/      Representative candidate-level samples from large detail outputs
docs/                       Protocol, dataset, limitation, and result-mapping notes
figures/paper/              Final paper figures and regenerated plotting outputs
figures/supplement/         Appendix/supplement figures moved out of the page-limited manuscript
results/figure_data/        Compact CSV files used by plotting scripts
results/paper_tables/       CSV files corresponding to the main manuscript tables
results/supplement_tables/  Appendix-level tables moved to the repository supplement
scripts/paper_figures/      Scripts for regenerating Fig. 3--Fig. 6 and the runtime-scaling supplement
scripts/source_pipeline/    Source-pipeline scripts used to build replay, labels, twin, certificates, and summaries
scripts/tnsm_extra/         Extra TNSM checks: rule/FUCB baselines, runtime, multiseed, and synthetic K scaling
```

## Main manuscript result mapping

The page-limited manuscript keeps the core tables and figures in the PDF.  Their repository files are:

| Manuscript item | Repository file |
|---|---|
| Table I, design requirements | `results/paper_tables/table_I_design_requirements_manifest.csv` |
| Table II, experimental setup | `results/paper_tables/table_II_experimental_setup_manifest.csv` |
| Table III, digital-twin diagnostics | `results/paper_tables/table_III_digital_twin_calibration_diagnostics.csv` |
| Table IV, paired raw-vs-shielded comparison | `results/paper_tables/table_IV_main_raw_vs_certitwin.csv` |
| Table V, rule/FUCB shield comparison | `results/paper_tables/table_V_rule_ucb_shield_comparison.csv` |
| Table VI, ablation | `results/paper_tables/table_VI_ablation_certification_projection.csv` |
| Table VII, runtime overhead | `results/paper_tables/table_VII_runtime_overhead_scalability.csv` |
| Table VIII, SLA-threshold robustness | `results/paper_tables/table_VIII_sla_threshold_robustness.csv` |
| Fig. 3 | `figures/paper/fig3_unsafe_rate_reduction.pdf` and `.png` |
| Fig. 4 | `figures/paper/fig4_action_distance_ablation.pdf` and `.png` |
| Fig. 5 | `figures/paper/fig5_certificate_inflation.pdf` and `.png` |
| Fig. 6 | `figures/paper/fig6_sla_threshold_robustness.pdf` and `.png` |

A more detailed mapping, including figure-data CSV files and supplement materials, is provided in `docs/result_table_mapping.md`.

## Supplement moved from the manuscript appendix

To satisfy the 16-page limit, the resubmitted manuscript keeps only a compact Appendix and moves detailed tables to this repository.  The transferred materials are under:

- `results/supplement_tables/`
- `figures/supplement/`
- `docs/repository_supplement.md`

They include candidate-field manifests, split usage, per-controller shielding details, projection-ablation details, evaluation-level stability, full certificate-inflation alpha grids, recommended alpha settings, full SLA-threshold shift details, and synthetic runtime-scaling data/figure.

## Quick start: regenerate paper figures

Create the environment:

```bash
conda env create -f environment.yml
conda activate certitwin-rsm
```

Regenerate figures:

```bash
python scripts/paper_figures/plot_fig3_unsafe_reduction.py
python scripts/paper_figures/plot_fig4_action_distance_ablation.py
python scripts/paper_figures/plot_fig5_certificate_inflation.py
python scripts/paper_figures/plot_fig6_sla_threshold_robustness.py
python scripts/paper_figures/plot_figC1_runtime_scaling.py
```

The scripts read data from `results/figure_data/` and write outputs to `figures/paper/`.

## Data availability boundary

This repository provides:

- complete paper-level CSV result tables;
- compact figure-data CSV files;
- representative candidate-level samples;
- supplement tables moved from the page-limited appendix;
- source-pipeline scripts and configuration templates.

It does **not** redistribute raw commercial traces or third-party data whose redistribution is restricted.  The files under `data/processed_samples/` are representative samples intended to demonstrate schema and replay-level fields.  Full large pairwise/detail outputs can be regenerated from the processed candidate-level data when such data are available to the user.

See `data_manifest.md` and `docs/known_limitations.md` for details.

## Reproducibility checklist

See `reproducibility_checklist.md` for a compact reviewer-facing checklist and `PACKAGE_AUDIT_REPORT.md` for the package audit summary.

## License

See `LICENSE`.


## Public ColO-RAN logged sanity validation

The repository now includes a secondary public logged O-RAN sanity validation under `results/coloran_logged_sanity_v4/` and `scripts/coloran_logged_sanity/`. The validation uses ColO-RAN `rome_static_medium` logs with scheduling-policy candidates (`sched0`, `sched1`, `sched2`). It uses rank-normalized finite-candidate risk proxies and is reported in the manuscript as Table IX. This validation is deliberately bounded as a logged proxy sanity check, not a live near-RT RIC deployment or hard operator-SLA guarantee.

Key full-run size: 175,014 candidate rows and 58,338 finite candidate groups. The held-out test split contains 11,669 groups.
