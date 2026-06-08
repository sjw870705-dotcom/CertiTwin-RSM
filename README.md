# CertiTwin-RSM Reproducibility Repository

This repository contains reproducibility materials for the experimental evaluation of **CertiTwin-RSM: Calibrated Risk-Shielded Spectrum Management for O-RAN Network Slicing**.

The repository is intentionally limited to experiment reproducibility. It **does not include the full manuscript source, manuscript PDF, cover letter, response letter, or any peer-review material**.

## What is included

- final paper-level result tables;
- figure-generation data and scripts;
- final experimental figures Fig. 3--Fig. 6 and Appendix Fig. C1;
- configuration templates;
- representative samples from large pairwise/detail outputs;
- documentation describing the evaluation protocol and file mapping.

## What is not included

- full manuscript LaTeX or PDF;
- commercial raw traces or any third-party data that cannot be redistributed;
- full pairwise/detail CSV files larger than normal GitHub limits.

## Quick start

```bash
conda env create -f environment.yml
conda activate certitwin-rsm
python scripts/paper_figures/plot_fig3_unsafe_reduction.py
python scripts/paper_figures/plot_fig4_action_distance_ablation.py
python scripts/paper_figures/plot_fig5_certificate_inflation.py
python scripts/paper_figures/plot_fig6_sla_threshold_robustness.py
python scripts/paper_figures/plot_figC1_runtime_scaling.py
```

The generated figures are written to `figures/paper/`. The source data used by the plotting scripts are under `results/figure_data/` and `results/paper_tables/`.

## Paper-result mapping

See `docs/result_table_mapping.md` for the mapping between paper tables/figures and repository files.

## Data note

The repository includes representative candidate-level samples and complete paper-level result tables. Full large pairwise/detail outputs are not included because they exceed ordinary repository limits and may be regenerated from the processed candidate data when available. See `data_manifest.md`.
