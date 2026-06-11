# How to Reproduce the Paper-Level Results

This repository supports two levels of reproduction:

1. **Paper-level figure/table reproduction** using included CSV files.
2. **Pipeline-level inspection** using the source scripts and configuration templates.  Full regeneration from raw commercial traces requires access to restricted raw/processed data and is therefore outside the public repository release.

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate certitwin-rsm
```

Alternatively, install Python dependencies with:

```bash
pip install -r requirements.txt
```

## 2. Regenerate paper figures

```bash
python scripts/paper_figures/plot_fig3_unsafe_reduction.py
python scripts/paper_figures/plot_fig4_action_distance_ablation.py
python scripts/paper_figures/plot_fig5_certificate_inflation.py
python scripts/paper_figures/plot_fig6_sla_threshold_robustness.py
python scripts/paper_figures/plot_figC1_runtime_scaling.py
```

Outputs are written to `figures/paper/`.  The first four scripts reproduce manuscript Fig. 3--Fig. 6.  The runtime-scaling script reproduces the supplement figure that was moved out of the page-limited manuscript.

## 3. Inspect paper tables

Main manuscript table files are under `results/paper_tables/`.  Current manuscript numbers are mapped in `docs/result_table_mapping.md`.

Supplement tables moved from the appendix are under `results/supplement_tables/`.

## 4. Inspect representative candidate samples

Representative large-output samples are under `data/processed_samples/`.  These samples demonstrate file schema and candidate-level records but are not intended to be a full raw commercial trace release.

## 5. Source pipeline scripts

The scripts under `scripts/source_pipeline/` document the step-wise experimental pipeline used to build the replay dataset, labels, digital twin, calibrated certificate, raw/shielded controller summaries, ablations, robustness checks, and figures.  Running the full pipeline requires the corresponding raw or processed candidate-level data, which may not be redistributed in this public repository.

## 6. Reviewer-oriented reading order

1. `README.md`
2. `docs/result_table_mapping.md`
3. `data_manifest.md`
4. `docs/repository_supplement.md`
5. `reproducibility_checklist.md`
