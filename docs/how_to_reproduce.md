# How to Reproduce

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate certitwin-rsm
```

## 2. Inspect paper-level tables

Final tables are under:

```text
results/paper_tables/
```

## 3. Regenerate paper figures

```bash
python scripts/paper_figures/plot_fig3_unsafe_reduction.py
python scripts/paper_figures/plot_fig4_action_distance_ablation.py
python scripts/paper_figures/plot_fig5_certificate_inflation.py
python scripts/paper_figures/plot_fig6_sla_threshold_robustness.py
python scripts/paper_figures/plot_figC1_runtime_scaling.py
```

The generated figures are saved to `figures/paper/`.

## 4. Full pipeline

The original step-wise pipeline scripts are archived under `scripts/source_pipeline/`. They are provided for transparency and may require full processed candidate-level data that are not included in this lightweight reproducibility package.
