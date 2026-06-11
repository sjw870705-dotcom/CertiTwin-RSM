# Repository Integrity Check

Generated for the final public reproducibility package.

## Summary

- Total files: 133
- CSV files: 47
- Python scripts: 50
- Figure files under `figures/`: 11
- Main paper table files with current manuscript numbering: 8 plus backward-compatible earlier names
- Supplement tables moved from manuscript appendix: 12

## Checks performed

- Required README, data manifest, result mapping, supplement guide, configuration files, paper tables, supplement tables, and plotting scripts are present.
- CSV files required by `scripts/validate_repository.py` contain header and data rows.
- Figure-generation scripts in `scripts/paper_figures/` were executed successfully:
  - `plot_fig3_unsafe_reduction.py`
  - `plot_fig4_action_distance_ablation.py`
  - `plot_fig5_certificate_inflation.py`
  - `plot_fig6_sla_threshold_robustness.py`
  - `plot_figC1_runtime_scaling.py`

## Truthfulness boundary

This package verifies repository-level completeness and paper-result reproducibility from included tables and figure data.  It does not independently audit restricted raw commercial traces, because raw traces are not redistributed in this public package.

## Inventory

A path/size/hash inventory is provided in `repository_file_inventory.csv`.
