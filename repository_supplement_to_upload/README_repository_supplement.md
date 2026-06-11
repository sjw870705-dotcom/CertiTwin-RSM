# Repository supplement for the 16-page TNSM resubmission

This folder contains the appendix-level materials that were moved out of the manuscript to satisfy the 16-page limit while preserving reproducibility.

## Contents

- `results/appendix_tables/candidate_field_manifest.csv`: representative candidate-level replay fields.
- `results/appendix_tables/split_usage_manifest.csv`: train/calibration/validation/test usage.
- `results/appendix_tables/per_controller_paired_shielding_details.csv`: per-controller paired raw-vs-shielded details.
- `results/appendix_tables/projection_ablation_action_distance_details.csv`: projection-ablation and action-distance details.
- `results/appendix_tables/evaluation_level_stability_details.csv`: evaluation-level stability details.
- `results/appendix_tables/full_certificate_inflation_alpha_grid.csv`: full alpha-grid under twin-prediction perturbation.
- `results/appendix_tables/recommended_certificate_inflation_alpha.csv`: recommended alpha settings.
- `results/appendix_tables/full_sla_threshold_shift_details.csv`: full SLA-threshold shift details.
- `results/appendix_tables/synthetic_k_scaling_runtime_table.csv`: synthetic candidate-size runtime table.
- `results/appendix_tables/synthetic_k_scaling_linear_fit.csv`: linear fit for synthetic runtime scaling.
- `figures/appendix/figC1_synthetic_candidate_runtime_scaling.*`: runtime-only scaling plot moved from the manuscript appendix.

These files are intended to be uploaded to the public reproducibility repository at:
https://github.com/sjw870705-dotcom/CertiTwin-RSM.git

The manuscript source is not included in the public repository. The repository is intended for experiment reproducibility: result tables, figure data, plotting scripts, configuration files, representative candidate samples, and data manifests.
