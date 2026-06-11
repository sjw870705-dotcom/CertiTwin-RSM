# Result Table and Figure Mapping

This file maps the compressed TNSM resubmission manuscript to repository files.  The manuscript uses a compact Appendix; detailed appendix-level tables and the runtime-scaling figure are stored as repository supplement files.

## Main manuscript tables

| Manuscript item | What it supports | Repository file |
|---|---|---|
| Table I | Design requirements and CertiTwin-RSM components | `results/paper_tables/table_I_design_requirements_manifest.csv` |
| Table II | Experimental setup and management metrics | `results/paper_tables/table_II_experimental_setup_manifest.csv` |
| Table III | Digital-twin prediction and calibrated certificate diagnostics | `results/paper_tables/table_III_digital_twin_calibration_diagnostics.csv` |
| Table IV | Main paired raw-vs-CertiTwin comparison | `results/paper_tables/table_IV_main_raw_vs_certitwin.csv` |
| Table V | Rule-based and fixed-UCB runtime shield comparison | `results/paper_tables/table_V_rule_ucb_shield_comparison.csv` |
| Table VI | Ablation of calibrated certification and distance-aware projection | `results/paper_tables/table_VI_ablation_certification_projection.csv` |
| Table VII | Runtime overhead and candidate-size scalability | `results/paper_tables/table_VII_runtime_overhead_scalability.csv` |
| Table VIII | SLA-threshold robustness | `results/paper_tables/table_VIII_sla_threshold_robustness.csv` |

## Main manuscript figures

| Manuscript item | Repository figure | Source data |
|---|---|---|
| Fig. 3 | `figures/paper/fig3_unsafe_rate_reduction.pdf` / `.png` | `results/figure_data/fig3_unsafe_rate_reduction_data.csv` |
| Fig. 4 | `figures/paper/fig4_action_distance_ablation.pdf` / `.png` | `results/figure_data/fig4_action_distance_ablation_data.csv` |
| Fig. 5 | `figures/paper/fig5_certificate_inflation.pdf` / `.png` | `results/figure_data/fig5_certificate_inflation_data.csv` |
| Fig. 6 | `figures/paper/fig6_sla_threshold_robustness.pdf` / `.png` | `results/figure_data/fig6_sla_threshold_robustness_data.csv` |

## Supplement moved from the page-limited appendix

| Supplement item | Repository file |
|---|---|
| Candidate-field manifest | `results/supplement_tables/candidate_field_manifest.csv` |
| Split-usage manifest | `results/supplement_tables/split_usage_manifest.csv` |
| Per-controller paired shielding details | `results/supplement_tables/per_controller_paired_shielding_details.csv` |
| Projection-ablation/action-distance details | `results/supplement_tables/projection_ablation_action_distance_details.csv` |
| Evaluation-level stability details | `results/supplement_tables/evaluation_level_stability_details.csv` |
| Full certificate-inflation alpha grid | `results/supplement_tables/full_certificate_inflation_alpha_grid.csv` |
| Recommended certificate-inflation alpha | `results/supplement_tables/recommended_certificate_inflation_alpha.csv` |
| Full SLA-threshold shift details | `results/supplement_tables/full_sla_threshold_shift_details.csv` |
| Aggregate SLA-threshold robustness table | `results/supplement_tables/aggregate_sla_threshold_robustness_paper.csv` |
| Replayed K=5 runtime table | `results/supplement_tables/replayed_candidate_runtime_table.csv` |
| Synthetic K-scaling runtime table | `results/supplement_tables/synthetic_k_scaling_runtime_table.csv` |
| Synthetic K-scaling linear fit | `results/supplement_tables/synthetic_k_scaling_linear_fit.csv` |
| Runtime-scaling supplement figure | `figures/supplement/figC1_synthetic_candidate_runtime_scaling.png` |

## Backward-compatible files

Some earlier file names are retained in `results/paper_tables/` for backward compatibility with previous manuscript drafts.  The current manuscript mapping above should be treated as authoritative.
