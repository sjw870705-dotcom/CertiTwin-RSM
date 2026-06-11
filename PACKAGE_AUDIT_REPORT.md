# Package Audit Report

This audit was prepared for the public reproducibility repository of:

**CertiTwin-RSM: Controller-Agnostic Runtime Risk Shielding for SLA-Assured O-RAN Service Management**

## Audit scope

The audit checks repository completeness, file organization, current manuscript mapping, and basic figure-script executability.  It does not independently validate restricted raw commercial traces, because those raw traces are not redistributed in this public package.

## Findings

1. The repository includes main paper-level result tables, figure-data files, generated figures, plotting scripts, configuration templates, source-pipeline scripts, and representative candidate-level samples.
2. Supplement materials moved out of the 16-page manuscript are now integrated into `results/supplement_tables/` and `figures/supplement/`.
3. The top-level README has been rewritten to match the final manuscript title and the compressed TNSM resubmission structure.
4. `docs/result_table_mapping.md` now maps current manuscript Table I--VIII and Fig. 3--Fig. 6 to repository files.
5. The plotting scripts in `scripts/paper_figures/` were executed successfully against the included CSV data.

## Limitations

- Full raw commercial traces are not included due to redistribution restrictions.
- Full large pairwise/detail outputs are represented through aggregate tables and representative samples.
- Full end-to-end regeneration from raw traces requires access to the restricted source data.

## Recommendation

The package is suitable as a public experiment-reproducibility repository for reviewer inspection, provided that the public GitHub repository preserves this structure and does not include manuscript-source or submission-system files.
