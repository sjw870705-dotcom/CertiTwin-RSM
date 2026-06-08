# Package Audit Report

This package was checked for public-release suitability.

## Checks performed

- No zero-byte files were found.
- No manuscript source files or manuscript PDFs are included.
- No cover letters, response letters, or peer-review materials are included.
- Python bytecode caches (`__pycache__`, `*.pyc`) were removed.
- Old/duplicate figure versions were removed; only final experimental figures are retained.
- Full large pairwise/detail CSV files above ordinary repository limits are not included; representative samples are provided under `data/processed_samples/`.
- Plotting scripts under `scripts/paper_figures/` were tested using the included `results/figure_data/` files and regenerated the final figures.
- The path template was converted to repository-relative example paths.

## Intended repository scope

This is a lightweight experiment-reproducibility package. It is intended to verify paper-level result tables and regenerate experimental figures. It is not a full manuscript release and does not include commercial raw traces.
