# Modification Report

This version was prepared after checking the uploaded repository package and the page-limited TNSM resubmission requirements.

## Main problems found in the uploaded package

1. The top-level `README.md` still used an earlier manuscript title and did not reflect the final TNSM title.
2. The newly added appendix/supplement materials were stored under `repository_supplement_to_upload/`, which is suitable for a temporary transfer folder but not ideal as the final public repository structure.
3. `docs/result_table_mapping.md` still referred to older Appendix B/C table names and older manuscript table numbering.
4. The public-repository boundary was not explicit enough: the repository contains paper-level tables, figure data, representative samples, and supplement tables, but not raw commercial traces or manuscript source.
5. There was no single validation script for checking required public-repository files.

## Changes made

1. Rewrote the top-level `README.md` for the final title: `CertiTwin-RSM: Controller-Agnostic Runtime Risk Shielding for SLA-Assured O-RAN Service Management`.
2. Integrated the supplement files into final repository folders:
   - `results/supplement_tables/`
   - `figures/supplement/`
3. Removed the temporary `repository_supplement_to_upload/` folder from the final package.
4. Rewrote `docs/result_table_mapping.md` to map the current compressed manuscript tables and figures to repository files.
5. Added `docs/repository_supplement.md` to explain why appendix-level details were moved to the repository.
6. Rewrote `data_manifest.md` to clarify included data, excluded raw traces, supplement tables, and representative samples.
7. Rewrote `docs/how_to_reproduce.md` to provide a clearer reviewer-facing reproduction path.
8. Updated `reproducibility_checklist.md` and `PACKAGE_AUDIT_REPORT.md`.
9. Added `scripts/validate_repository.py` for basic repository integrity checks.
10. Added current-manuscript table aliases under `results/paper_tables/`, including files for Table I--VIII.
11. Generated `repository_file_inventory.csv` with file paths, sizes, and short SHA-256 hashes.
12. Added `REPOSITORY_INTEGRITY_CHECK.md` summarizing the performed checks and truthfulness boundary.
13. Ran all paper figure scripts successfully inside the final package.

## Important boundary

This repository supports paper-level reproducibility and reviewer inspection.  It does not claim to provide unrestricted raw commercial traces, because full raw trace redistribution may be restricted.
