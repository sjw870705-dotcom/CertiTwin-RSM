# Data Manifest

This repository provides experiment reproducibility materials rather than a full raw-data release.

## Included data

- `results/paper_tables/`: complete paper-level CSV tables used in the compressed manuscript.
- `results/figure_data/`: compact CSV files used to reproduce Fig. 3--Fig. 6 and the runtime-scaling supplement.
- `results/supplement_tables/`: appendix-level tables moved to the repository supplement because of the 16-page manuscript limit.
- `figures/paper/`: final paper figures and regenerated figure outputs.
- `figures/supplement/`: appendix/supplement figure moved from the manuscript.
- `data/processed_samples/`: representative samples from large pairwise/detail files.

## Excluded data

The full commercial raw traces are not redistributed unless redistribution is explicitly permitted by the data provider.  The repository therefore provides representative candidate-level samples, result tables, scripts, and manifests rather than a full commercial trace dump.

Several full pairwise/detail CSV files are also excluded from the repository because they can be large and are not necessary for verifying the paper-level tables and figures.  Representative samples and aggregate tables are provided where appropriate.

## Large files excluded from the original working directory

Examples include:

- full multiseed detail outputs;
- full certificate-inflation pairwise outputs;
- full twin-perturbation pairwise outputs;
- full SLA-threshold-shift pairwise outputs;
- full ablation pairwise outputs;
- very large runtime and rule-shield detail files.

The schema and representative contents are documented through `data/processed_samples/`, `results/supplement_tables/candidate_field_manifest.csv`, and `docs/dataset_description.md`.


## Public ColO-RAN logged sanity validation

The repository now includes a secondary public logged O-RAN sanity validation under `results/coloran_logged_sanity_v4/` and `scripts/coloran_logged_sanity/`. The validation uses ColO-RAN `rome_static_medium` logs with scheduling-policy candidates (`sched0`, `sched1`, `sched2`). It uses rank-normalized finite-candidate risk proxies and is reported in the manuscript as Table IX. This validation is deliberately bounded as a logged proxy sanity check, not a live near-RT RIC deployment or hard operator-SLA guarantee.

Key full-run size: 175,014 candidate rows and 58,338 finite candidate groups. The held-out test split contains 11,669 groups.
