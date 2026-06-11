# ColO-RAN Logged Sanity-Validation Repository Update

Added files:

- `results/coloran_logged_sanity_v4/table_coloran_logged_sanity_validation_v4.csv`
- `results/coloran_logged_sanity_v4/coloran_logged_sanity_diagnostics_v4.csv`
- `results/coloran_logged_sanity_v4/coloran_logged_sanity_metadata_v4.txt`
- `results/coloran_logged_sanity_v4/coloran_candidate_group_dataset_v4.csv`
- `results/paper_tables/table_IX_coloran_logged_sanity_validation.csv`
- `scripts/coloran_logged_sanity/run_coloran_logged_sanity_validation_v4.py`

Full-run summary:

- candidate rows: 175,014
- finite candidate groups: 58,338
- held-out test groups: 11,669
- candidate action: scheduling-policy candidates (`sched0`, `sched1`, `sched2`)
- risk proxy: rank-normalized finite-candidate eMBB and URLLC-like risk

Boundary:
This is a secondary public logged O-RAN sanity validation, not a live near-RT RIC deployment.
