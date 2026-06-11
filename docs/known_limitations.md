# Known Limitations of This Public Package

- The full manuscript source and PDF are intentionally not included.
- Full large pairwise/detail CSV files are excluded due to size; samples are provided.
- Full commercial raw traces are not included unless redistribution is permitted.
- Some source-pipeline scripts may require full processed candidate data and local hardware-specific paths to be adapted.
- The paper-level tables and plotting data are included to support verification of the reported results.


## Public ColO-RAN logged sanity validation

The repository now includes a secondary public logged O-RAN sanity validation under `results/coloran_logged_sanity_v4/` and `scripts/coloran_logged_sanity/`. The validation uses ColO-RAN `rome_static_medium` logs with scheduling-policy candidates (`sched0`, `sched1`, `sched2`). It uses rank-normalized finite-candidate risk proxies and is reported in the manuscript as Table IX. This validation is deliberately bounded as a logged proxy sanity check, not a live near-RT RIC deployment or hard operator-SLA guarantee.

Key full-run size: 175,014 candidate rows and 58,338 finite candidate groups. The held-out test split contains 11,669 groups.
