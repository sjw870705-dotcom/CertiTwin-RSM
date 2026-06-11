# Public ColO-RAN Logged Sanity Validation

This folder contains the reproduction script for the public ColO-RAN logged sanity validation reported as Table IX in the manuscript.

## Boundary

This is a secondary public logged O-RAN sanity validation. It is not a live near-RT RIC deployment and does not claim hard operator-SLA guarantees. Candidate actions are scheduling-policy candidates (`sched0`, `sched1`, `sched2`) from the ColO-RAN `rome_static_medium` logs. Risks are rank-normalized within each finite candidate group.

## Run

After cloning the public ColO-RAN dataset into a local folder, run:

```bash
python scripts/coloran_logged_sanity/run_coloran_logged_sanity_validation_v4.py --dataset-root /path/to/colosseum-oran-coloran-dataset --out-dir results/coloran_logged_sanity_v4 --max-tr 28 --max-exp 5 --max-windows 60
```

The repository includes the processed outputs used for the paper under `results/coloran_logged_sanity_v4/`.
