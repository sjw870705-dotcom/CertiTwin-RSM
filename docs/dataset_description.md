# Dataset Description

The experiments use finite candidate groups for O-RAN slicing decisions. Each candidate has predicted SLA-risk outcomes, uncertainty estimates, certified-safe flags, true replay labels, and management utility.

The repository includes representative samples only. The paper-level result tables are complete and are sufficient to reproduce the reported tables and plotted figures.

Key concepts:

- `candidate_group_id`: decision-state group identifier;
- `candidate_id`: executable candidate within a group;
- `true_V_*`: true replayed SLA-risk outcome;
- `pred_mean_*`: digital-twin mean prediction;
- `pred_std_*`: digital-twin uncertainty estimate;
- `cert_safe_*`: calibrated certificate safety flag;
- `management_utility`: utility used for raw-vs-shielded evaluation.


## ColO-RAN logged validation subset

The ColO-RAN logged validation uses the public `rome_static_medium` dataset. For each finite candidate group, the same training configuration, experiment instance, base station, and time window are paired across scheduling-policy directories (`sched0`, `sched1`, `sched2`). Slice 0, slice 1, and slice 2 are treated as eMBB, MTC, and URLLC-like services, respectively. The validation constructs rank-normalized risk proxies within each candidate group and reports only proxy-level logged sanity results.
