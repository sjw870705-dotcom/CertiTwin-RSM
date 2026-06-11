# Repository Supplement for the Page-Limited TNSM Resubmission

The submitted manuscript is constrained to 16 pages.  To keep the PDF within the page limit while preserving transparency, detailed appendix-level materials are stored in this repository rather than in the manuscript PDF.

## Included supplement materials

- Candidate-level replay field manifest.
- Chronological split usage manifest.
- Per-controller paired raw-vs-shielded shielding details.
- Projection-ablation and action-distance details.
- Evaluation-level stability details for the real K=5 and stress K=3 settings.
- Full certificate-inflation alpha grid and recommended alpha settings.
- Full SLA-threshold shift details.
- Replayed candidate runtime table and synthetic K-scaling runtime table.
- Synthetic runtime-scaling figure moved from the original appendix.

## Why these files are outside the manuscript

The main PDF keeps the core evidence: digital-twin diagnostics, main paired raw-vs-shielded results, rule/FUCB baselines, ablation, runtime, perturbation/inflation, SLA-shift robustness, and an appendix summarizing replay/split/fairness boundaries.  The repository supplement contains supporting tables that are useful for audit and reproduction but are too detailed for the page-limited manuscript.

## How to use this supplement

1. Start with `docs/result_table_mapping.md` to locate the paper result.
2. Use `results/paper_tables/` for main manuscript tables.
3. Use `results/supplement_tables/` for appendix-level details.
4. Use `scripts/paper_figures/` to regenerate main figures and the runtime-scaling supplement plot.
