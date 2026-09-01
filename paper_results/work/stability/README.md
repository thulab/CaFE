# CaFE stability analysis work package

This directory is a read-only analysis snapshot for:

`/data/xmy/CaFE/runtime/orchestration/short_stability10_inference_3node_78ef32f_20260831/stability`

No remote artifact or production code was modified.

## Main deliverables

- `paper_text_zh.md`: paper-oriented findings, tables, captions, claim boundaries, and limitations.
- `paper_tables.md`: generated compact Markdown tables.
- `analysis_summary.json`: machine-readable key results.
- `remote_audit.json`: read-only remote completeness and representative schema audit.
- `tables/`: full CSV summaries, including rank agreement, cell winners, model-gap/seed-noise comparisons, task coverage, and schemas.
- `figures/`: publication-ready PNG and vector PDF figures.
- `input_manifest.json`: SHA-256 manifest of all locally copied source snapshots.
- `verification.json`: cross-check results against the copied remote summaries and remote completeness audit.

## Reproduction

From the repository root:

```bash
bash paper_results/work/stability/fetch_stability_inputs.sh
uv run python paper_results/work/stability/collect_remote_audit.py
uv run --extra plots python paper_results/work/stability/analyze_stability.py
uv run python paper_results/work/stability/verify_stability_outputs.py
```

`fetch_stability_inputs.sh` only uses `scp` to copy the remote stability summary and ten suite-level JSON files into `raw/`. `collect_remote_audit.py` runs a read-only remote Python process that opens JSON and Parquet metadata and returns the audit through stdout.

## Statistical definitions

- Macro effect NRMSE: equal mean of 8 capabilities × 5 levels after each cell has been aggregated task-equally by the official analysis suite.
- Seed SD: sample SD across the ten augmentation seeds.
- Seed 95% empirical interval: linear 2.5% and 97.5% quantiles of ten values; descriptive only.
- Seed-mean 95% CI: Student-t interval with 9 degrees of freedom.
- Seed SD / task SE: across-seed SD divided by the mean task-bootstrap SE approximated from each suite row's 95% bootstrap interval.
- Gap noise ratio: SD of the paired per-seed model difference divided by the absolute mean model gap.
- Winner consistency: frequency of the modal lowest-NRMSE model in a capability × level cell across ten seeds.

## Recommended entry points

- Main stability story: `figures/fig_stability_overall.pdf`
- Fine-grained winner caveat: `figures/fig_capability_level_winner_consistency.pdf`
- Capability and uncertainty diagnosis: `figures/fig_direction_and_uncertainty.pdf`
- Detailed model table: `tables/model_overall_stability_extended.csv`
- Exact model-gap audit: `tables/model_pairwise_gap_vs_seed_noise.csv`
- Coverage caveat: `tables/capability_task_coverage.csv`
