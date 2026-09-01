# Main-experiment analysis workspace

This directory contains read-only derived summaries for the four frozen main
experiments. Remote production artifacts are never modified.

- `notes.md`: paper-facing findings, caveats, draft prose, and figure guidance.
- `summary.json`: compact machine-readable evidence.
- `tables/`: suite-, task-, capability-, level-, coverage-, rank-, and bootstrap
  CSV files.
- `figures/`: publication-oriented PNG/PDF heatmaps, curves, and rank plots.
- `validation_report.json`: aggregation and completeness checks.
- `analyze_main_experiments.py`: deterministic derivation script.
- `source_snapshot/`: copied JSON summaries/manifests from `timecho92`; no remote
  artifact was modified.

Reproduce with:

```bash
uv run --with matplotlib python \
  paper_results/work/main_experiments/analyze_main_experiments.py
```
