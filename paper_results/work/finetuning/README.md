# Fine-tuning analysis artifacts

Run the full local reconstruction and figure generation from the CaFE root:

```bash
uv run --with matplotlib python paper_results/work/finetuning/analyze_finetuning.py
```

The script consumes the read-only server snapshot under `raw/`, verifies every
distributed metric part, reconstructs all macro metrics, and writes the summary
CSV/JSON files and PNG/PDF figures in this directory. It uses a fixed bootstrap
seed (`20260902`).

Start with:

- `paper_section_zh.md`: paper-ready Chinese analysis and captions;
- `paper_section_en.md`: concise English draft;
- `notes.md`: protocol audit, claim-strength assessment, caveats, and proposed
  follow-up controls;
- `summary.json`: machine-readable key results;
- `checkpoint_summary_wide.csv` and `seed_transfer_summary.csv`: the requested
  baseline/best/final and cross-seed statistics;
- `finetuning_relative_change.pdf` and `finetuning_pareto_trajectory.pdf`: main
  paper figure candidates;
- `finetuning_capability_heatmap.pdf`: capability-level appendix candidate.

All metrics are lower-is-better. “Best” checkpoints are descriptive minima on
the named evaluation corpus; use the fixed 40k endpoint for a pre-specified
comparison unless a separate validation seed is added.
