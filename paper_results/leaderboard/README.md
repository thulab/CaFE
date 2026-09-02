# CaFE interactive leaderboard

This directory is a dependency-free static view of the frozen public-model
results. From the repository root, preview it with:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/paper_results/leaderboard/`. The page also
works when the repository is served by GitHub Pages or another static host.

Regenerate `leaderboard-data.js` after changing the frozen result tables:

```bash
uv run --extra plots python paper_results/leaderboard/build_leaderboard_data.py
```

The overall table averages capability-level cells within each suite and then
weights available suites equally. The capability table first weights the five
levels equally within each `(suite, model, capability)` cell and then weights
available suites equally. Missing suites or cells are never imputed.
