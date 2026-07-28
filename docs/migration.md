# TSBenchmark → CaFE migration

CaFE retains the complete TSBenchmark ancestor history. The extraction base is
commit `21b8452`, tagged `monorepo-cutover-2026-07-28`.

Core path mapping:

| TSBenchmark path | CaFE path |
|---|---|
| `scripts/paper_v8_pipeline_common.py` | `src/cafe/protocol.py` |
| `backend/app/services/synthetic_v8_generation.py` | `src/cafe/generation/families.py` |
| `backend/app/services/synthetic_v8_feature_gate.py` | `src/cafe/validation/mechanisms.py` |
| `scripts/calibrate_paper_v8.py` | `src/cafe/calibration/runner.py` |
| `scripts/generate_paper_v8_samples.py` | `src/cafe/generation/runner.py` |
| `scripts/validate_paper_v8_samples.py` | `src/cafe/validation/runner.py` |
| `scripts/run_paper_v8_inference.py` | `src/cafe/inference/runner.py` |
| `scripts/analyze_paper_v8.py` | `src/cafe/analysis/runner.py` |

The platform frontend/backend, database layer, old experiment pipelines, and
standalone pilot launchers were removed from the CaFE working tree. They remain
available before the cutover tag.

To inspect or rerun the exact pre-extraction tree without switching the active
checkout:

```bash
git worktree add ../cafe-monorepo-cutover monorepo-cutover-2026-07-28
```
