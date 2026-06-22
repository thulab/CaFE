# Remove Value Columns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MVP single-target-only by removing the `value_columns` concept from user-facing APIs, backend storage metadata, frontend configuration, and docs.

**Architecture:** The load job remains the place where the target is selected via `split_config.target_columns`, constrained to exactly one element. Readers only read that selected target column or TsFile series, `SeriesPoint.values_json` stores only that target, and sample views stay unchanged.

**Tech Stack:** FastAPI, SQLModel, pytest, Vue 3, TypeScript, Vitest, SQLite runtime reset.

---

### Task 1: Backend Single-Target Data Path

**Files:**
- Modify: `backend/app/models/dataset.py`
- Modify: `backend/app/api/routes/dataset_manifests.py`
- Modify: `backend/app/services/dataset_reader.py`
- Modify: `backend/app/services/csv_dataset_reader.py`
- Modify: `backend/app/services/tsfile_dataset_reader.py`
- Modify: `backend/app/services/dataset_load_service.py`
- Modify: `backend/app/services/series_store.py`
- Modify tests under `backend/tests/`

- [x] Write failing tests that create manifests without `value_columns`, load with one `target_columns` entry, assert shards expose no `value_columns`, and assert multi-device TsFile full-series target succeeds.
- [x] Run focused backend tests and confirm failures mention stale `value_columns` assumptions.
- [x] Remove `value_columns` from `DatasetManifest` and `Shard`.
- [x] Change `DatasetReadResult.value_columns` to `target_columns` and keep `column_matrix` behavior against selected targets.
- [x] Change reader protocol argument from `value_columns` to `target_columns`.
- [x] Make CSV/TsFile readers require one target, validate it, and read only that target.
- [x] Change load service to parse and validate `split_config.target_columns` before reading, pass them to the reader, and write only target columns into `SeriesPoint`.
- [x] Update backend tests and helper factories.
- [x] Run focused backend tests, then commit backend changes.

### Task 2: Frontend Target-Only Configuration

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/components/wizard/ColumnAndSplitStep.vue`
- Modify: `frontend/src/pages/DatasetsPage.vue`
- Modify: `frontend/src/pages/DatasetManifestPage.vue`
- Modify frontend tests under `frontend/src/tests/`

- [x] Write failing frontend tests that assert manifest payloads omit `value_columns`, TsFile target-only creation submits only `split_config.target_columns`, and failed load jobs show an error.
- [x] Remove Value columns UI and state.
- [x] Submit manifest without `value_columns`.
- [x] Check `job.status` after `createLoadJob`; if not `succeeded`, display `error_code · error_message` and do not advance.
- [x] Update TypeScript DTOs and tests.
- [x] Run frontend tests and commit frontend changes.

### Task 3: Docs, Scripts, Runtime Reset, Final Verification

**Files:**
- Modify: `docs/manual/README.md`
- Modify: `docs/developer/key-flows.md`
- Modify: `docs/developer/data-model.md`
- Modify: `scripts/baseline_run.py`
- Delete local runtime data outside git: `backend/runtime/tsbenchmark.db`, `backend/runtime/uploads/*`

- [x] Update docs to describe single target selection only.
- [x] Update baseline script output so it no longer references `value_columns`.
- [x] Stop the local system if running, remove runtime test DB/uploads, and restart if needed.
- [x] Run `cd backend && uv run pytest`.
- [x] Run `cd frontend && npm test`.
- [x] Run `git diff --check`.
- [x] Commit docs and cleanup-related code changes.
