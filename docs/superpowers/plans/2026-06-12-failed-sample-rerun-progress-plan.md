# Failed Sample Rerun Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make failed-sample reruns observable, resumable after refresh, and useful for diagnosis through reason summaries and paginated details.

**Architecture:** Add a persisted `FailedSampleRerunJob` table and run failed-sample reruns in a background thread. Keep forecast artifacts as the source of truth for terminal sample counts; use task counters only while artifacts are not yet available. Frontend run detail loads run progress, failed-sample summary, and active rerun job, then polls the rerun job while active.

**Tech Stack:** FastAPI, SQLModel, background threads, Vue 3 Composition API, Vitest, pytest.

---

### Task 1: Backend Rerun Job Model And Progress API

**Files:**
- Modify: `backend/app/models/benchmark.py`
- Modify: `backend/app/db/init_db.py`
- Modify: `backend/app/services/run_executor.py`
- Modify: `backend/app/api/routes/benchmarking_runs.py`
- Test: `backend/tests/unit/test_run_adapter_failure.py`

- [x] Write tests that starting a failed-sample rerun returns a `rerun_job_id`, exposes `queued/running/succeeded` progress, blocks a second active rerun, and updates job counters.
- [x] Add `FailedSampleRerunJob` with `rerun_job_id`, `benchmarking_run_id`, `status`, `activity_status`, `total_samples`, `processed_samples`, `succeeded_samples`, `failed_samples`, `error_code`, `error_message`, `started_at`, `finished_at`, `created_at`, `updated_at`.
- [x] Register the table in `init_db`.
- [x] Split current synchronous rerun implementation into reusable worker logic that updates the job after each sample.
- [x] Add service functions `start_failed_sample_rerun`, `get_failed_sample_rerun_job`, and `get_active_failed_sample_rerun_job`.
- [x] Add routes `POST /benchmarking-runs/{id}/failed-samples/rerun`, `GET /benchmarking-runs/{id}/failed-samples/rerun`, and `GET /benchmarking-runs/{id}/failed-samples/rerun/{job_id}`.

### Task 2: Failed Sample Summary And Pagination

**Files:**
- Modify: `backend/app/services/run_executor.py`
- Modify: `backend/app/api/routes/benchmarking_runs.py`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/runs.ts`
- Test: `backend/tests/unit/test_run_adapter_failure.py`

- [x] Write tests that `list_failed_samples` returns grouped `summary` entries and honors `limit`, `offset`, `error_code`, and `error_message` filters.
- [x] Change `list_failed_samples` to return `items`, `total`, `limit`, `offset`, and `summary`.
- [x] Keep default detail limit bounded to 50.
- [x] Update frontend DTOs and API call parameters.

### Task 3: Correct Failed Count Source After Rerun

**Files:**
- Modify: `backend/app/services/run_executor.py`
- Test: `backend/tests/unit/test_run_progress_counts.py`

- [x] Write a regression test where a task counter still has an old failed count but forecast artifacts have no failed rows.
- [x] Change `_sample_counts` so artifact-backed terminal tasks use artifact counts, while running tasks without complete artifacts can still use task counters.

### Task 4: Run Detail UI

**Files:**
- Modify: `frontend/src/pages/RunDetailPage.vue`
- Modify: `frontend/src/api/runs.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Modify: `frontend/src/styles.css`
- Test: `frontend/src/tests/WorkspacePages.test.ts`

- [x] Write a Vitest test that an active rerun job from initial page load disables the rerun button and displays sample progress.
- [x] Write a Vitest test that the failed-sample card shows reason summaries first, and expanding one reason loads paginated sample details.
- [x] Replace the default full sample list with a reason summary table.
- [x] Add a rerun progress panel with processed/total, succeeded, still failed, pending, and activity status.
- [x] Poll active rerun job every 2 seconds; on terminal status refresh run progress and failed-sample summary.

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/manual/README.md`
- Modify: `docs/developer/data-model.md`
- Modify: `docs/developer/key-flows.md`

- [x] Document rerun job state, progress fields, failed-sample summary, and pagination.
- [x] Run backend focused tests.
- [x] Run frontend focused tests.
- [x] Run `cd backend && uv run pytest`.
- [x] Run `cd frontend && npm test`.
- [x] Run `cd frontend && npm run build`.
- [x] Commit all changes.
