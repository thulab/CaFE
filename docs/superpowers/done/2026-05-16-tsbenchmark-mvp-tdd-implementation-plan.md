# TSBenchmark MVP TDD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the merged TSBenchmark MVP that loads a real CSV dataset into `Shard(real)`, materializes samples, runs deterministic stub model evaluation, computes MSE/MAE, generates report JSON, refreshes ranking snapshots, and exposes a Vue evaluation wizard for the full flow.

**Architecture:** The backend is a FastAPI application with SQLModel entities, SQLite metadata, and runtime file artifacts. Domain services own data loading, JSONL stores, metrics, model adapters, run execution, ranking, and report generation. The frontend is a Vue + Vite single evaluation wizard plus result views, consuming stable API DTOs.

**Tech Stack:** FastAPI, SQLModel, SQLite, pytest, uv, Vue, Vite, npm, Vitest, Vue Test Utils, optional MSW for frontend API mocks.

---

## Execution Constraints

- This document is a detailed design and task split only. It does not execute development.
- All implementation must follow TDD: write failing test, verify RED, write minimal code, verify GREEN, refactor with tests still passing.
- Git operations are owned by the user. This plan intentionally assigns no version-control actions to implementation agents.
- Every backend test must use a temporary `TSBENCHMARK_RUNTIME_DIR` and temporary SQLite database, never the real `runtime/` directory.
- The MVP is CSV-only, single target column, local trusted environment, no login, no production model service.

## Source Documents

- `docs/superpowers/specs/2026-05-15-tsbenchmark-platform-functional-definition-design.md`
- `docs/superpowers/specs/2026-05-16-tsbenchmark-mvp-entity-structure-design.md`
- `README.md`

## Detailed Design

### Backend Package Layout

```text
backend/
  pyproject.toml
  app/
    main.py
    core/
      config.py
      errors.py
      ids.py
      time.py
      storage_paths.py
    db/
      session.py
      init_db.py
    models/
      dataset.py
      sample.py
      benchmark.py
      metric.py
      report.py
      ranking.py
      model_registry.py
    schemas/
      common.py
      dataset.py
      sample.py
      benchmark.py
      metric.py
      report.py
      ranking.py
      model_registry.py
    api/
      deps.py
      routes/
        dataset_manifests.py
        dataset_load_jobs.py
        shards.py
        capability_blocks.py
        tracks.py
        models.py
        benchmarking_runs.py
        reports.py
        ranking_lists.py
        samples.py
        wizard.py
    services/
      dataset_reader.py
      csv_dataset_reader.py
      dataset_load_service.py
      sample_store.py
      forecast_store.py
      metric_service.py
      model_adapter.py
      stub_timer_adapter.py
      run_executor.py
      ranking_service.py
      report_service.py
    workers/
      run_queue.py
      lifecycle.py
  tests/
    unit/
    api/
    e2e/
    fixtures/
```

### Backend Boundaries

- `models/` contains SQLModel persistence entities only. It must not contain workflow logic.
- `schemas/` contains API request/response DTOs and read models. DTOs are separate from SQLModel entities.
- `DatasetReader` is the data access boundary. `CsvDatasetReader` implements the MVP CSV rules and is the only concrete reader in this phase.
- `dataset_load_service.py` owns `DatasetManifest -> DatasetLoadJob -> Shard(real) -> SampleIndex -> Sample JSONL`.
- `sample_store.py` and `forecast_store.py` own JSONL schema, canonical JSON, checksum, and lookup by `sample_id`.
- `metric_service.py` owns pure MSE/MAE calculation and aggregation rules.
- `model_adapter.py` defines `ModelAdapter.forecast(sample, model, timeout)`. `StubTimerAdapter` is the MVP implementation.
- `run_executor.py` owns run expansion, task execution, cancellation, status transitions, forecast writing, and metric persistence.
- `ranking_service.py` owns persisted `RankingEntry` snapshot refresh for `latest_valid_result` and `best_result`.
- `report_service.py` owns `runtime/reports/{run_id}.json` generation.
- Route modules only validate requests, inject dependencies, call services, and return DTOs.

### Runtime Artifact Layout

```text
runtime/
  uploads/
  samples/
  forecasts/
  reports/
  tsbenchmark.db
```

`TSBENCHMARK_RUNTIME_DIR` defaults to `runtime`. `TSBENCHMARK_DATABASE_URL` defaults to `sqlite:///runtime/tsbenchmark.db`.

### API Contracts

```text
POST /dataset-manifests/upload
POST /dataset-manifests
GET  /dataset-manifests/{dataset_manifest_id}
POST /dataset-load-jobs
GET  /dataset-load-jobs/{load_job_id}
GET  /shards/{shard_id}
GET  /shards/{shard_id}/samples?limit=20&offset=0
POST /capability-blocks
POST /tracks
POST /wizard/real-dataset-track
GET  /models
POST /models
POST /benchmarking-runs
GET  /benchmarking-runs/{benchmarking_run_id}/progress
POST /benchmarking-runs/{benchmarking_run_id}/cancel
GET  /reports/{report_id}
GET  /tracks/{track_id}/ranking?metric=mse&policy=latest_valid_result
GET  /samples/{sample_id}/preview
GET  /samples/{sample_id}/forecast?run_id={benchmarking_run_id}
```

Unified error response:

```json
{
  "error_code": "csv_time_not_monotonic",
  "message": "time_column must be strictly increasing",
  "details": {
    "row_index": 12
  }
}
```

### Key DTOs

`UploadPreviewDTO`:

```text
upload_id
source_uri
filename
file_size
detected_delimiter
encoding
columns: [{ name, inferred_type, nullable, sample_values }]
preview_rows: [{ column_name: value }]
row_count_estimate
validation_summary: { has_header, duplicate_columns, parse_warnings }
created_at
```

`DatasetManifestCreateDTO`:

```text
name
domain
source_uri
file_format = csv
time_column
target_columns
frequency?
timezone?
```

`DatasetLoadJobCreateDTO`:

```text
dataset_manifest_id
split_config: { context_length, horizon, stride? }
seed?
```

`RunProgressDTO`:

```text
benchmarking_run_id
status
progress: { total_models, completed_models, total_tasks, completed_tasks, total_samples, completed_samples, failed_samples }
units: [{ unit_id, model_id, model_name, status, task_count, completed_task_count, metrics?, error_code?, error_message? }]
tasks: [{ task_id, unit_id, model_id, capability_block_id, capability_block_name, status, shard_count, sample_count, completed_sample_count, metrics?, error_code?, error_message? }]
recent_events: [{ level, event_type, message, created_at }]
report_id?
ranking_list_id?
```

`SampleForecastDTO`:

```text
sample_id
benchmarking_run_id
shard_id
capability_block_id
history_timestamps
future_timestamps
target_column_names
target_history
target_future
models: [{ model_id, model_name, unit_id, task_id, status, forecast?, metrics, forecast_artifact_id?, error_code?, error_message? }]
links: { run, report, ranking, task? }
```

### Frontend Layout

```text
frontend/
  package.json
  src/
    main.ts
    App.vue
    api/
      client.ts
      types.ts
      datasets.ts
      tracks.ts
      models.ts
      runs.ts
      results.ts
    stores/
      wizard.ts
    pages/
      EvaluationWizardPage.vue
      RankingPage.vue
      ReportPage.vue
      SampleForecastPage.vue
    components/
      wizard/
        UploadStep.vue
        ColumnAndSplitStep.vue
        LoadShardStep.vue
        TrackStep.vue
        ModelSelectionStep.vue
        RunStep.vue
        ResultStep.vue
      results/
        RankingTable.vue
        ReportSummary.vue
        ForecastChart.vue
        SampleMetricTable.vue
    tests/
      unit/
      e2e/
```

Frontend state flow:

```text
Upload CSV
-> Raw preview
-> Configure time/target/split
-> Create DatasetManifest
-> Create DatasetLoadJob
-> Poll LoadJob
-> Create real CapabilityBlock + Track
-> Select Models
-> Create BenchmarkingRun
-> Poll RunProgress every 5 seconds
-> View Ranking / Report / Sample Forecast
```

## TDD Task Breakdown

### Task 0: Backend Test Harness And Configuration

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/errors.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/unit/test_config.py`
- Create: `backend/tests/api/test_error_contract.py`

- [ ] **Step 0.1: RED config test**

Write a test that sets temporary environment values and expects `get_settings()` to return isolated runtime and database paths.

Run: `cd backend && uv run pytest tests/unit/test_config.py -v`

Expected: FAIL because `backend/app/core/config.py` and `get_settings()` do not exist.

- [ ] **Step 0.2: GREEN config implementation**

Implement only `Settings`, `get_settings()`, and derived paths for `uploads`, `samples`, `forecasts`, and `reports`.

Run: `cd backend && uv run pytest tests/unit/test_config.py -v`

Expected: PASS.

- [ ] **Step 0.3: RED API error contract**

Write an API test that calls a route with invalid input and asserts response shape `{error_code, message, details}`.

Run: `cd backend && uv run pytest tests/api/test_error_contract.py -v`

Expected: FAIL because error handlers are not registered.

- [ ] **Step 0.4: GREEN API error contract**

Implement the FastAPI app factory and custom exception handler with the unified error schema.

Run: `cd backend && uv run pytest tests/api/test_error_contract.py -v`

Expected: PASS.

### Task 1: SQLModel Entities And Database Session

**Files:**
- Create: `backend/app/db/session.py`
- Create: `backend/app/db/init_db.py`
- Create: `backend/app/models/dataset.py`
- Create: `backend/app/models/sample.py`
- Create: `backend/app/models/benchmark.py`
- Create: `backend/app/models/metric.py`
- Create: `backend/app/models/report.py`
- Create: `backend/app/models/ranking.py`
- Create: `backend/app/models/model_registry.py`
- Create: `backend/tests/unit/test_dataset_entities.py`
- Create: `backend/tests/unit/test_dataset_load_constraints.py`
- Create: `backend/tests/unit/test_track_relationships.py`
- Create: `backend/tests/unit/test_ranking_list_entity.py`

- [ ] **Step 1.1: RED entity creation test**

Write tests creating `DatasetManifest`, `DatasetLoadJob`, `Shard`, `SampleIndex`, `Track`, `CapabilityBlock`, `BenchmarkingRun`, `Unit`, `Task`, `MetricDefinition`, `MetricResult`, `Report`, `RankingList`, `RankingEntry`, and `RunEvent`. Assert UUID4 string IDs, default statuses, and ISO datetime serialization.

Run: `cd backend && uv run pytest tests/unit/test_dataset_entities.py -v`

Expected: FAIL because SQLModel entities do not exist.

- [ ] **Step 1.2: GREEN entity definitions**

Implement SQLModel entities with fields from the entity design document. Use JSON columns for list/dict fields where needed.

Run: `cd backend && uv run pytest tests/unit/test_dataset_entities.py -v`

Expected: PASS.

- [ ] **Step 1.3: RED persistence constraints**

Write tests proving a `DatasetManifest` can have many failed load jobs, at most one succeeded load job, and at most one successful `Shard(real)`.

Run: `cd backend && uv run pytest tests/unit/test_dataset_load_constraints.py -v`

Expected: FAIL because constraints and repository checks do not exist.

- [ ] **Step 1.4: GREEN persistence constraints**

Implement database indexes or service-level guard functions for unique successful load and shard creation.

Run: `cd backend && uv run pytest tests/unit/test_dataset_load_constraints.py -v`

Expected: PASS.

- [ ] **Step 1.5: Relationship tests**

Write and satisfy tests that `Track` references `CapabilityBlock`, `CapabilityBlock` references `Shard`, and `Track` never references `DatasetManifest` directly.

Run: `cd backend && uv run pytest tests/unit/test_track_relationships.py -v`

Expected: PASS after minimal relationship support is implemented.

### Task 2: CSV DatasetReader

**Files:**
- Create: `backend/app/services/dataset_reader.py`
- Create: `backend/app/services/csv_dataset_reader.py`
- Create: `backend/tests/fixtures/valid_hourly_20.csv`
- Create: `backend/tests/fixtures/csv_factory.py`
- Create: `backend/tests/unit/test_csv_reader_happy_path.py`
- Create: `backend/tests/unit/test_csv_reader_time_validation.py`
- Create: `backend/tests/unit/test_csv_reader_frequency.py`
- Create: `backend/tests/unit/test_csv_reader_targets.py`
- Create: `backend/tests/unit/test_csv_reader_format.py`

- [ ] **Step 2.1: RED happy path reader test**

Write a test for `valid_hourly_20.csv` with `time` and `target`. Assert 20 rows, inferred hourly frequency, UTF-8 encoding, detected delimiter, and float target values.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_happy_path.py -v`

Expected: FAIL because `CsvDatasetReader` does not exist.

- [ ] **Step 2.2: GREEN reader happy path**

Implement the reader protocol and the minimal CSV reader for the valid fixture.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_happy_path.py -v`

Expected: PASS.

- [ ] **Step 2.3: RED time validation suite**

Write tests for supported time formats, parse failure, duplicate timestamps, non-monotonic timestamps, and non-equidistant timestamps.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_time_validation.py -v`

Expected: FAIL on missing validation.

- [ ] **Step 2.4: GREEN time validation**

Implement strict parsing, monotonic check, duplicate rejection, and equal-interval check.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_time_validation.py -v`

Expected: PASS.

- [ ] **Step 2.5: RED target and CSV format tests**

Write tests for single target only, string-to-float conversion, missing target, non-float target, NaN, Inf, UTF-8 BOM, comma/tab/semicolon delimiter, missing header, duplicate columns, missing time column, and missing target column.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_targets.py tests/unit/test_csv_reader_format.py -v`

Expected: FAIL on unsupported validations.

- [ ] **Step 2.6: GREEN target and format validation**

Implement all CSV validation rules while keeping the reader independent from the database.

Run: `cd backend && uv run pytest tests/unit/test_csv_reader_targets.py tests/unit/test_csv_reader_format.py -v`

Expected: PASS.

### Task 3: Sample Windowing, Sample JSONL, And DatasetLoadJob

**Files:**
- Create: `backend/app/services/sample_store.py`
- Create: `backend/app/services/dataset_load_service.py`
- Create: `backend/app/schemas/dataset.py`
- Create: `backend/app/schemas/sample.py`
- Create: `backend/app/api/routes/dataset_manifests.py`
- Create: `backend/app/api/routes/dataset_load_jobs.py`
- Create: `backend/app/api/routes/shards.py`
- Create: `backend/app/api/routes/samples.py`
- Create: `backend/tests/unit/test_sample_windowing.py`
- Create: `backend/tests/unit/test_sample_jsonl_schema.py`
- Create: `backend/tests/unit/test_sample_index_checksum.py`
- Create: `backend/tests/unit/test_dataset_load_job_service.py`
- Create: `backend/tests/api/test_dataset_load_flow.py`

- [ ] **Step 3.1: RED windowing test**

Write a test that uses 20 hourly rows with `context_length=6`, `horizon=3`, `stride=3` and expects 4 windows with starts `0, 3, 6, 9`.

Run: `cd backend && uv run pytest tests/unit/test_sample_windowing.py -v`

Expected: FAIL because windowing is not implemented.

- [ ] **Step 3.2: GREEN windowing**

Implement a pure windowing function that returns sample slices and fails when `context_length + horizon` exceeds row count.

Run: `cd backend && uv run pytest tests/unit/test_sample_windowing.py -v`

Expected: PASS.

- [ ] **Step 3.3: RED Sample JSONL schema test**

Write a test that materializes one sample and asserts `schema_version=sample.v1`, IDs, `target_column_names`, `target_history` shape `[6,1]`, `target_future` shape `[3,1]`, timestamp arrays, empty cov arrays, and `source_row_start/source_row_end`.

Run: `cd backend && uv run pytest tests/unit/test_sample_jsonl_schema.py -v`

Expected: FAIL because `SampleStore` does not exist.

- [ ] **Step 3.4: GREEN SampleStore**

Implement JSONL write/read, canonical JSON checksum, and line or offset storage in `SampleIndex.storage_ref`.

Run: `cd backend && uv run pytest tests/unit/test_sample_jsonl_schema.py tests/unit/test_sample_index_checksum.py -v`

Expected: PASS.

- [ ] **Step 3.5: RED DatasetLoadJob success API**

Write an API test for upload preview, manifest creation, load job creation, succeeded job, generated shard, 4 `SampleIndex` rows, and sample preview.

Run: `cd backend && uv run pytest tests/api/test_dataset_load_flow.py -v`

Expected: FAIL because load service and routes are absent.

- [ ] **Step 3.6: GREEN DatasetLoadJob success path**

Implement upload preview, manifest creation, load job execution, shard creation, sample JSONL materialization, and sample preview routes.

Run: `cd backend && uv run pytest tests/api/test_dataset_load_flow.py -v`

Expected: PASS.

- [ ] **Step 3.7: RED failure and retry tests**

Write tests proving failed load cleans intermediate artifacts, successful manifest cannot reload, and failed manifest can retry after config changes.

Run: `cd backend && uv run pytest tests/unit/test_dataset_load_job_service.py -v`

Expected: FAIL on missing failure handling.

- [ ] **Step 3.8: GREEN failure and retry rules**

Implement cleanup and retry rules.

Run: `cd backend && uv run pytest tests/unit/test_dataset_load_job_service.py tests/api/test_dataset_load_flow.py -v`

Expected: PASS.

### Task 4: CapabilityBlock, Track, Models, And RankingList Initialization

**Files:**
- Create: `backend/app/schemas/benchmark.py`
- Create: `backend/app/schemas/model_registry.py`
- Create: `backend/app/api/routes/capability_blocks.py`
- Create: `backend/app/api/routes/tracks.py`
- Create: `backend/app/api/routes/models.py`
- Create: `backend/app/api/routes/wizard.py`
- Create: `backend/tests/unit/test_capability_blocks.py`
- Create: `backend/tests/unit/test_tracks.py`
- Create: `backend/tests/api/test_dataset_to_track_flow.py`
- Create: `backend/tests/api/test_models_api.py`

- [ ] **Step 4.1: RED CapabilityBlock and Track tests**

Write tests that a `CapabilityBlock(block_type=real)` can contain one or more real shards, a shard belongs to only one block, and a track contains blocks without directly referencing manifest or uploaded files.

Run: `cd backend && uv run pytest tests/unit/test_capability_blocks.py tests/unit/test_tracks.py -v`

Expected: FAIL because services and routes do not exist.

- [ ] **Step 4.2: GREEN block and track creation**

Implement capability block and track services, including automatic `RankingList` creation from `Track.primary_metric_id`.

Run: `cd backend && uv run pytest tests/unit/test_capability_blocks.py tests/unit/test_tracks.py -v`

Expected: PASS.

- [ ] **Step 4.3: RED wizard API test**

Write API test for `POST /wizard/real-dataset-track` returning `track_id`, `capability_block_id`, and `ranking_list_id`.

Run: `cd backend && uv run pytest tests/api/test_dataset_to_track_flow.py -v`

Expected: FAIL because wizard route is absent.

- [ ] **Step 4.4: GREEN wizard API and model registry**

Implement the wizard route and seed/register the 5 MVP stub models: Timer 3.5, Timer 3.0, Chronos 2, toto, TimesFM 2.5.

Run: `cd backend && uv run pytest tests/api/test_dataset_to_track_flow.py tests/api/test_models_api.py -v`

Expected: PASS.

### Task 5: Metric Service

**Files:**
- Create: `backend/app/services/metric_service.py`
- Create: `backend/app/schemas/metric.py`
- Create: `backend/tests/unit/test_sample_metrics.py`
- Create: `backend/tests/unit/test_shard_metrics.py`
- Create: `backend/tests/unit/test_task_unit_metrics.py`
- Create: `backend/tests/unit/test_partial_unit_metric_rules.py`

- [ ] **Step 5.1: RED sample metric tests**

Write tests for sample-level MSE and MAE by flattening all `forecast - target_future` elements into one scalar.

Run: `cd backend && uv run pytest tests/unit/test_sample_metrics.py -v`

Expected: FAIL because metric service is absent.

- [ ] **Step 5.2: GREEN sample metrics**

Implement `compute_sample_metrics(target_future, forecast)`.

Run: `cd backend && uv run pytest tests/unit/test_sample_metrics.py -v`

Expected: PASS.

- [ ] **Step 5.3: RED aggregation tests**

Write tests for shard metrics averaging successful samples, all-failed shard producing no metric, task metrics averaging successful shards, and unit metrics averaging successful tasks.

Run: `cd backend && uv run pytest tests/unit/test_shard_metrics.py tests/unit/test_task_unit_metrics.py -v`

Expected: FAIL on missing aggregation.

- [ ] **Step 5.4: GREEN aggregation metrics**

Implement aggregation helpers and result metadata for success/failure counts.

Run: `cd backend && uv run pytest tests/unit/test_shard_metrics.py tests/unit/test_task_unit_metrics.py -v`

Expected: PASS.

- [ ] **Step 5.5: RED partial unit rule**

Write a test that a partial unit does not produce a ranking metric.

Run: `cd backend && uv run pytest tests/unit/test_partial_unit_metric_rules.py -v`

Expected: FAIL on missing partial-unit rule.

- [ ] **Step 5.6: GREEN partial unit rule**

Implement status-aware unit metric behavior.

Run: `cd backend && uv run pytest tests/unit/test_partial_unit_metric_rules.py -v`

Expected: PASS.

### Task 6: ModelAdapter And Forecast JSONL

**Files:**
- Create: `backend/app/services/model_adapter.py`
- Create: `backend/app/services/stub_timer_adapter.py`
- Create: `backend/app/services/forecast_store.py`
- Create: `backend/tests/unit/test_stub_timer_adapter.py`
- Create: `backend/tests/unit/test_stub_forecast_rule.py`
- Create: `backend/tests/unit/test_forecast_jsonl_schema.py`
- Create: `backend/tests/unit/test_forecast_failure_recording.py`
- Create: `backend/tests/unit/test_forecast_timeout_config.py`

- [ ] **Step 6.1: RED deterministic stub tests**

Write tests proving the same `model_id + sample_id + seed` produces the same forecast and different model bias can change output.

Run: `cd backend && uv run pytest tests/unit/test_stub_timer_adapter.py -v`

Expected: FAIL because adapter does not exist.

- [ ] **Step 6.2: GREEN deterministic stub**

Implement `ModelAdapter` protocol and `StubTimerAdapter` based on last-value naive forecast plus deterministic noise and model bias.

Run: `cd backend && uv run pytest tests/unit/test_stub_timer_adapter.py tests/unit/test_stub_forecast_rule.py -v`

Expected: PASS.

- [ ] **Step 6.3: RED Forecast JSONL tests**

Write tests asserting `forecast.v1`, IDs, `[horizon,target_dim]` forecast shape, matching future timestamps, status field, and failure row error fields.

Run: `cd backend && uv run pytest tests/unit/test_forecast_jsonl_schema.py tests/unit/test_forecast_failure_recording.py -v`

Expected: FAIL because `ForecastStore` is absent.

- [ ] **Step 6.4: GREEN ForecastStore**

Implement forecast JSONL write/read and failure-row persistence.

Run: `cd backend && uv run pytest tests/unit/test_forecast_jsonl_schema.py tests/unit/test_forecast_failure_recording.py -v`

Expected: PASS.

- [ ] **Step 6.5: RED timeout config test**

Write a test asserting default sample forecast timeout is 300 seconds and can be configured.

Run: `cd backend && uv run pytest tests/unit/test_forecast_timeout_config.py -v`

Expected: FAIL on missing config field.

- [ ] **Step 6.6: GREEN timeout config**

Add timeout setting and pass it through adapter calls.

Run: `cd backend && uv run pytest tests/unit/test_forecast_timeout_config.py -v`

Expected: PASS.

### Task 7: BenchmarkingRun Executor And Queue

**Files:**
- Create: `backend/app/services/run_executor.py`
- Create: `backend/app/workers/run_queue.py`
- Create: `backend/app/workers/lifecycle.py`
- Create: `backend/app/api/routes/benchmarking_runs.py`
- Create: `backend/tests/unit/test_run_unit_task_creation.py`
- Create: `backend/tests/unit/test_run_queue.py`
- Create: `backend/tests/unit/test_run_execution_success.py`
- Create: `backend/tests/unit/test_run_partial_success.py`
- Create: `backend/tests/unit/test_run_cancellation.py`
- Create: `backend/tests/unit/test_run_startup_recovery.py`
- Create: `backend/tests/api/test_benchmarking_run_create.py`
- Create: `backend/tests/api/test_run_progress.py`

- [ ] **Step 7.1: RED create-run API test**

Write API test that `POST /benchmarking-runs` returns `benchmarking_run_id` immediately and status is `queued` or `running`.

Run: `cd backend && uv run pytest tests/api/test_benchmarking_run_create.py -v`

Expected: FAIL because run API does not exist.

- [ ] **Step 7.2: GREEN create-run API**

Implement run creation, unit/task expansion, and queue submission without executing all samples inline in the request.

Run: `cd backend && uv run pytest tests/api/test_benchmarking_run_create.py tests/unit/test_run_unit_task_creation.py -v`

Expected: PASS.

- [ ] **Step 7.3: RED queue and execution tests**

Write tests for single running run limit, successful run execution, partial success, cooperative cancellation, and startup recovery to `interrupted_by_server_restart`.

Run: `cd backend && uv run pytest tests/unit/test_run_queue.py tests/unit/test_run_execution_success.py tests/unit/test_run_partial_success.py tests/unit/test_run_cancellation.py tests/unit/test_run_startup_recovery.py -v`

Expected: FAIL on missing executor behavior.

- [ ] **Step 7.4: GREEN executor and queue**

Implement run queue, executor loop, status transitions, cancellation flag checks, startup recovery, and per-thread database session usage.

Run: `cd backend && uv run pytest tests/unit/test_run_queue.py tests/unit/test_run_execution_success.py tests/unit/test_run_partial_success.py tests/unit/test_run_cancellation.py tests/unit/test_run_startup_recovery.py -v`

Expected: PASS.

- [ ] **Step 7.5: RED progress API test**

Write test for `RunProgressDTO` containing run status, units, tasks, recent events, progress counts, ISO datetime strings, and optional report/ranking IDs.

Run: `cd backend && uv run pytest tests/api/test_run_progress.py -v`

Expected: FAIL on missing progress DTO.

- [ ] **Step 7.6: GREEN progress API**

Implement progress aggregation route.

Run: `cd backend && uv run pytest tests/api/test_run_progress.py -v`

Expected: PASS.

### Task 8: Report, Ranking, And Sample Forecast API

**Files:**
- Create: `backend/app/services/report_service.py`
- Create: `backend/app/services/ranking_service.py`
- Create: `backend/app/schemas/report.py`
- Create: `backend/app/schemas/ranking.py`
- Modify: `backend/app/api/routes/reports.py`
- Modify: `backend/app/api/routes/ranking_lists.py`
- Modify: `backend/app/api/routes/samples.py`
- Create: `backend/tests/unit/test_run_summary_report.py`
- Create: `backend/tests/unit/test_cancelled_report.py`
- Create: `backend/tests/unit/test_latest_valid_result.py`
- Create: `backend/tests/unit/test_best_result.py`
- Create: `backend/tests/api/test_ranking_list_api.py`
- Create: `backend/tests/api/test_sample_forecast_api.py`

- [ ] **Step 8.1: RED report tests**

Write tests that succeeded and cancelled runs generate `runtime/reports/{run_id}.json` with model metrics, task summaries, sample forecast links, and cancellation reason when applicable.

Run: `cd backend && uv run pytest tests/unit/test_run_summary_report.py tests/unit/test_cancelled_report.py -v`

Expected: FAIL because report service is absent.

- [ ] **Step 8.2: GREEN report service**

Implement report JSON generation from persisted run, task, metric, and forecast artifact records.

Run: `cd backend && uv run pytest tests/unit/test_run_summary_report.py tests/unit/test_cancelled_report.py -v`

Expected: PASS.

- [ ] **Step 8.3: RED ranking tests**

Write tests that latest valid is not overwritten by failed/partial units, best result picks the lowest metric, and one `RankingList` supports metric/policy views through persisted entries.

Run: `cd backend && uv run pytest tests/unit/test_latest_valid_result.py tests/unit/test_best_result.py -v`

Expected: FAIL because ranking refresh does not exist.

- [ ] **Step 8.4: GREEN ranking service**

Implement ranking snapshot refresh and query DTO.

Run: `cd backend && uv run pytest tests/unit/test_latest_valid_result.py tests/unit/test_best_result.py tests/api/test_ranking_list_api.py -v`

Expected: PASS.

- [ ] **Step 8.5: RED Sample Forecast API test**

Write test that one `sample_id + run_id` returns history, ground truth, multiple model forecasts, model statuses, and sample-level MSE/MAE.

Run: `cd backend && uv run pytest tests/api/test_sample_forecast_api.py -v`

Expected: FAIL because sample forecast route cannot join sample, forecast artifact, model, and metric data.

- [ ] **Step 8.6: GREEN Sample Forecast API**

Implement sample forecast read model aggregation.

Run: `cd backend && uv run pytest tests/api/test_sample_forecast_api.py -v`

Expected: PASS.

### Task 9: Backend End-To-End MVP Flow

**Files:**
- Create: `backend/tests/e2e/test_mvp_benchmarking_flow.py`

- [ ] **Step 9.1: RED full API flow**

Write one end-to-end API test using `valid_hourly_20.csv`: upload preview, create manifest, create load job, get shard with 4 samples, create real dataset track, list/select models, create run, poll to terminal success, read ranking, read report, read sample forecast.

Run: `cd backend && uv run pytest tests/e2e/test_mvp_benchmarking_flow.py -v`

Expected: FAIL on whichever integration gap remains.

- [ ] **Step 9.2: GREEN full API flow**

Close only the integration gaps required by the end-to-end test.

Run: `cd backend && uv run pytest tests/e2e/test_mvp_benchmarking_flow.py -v`

Expected: PASS.

- [ ] **Step 9.3: Backend full verification**

Run: `cd backend && uv run pytest`

Expected: PASS.

### Task 10: Frontend Project Skeleton And API Client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.ts`
- Create: `frontend/src/App.vue`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/datasets.ts`
- Create: `frontend/src/api/tracks.ts`
- Create: `frontend/src/api/models.ts`
- Create: `frontend/src/api/runs.ts`
- Create: `frontend/src/api/results.ts`
- Create: `frontend/src/stores/wizard.ts`
- Create: `frontend/src/tests/api-client.test.ts`

- [ ] **Step 10.1: RED API client tests**

Write Vitest tests for request construction, unified error parsing, ISO datetime pass-through, and typed response handling for upload, load job, run progress, ranking, report, and sample forecast.

Run: `cd frontend && npm test -- api-client`

Expected: FAIL because frontend project and API client do not exist.

- [ ] **Step 10.2: GREEN API client**

Implement minimal Vite/Vue project and API client modules.

Run: `cd frontend && npm test -- api-client`

Expected: PASS.

### Task 11: Frontend Evaluation Wizard

**Files:**
- Create: `frontend/src/pages/EvaluationWizardPage.vue`
- Create: `frontend/src/components/wizard/UploadStep.vue`
- Create: `frontend/src/components/wizard/ColumnAndSplitStep.vue`
- Create: `frontend/src/components/wizard/LoadShardStep.vue`
- Create: `frontend/src/components/wizard/TrackStep.vue`
- Create: `frontend/src/components/wizard/ModelSelectionStep.vue`
- Create: `frontend/src/components/wizard/RunStep.vue`
- Create: `frontend/src/components/wizard/ResultStep.vue`
- Create: `frontend/src/tests/UploadStep.test.ts`
- Create: `frontend/src/tests/ColumnAndSplitStep.test.ts`
- Create: `frontend/src/tests/LoadShardStep.test.ts`
- Create: `frontend/src/tests/RunStep.test.ts`

- [ ] **Step 11.1: RED upload and config component tests**

Write component tests that upload displays preview rows and columns, blocks next step before preview, enforces single target, validates positive context/horizon/stride, and displays API errors.

Run: `cd frontend && npm test -- UploadStep ColumnAndSplitStep`

Expected: FAIL because wizard components do not exist.

- [ ] **Step 11.2: GREEN upload and config components**

Implement upload and configuration steps using typed API client and wizard store.

Run: `cd frontend && npm test -- UploadStep ColumnAndSplitStep`

Expected: PASS.

- [ ] **Step 11.3: RED load and run tests**

Write tests that load job polling displays status timeline and shard summary, model selection requires at least one model, run creation starts 5-second polling, and terminal states stop polling.

Run: `cd frontend && npm test -- LoadShardStep RunStep`

Expected: FAIL on missing components.

- [ ] **Step 11.4: GREEN load and run components**

Implement load, track, model selection, and run steps.

Run: `cd frontend && npm test -- LoadShardStep RunStep`

Expected: PASS.

### Task 12: Frontend Result Views

**Files:**
- Create: `frontend/src/pages/RankingPage.vue`
- Create: `frontend/src/pages/ReportPage.vue`
- Create: `frontend/src/pages/SampleForecastPage.vue`
- Create: `frontend/src/components/results/RankingTable.vue`
- Create: `frontend/src/components/results/ReportSummary.vue`
- Create: `frontend/src/components/results/ForecastChart.vue`
- Create: `frontend/src/components/results/SampleMetricTable.vue`
- Create: `frontend/src/tests/RankingPage.test.ts`
- Create: `frontend/src/tests/ReportPage.test.ts`
- Create: `frontend/src/tests/SampleForecastPage.test.ts`

- [ ] **Step 12.1: RED ranking and report tests**

Write component tests that metric/policy controls query ranking, lower-is-better ranking is displayed, report shows model metrics and task errors, and sample forecast links navigate.

Run: `cd frontend && npm test -- RankingPage ReportPage`

Expected: FAIL because result pages do not exist.

- [ ] **Step 12.2: GREEN ranking and report pages**

Implement ranking and report views.

Run: `cd frontend && npm test -- RankingPage ReportPage`

Expected: PASS.

- [ ] **Step 12.3: RED Sample Forecast page tests**

Write tests that one sample shows history, ground truth, multiple model forecasts, failed model status without a line, and MSE/MAE table.

Run: `cd frontend && npm test -- SampleForecastPage`

Expected: FAIL because page and chart components are absent.

- [ ] **Step 12.4: GREEN Sample Forecast page**

Implement the sample forecast view using SVG or canvas charting with stable dimensions and accessible table fallback.

Run: `cd frontend && npm test -- SampleForecastPage`

Expected: PASS.

### Task 13: Full Verification And Handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-tsbenchmark-mvp-entity-structure-design.md` only if implementation discovers a documented contradiction.
- Modify: `docs/superpowers/done/2026-05-16-tsbenchmark-mvp-tdd-implementation-plan.md` only if task order needs correction before implementation begins.

- [ ] **Step 13.1: Backend full suite**

Run: `cd backend && uv run pytest`

Expected: PASS.

- [ ] **Step 13.2: Frontend unit suite**

Run: `cd frontend && npm test`

Expected: PASS.

- [ ] **Step 13.3: Frontend end-to-end suite**

Run: `cd frontend && npm run test:e2e`

Expected: PASS once the dev server and backend test server are configured.

- [ ] **Step 13.4: Manual MVP smoke test**

Use the browser to run: upload fixed CSV, configure `time/target/context_length=6/horizon=3/stride=3`, load shard, create track, select models, run benchmark, open ranking, report, and sample forecast.

Expected: Full flow completes without console errors or broken artifact links.

## Parallelization Guidance

- Task 0 must finish before backend service tasks.
- After Task 1 defines entities, Task 2 CSV reader and Task 5 metric service can proceed independently.
- After Task 3 sample JSONL is stable, Task 4 track setup and Task 6 adapter/forecast store can proceed independently.
- Task 7 depends on Tasks 3, 4, 5, and 6.
- Task 8 depends on Task 7.
- Task 10 frontend client can start once API DTOs in this plan are accepted.
- Tasks 11 and 12 can use mocked API responses, but final acceptance depends on backend Tasks 7 and 8.

## Risk Controls

- Database sessions must not be shared across request thread and background executor thread.
- Dataset load and run execution must commit state transitions in small transactions.
- File writes should use temporary files and atomic rename for JSONL/report artifacts.
- Failed `DatasetLoadJob` must clean intermediate files but preserve uploaded source and validation summary.
- Cancelled `BenchmarkingRun` must not update ranking.
- Partial `Unit` must not enter ranking even if some task metrics exist.
- Stub forecast determinism is a test boundary for stable report and ranking results.
- Frontend polling must stop on terminal run states and when the page unmounts.

## Review Checklist

- Every confirmed decision from items 1-71 has a task or explicit boundary in this plan.
- No development is executed by this document.
- No git operation is assigned to the implementation agent.
- All production behavior starts with a failing test.
- Backend MVP can be accepted with `cd backend && uv run pytest`.
- Frontend MVP can be accepted with `cd frontend && npm test` and `cd frontend && npm run test:e2e`.
