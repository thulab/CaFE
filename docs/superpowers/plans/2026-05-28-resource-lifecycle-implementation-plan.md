# Resource Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe archive/restore and administrator-only physical purge flows for dataset manifests, shards, tracks, and benchmarking runs.

**Architecture:** A new `ArchivedResource` table stores archive state without changing existing entity schemas or run execution status. Backend lifecycle service owns impact analysis, archive/restore, and explicit purge ordering. Frontend workspace pages use a shared lifecycle API helper and compact confirmation dialog to keep destructive actions predictable.

**Tech Stack:** FastAPI, SQLModel, pytest, Vue 3, TypeScript, Vitest, Vue Testing Library, existing hash routing and i18n.

---

## Source Spec

Implement `docs/superpowers/specs/2026-05-28-resource-lifecycle-design.md`.

Do not add database foreign keys. Do not overwrite `BenchmarkingRun.status` for archive state. Preserve deep-link access to archived resource detail pages and reports.

---

## File Structure

Create:

- `backend/app/models/lifecycle.py` - `ArchivedResource` state table.
- `backend/app/services/resource_lifecycle.py` - impact analysis, archive/restore, purge helpers.
- `backend/tests/api/test_resource_lifecycle.py` - API behavior tests.
- `frontend/src/api/lifecycle.ts` - frontend lifecycle API wrappers.
- `frontend/src/components/ui/ResourceActionDialog.vue` - reusable impact/confirm modal.
- `frontend/src/tests/ResourceLifecycle.test.ts` - frontend lifecycle workflow tests.

Modify:

- `backend/app/db/init_db.py` - import lifecycle model for table creation.
- `backend/app/core/permissions.py` - add `track.delete`, `run.delete`, `admin.purge`.
- `backend/app/api/routes/dataset_manifests.py`, `shards.py`, `tracks.py`, `benchmarking_runs.py` - list filters and lifecycle endpoints.
- `backend/app/services/run_executor.py` - reject new runs for archived tracks.
- `frontend/src/api/datasets.ts`, `tracks.ts`, `runs.ts`, `types.ts` - include archive filters and lifecycle DTOs.
- `frontend/src/pages/DatasetsPage.vue`, `TracksPage.vue`, `TrackPage.vue`, `RunsPage.vue`, `RunDetailPage.vue` - show archived state and actions.
- `frontend/src/i18n/locales/en-US.ts`, `zh-CN.ts` - lifecycle labels and errors.
- `docs/manual/README.md`, `docs/developer/data-model.md`, `docs/developer/key-flows.md` - lifecycle documentation.

---

## Task 1: Backend Archive State and Impact API

**Files:**

- Create: `backend/app/models/lifecycle.py`
- Create: `backend/app/services/resource_lifecycle.py`
- Modify: `backend/app/db/init_db.py`
- Modify: `backend/app/core/permissions.py`
- Modify: `backend/app/api/routes/dataset_manifests.py`
- Modify: `backend/app/api/routes/shards.py`
- Modify: `backend/app/api/routes/tracks.py`
- Modify: `backend/app/api/routes/benchmarking_runs.py`
- Test: `backend/tests/api/test_resource_lifecycle.py`

- [x] **Step 1: Write failing backend tests**

Add tests that create a dataset -> shard -> track -> run chain, then assert:

```python
def test_archived_track_is_hidden_from_list_but_detail_remains(client):
    # archive track, GET /tracks hides it, GET /tracks/{id} returns archived_at

def test_archived_track_cannot_start_new_run(client):
    # archive track, POST /benchmarking-runs returns 409 resource_archived

def test_run_archive_hides_list_but_report_still_loads(client):
    # archive terminal run, list hides it, include_archived shows it, report endpoint still works

def test_dataset_impact_reports_downstream_counts(client):
    # GET /dataset-manifests/{id}/deletion-impact returns affected shards/tracks/runs counts
```

Run:

```bash
cd backend && uv run pytest tests/api/test_resource_lifecycle.py -q
```

Expected: tests fail because lifecycle endpoints do not exist.

- [x] **Step 2: Implement archive table and service primitives**

Add `ArchivedResource` with compound primary key `(resource_type, resource_id)`. Implement:

```python
archive_resource(session, resource_type, resource_id, reason=None)
restore_resource(session, resource_type, resource_id)
archived_at(session, resource_type, resource_id)
archive_map(session, resource_type, resource_ids)
active_filter_ids(session, resource_type, resource_ids, include_archived)
```

- [x] **Step 3: Implement impact builders**

Implement one public function:

```python
deletion_impact(session: Session, resource_type: str, resource_id: str) -> dict
```

Return:

```json
{
  "resource_type": "track",
  "resource_id": "track-id",
  "archive_available": true,
  "purge_available": true,
  "cascade_required": true,
  "affected": {
    "dataset_manifests": 0,
    "load_jobs": 0,
    "shards": 0,
    "series_points": 0,
    "sample_indices": 0,
    "capability_blocks": 1,
    "tracks": 1,
    "benchmarking_runs": 2,
    "reports": 2,
    "forecast_artifacts": 4,
    "metric_results": 12,
    "ranking_entries": 2
  },
  "warnings": []
}
```

- [x] **Step 4: Wire list filters and archive endpoints**

Add `include_archived: bool = False` to list routes. Default lists exclude archived resources; details include `archived_at`.

Add:

```text
GET    /.../{id}/deletion-impact
POST   /.../{id}/archive
POST   /.../{id}/restore
```

Use permissions from the spec.

- [x] **Step 5: Reject new runs on archived tracks**

In run creation service, check `ArchivedResource(resource_type="track", resource_id=track_id)` and raise:

```python
ApiError("resource_archived", "track is archived", {"track_id": track_id}, 409)
```

- [x] **Step 6: Verify and commit**

Run:

```bash
cd backend && uv run pytest tests/api/test_resource_lifecycle.py -q
cd backend && uv run pytest tests/api/test_resource_lists.py tests/api/test_benchmarking_run_create.py -q
git diff --check
```

Commit:

```bash
git add backend/app backend/tests/api/test_resource_lifecycle.py
git commit -m "添加资源归档和影响预览接口"
```

---

## Task 2: Backend Physical Purge

**Files:**

- Modify: `backend/app/services/resource_lifecycle.py`
- Modify: `backend/app/api/routes/dataset_manifests.py`
- Modify: `backend/app/api/routes/shards.py`
- Modify: `backend/app/api/routes/tracks.py`
- Modify: `backend/app/api/routes/benchmarking_runs.py`
- Test: `backend/tests/api/test_resource_lifecycle.py`

- [x] **Step 1: Write failing purge tests**

Extend lifecycle tests:

```python
def test_non_cascade_track_purge_with_runs_returns_409(client):
    # DELETE /tracks/{id} returns purge_requires_cascade

def test_cascade_track_purge_removes_runs_reports_and_ranking(client):
    # DELETE /tracks/{id}?cascade=true removes detail endpoints and list rows

def test_cascade_dataset_purge_removes_downstream_track_and_data(client):
    # DELETE /dataset-manifests/{id}?cascade=true removes manifest, shard, track, run

def test_running_run_cannot_be_purged(client):
    # running/queued run returns run_not_terminal
```

Run:

```bash
cd backend && uv run pytest tests/api/test_resource_lifecycle.py -q
```

Expected: purge tests fail because DELETE endpoints do not exist.

- [x] **Step 2: Implement purge ordering**

Implement:

```python
purge_run(session, run_id)
purge_track(session, track_id, cascade=False)
purge_shard(session, shard_id, cascade=False)
purge_dataset_manifest(session, dataset_manifest_id, cascade=False)
```

Delete child rows explicitly before parents. If `cascade=False` and downstream references exist, raise `purge_requires_cascade` with impact details. If a run is `queued` or `running`, raise `run_not_terminal`.

- [x] **Step 3: Wire DELETE endpoints**

Add DELETE routes requiring `admin.purge`:

```text
DELETE /dataset-manifests/{id}?cascade=false
DELETE /shards/{id}?cascade=false
DELETE /tracks/{id}?cascade=false
DELETE /benchmarking-runs/{id}?cascade=false
```

Return `{ "ok": true, "purged": impact }`.

- [x] **Step 4: Verify and commit**

Run:

```bash
cd backend && uv run pytest tests/api/test_resource_lifecycle.py -q
cd backend && uv run pytest -q
git diff --check
```

Commit:

```bash
git add backend/app backend/tests/api/test_resource_lifecycle.py
git commit -m "添加管理员物理删除资源接口"
```

---

## Task 3: Frontend Lifecycle Controls

**Files:**

- Create: `frontend/src/api/lifecycle.ts`
- Create: `frontend/src/components/ui/ResourceActionDialog.vue`
- Modify: `frontend/src/api/datasets.ts`
- Modify: `frontend/src/api/tracks.ts`
- Modify: `frontend/src/api/runs.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/pages/DatasetsPage.vue`
- Modify: `frontend/src/pages/TracksPage.vue`
- Modify: `frontend/src/pages/TrackPage.vue`
- Modify: `frontend/src/pages/RunsPage.vue`
- Modify: `frontend/src/pages/RunDetailPage.vue`
- Modify: `frontend/src/i18n/locales/en-US.ts`
- Modify: `frontend/src/i18n/locales/zh-CN.ts`
- Test: `frontend/src/tests/ResourceLifecycle.test.ts`

- [x] **Step 1: Write failing frontend tests**

Add tests asserting:

```ts
it('loads archived rows when the workspace toggle is enabled')
it('archives and restores a track through the confirmation dialog')
it('shows purge impact before permanent deletion')
it('shows archived track detail without start-run controls')
```

Run:

```bash
cd frontend && npm test -- ResourceLifecycle.test.ts
```

Expected: tests fail because controls are absent.

- [x] **Step 2: Add lifecycle API helpers and types**

Define:

```ts
export type ResourceType = 'dataset_manifest' | 'shard' | 'track' | 'benchmarking_run';
export interface DeletionImpactDTO {
  resource_type: ResourceType;
  resource_id: string;
  archive_available: boolean;
  purge_available: boolean;
  cascade_required: boolean;
  affected: Record<string, number>;
  warnings: string[];
}
```

Add `includeArchived?: boolean` to list helpers.

- [x] **Step 3: Build reusable confirmation dialog**

`ResourceActionDialog.vue` takes `resourceType`, `resourceId`, `action`, `open` and emits `done`. It fetches impact on open, shows affected counts, and calls archive/restore/purge.

- [x] **Step 4: Wire workspace pages**

Add “Show archived” toggle and row actions to datasets/tracks/runs. Archived rows show a badge. After any action, reload the current list.

- [x] **Step 5: Wire detail pages**

Track detail displays archived state and hides/disables new-run controls while archived. Run detail displays archived state and offers restore/purge actions.

- [x] **Step 6: Verify and commit**

Run:

```bash
cd frontend && npm test -- ResourceLifecycle.test.ts WorkspacePages.test.ts AppRoutes.test.ts
cd frontend && npx vue-tsc --noEmit
cd frontend && npm test
git diff --check
```

Commit:

```bash
git add frontend/src
git commit -m "添加前端资源归档删除操作"
```

---

## Task 4: Documentation and Final Verification

**Files:**

- Modify: `docs/manual/README.md`
- Modify: `docs/developer/data-model.md`
- Modify: `docs/developer/key-flows.md`

- [x] **Step 1: Update documentation**

Document:

- 普通删除是归档，可恢复。
- 已归档资源默认隐藏但深链可访问。
- 管理员永久删除需要影响预览和二次确认。
- 数据模型新增 `ArchivedResource`。
- 物理删除的级联边界。

- [x] **Step 2: Verify all relevant suites**

Run:

```bash
cd backend && uv run pytest
cd frontend && npm test
cd frontend && npx vue-tsc --noEmit
bash scripts/tests/test_system_scripts.sh
git diff --check
```

- [x] **Step 3: Commit docs and final fixes**

Commit:

```bash
git add docs backend frontend
git commit -m "完善资源生命周期文档"
```

---

## Self Review

- Spec coverage: archive, restore, impact preview, purge, default list filtering, detail access, archived track run rejection, docs, and tests are covered.
- Placeholder scan: no TBD/TODO placeholders are required by the plan.
- Type consistency: backend uses `dataset_manifest`, `shard`, `track`, `benchmarking_run`; frontend `ResourceType` matches those strings.
