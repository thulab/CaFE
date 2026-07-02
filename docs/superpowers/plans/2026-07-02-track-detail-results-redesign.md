# Track Detail Results Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the track detail page toward a model-results view with hidden run history, track-level capability/sample results, and primary-metric model comparison.

**Architecture:** Add a track-level results read model on the backend, then incrementally reuse existing report visual components on the frontend. Keep ranking semantics on `latest_valid_result` and keep failed units out of result aggregates.

**Tech Stack:** FastAPI, SQLModel, Vue 3 Composition API, Vitest, pytest.

---

### Task 1: Backend Track Results API

**Files:**
- Modify: `backend/app/services/report_service.py`
- Modify: `backend/app/api/routes/tracks.py`
- Test: `backend/tests/api/test_track_results_api.py`

- [x] Write API tests for `GET /tracks/{track_id}/results` covering model status, latest successful unit selection, and sample aggregation.
- [x] Implement `read_track_results(...)` in `report_service.py`.
- [x] Add the route to `tracks.py`.
- [x] Run `cd backend && uv run pytest tests/api/test_track_results_api.py -q`.
- [x] Commit as a backend API step.

### Task 2: Run History Collapse And Model Status Hints

**Files:**
- Modify: `frontend/src/components/tracks/TrackRunPanel.vue`
- Modify: `frontend/src/pages/TrackPage.vue`
- Modify: `frontend/src/api/results.ts`
- Modify: `frontend/src/api/types.ts`
- Test: `frontend/src/tests/TrackRunPanel.test.ts`
- Test: `frontend/src/tests/ResourceLifecycle.test.ts`

- [x] Write failing frontend tests for default-hidden run history and model status labels.
- [x] Add `getTrackResults` client types.
- [x] Pass `model_statuses` into `TrackRunPanel`.
- [x] Remove loaded/loading/not-loaded labels from model choices.
- [x] Add the history toggle button to the start-run card.
- [x] Run focused frontend tests and commit.

### Task 3: Ranking Card Simplification

**Files:**
- Modify: `frontend/src/pages/TrackPage.vue`
- Test: `frontend/src/tests/ResourceLifecycle.test.ts`

- [x] Write a failing test that the policy selector is absent and ranking chart is not duplicated.
- [x] Keep metric selection, fix policy to `latest_valid_result`, and render only `RankingTable`.
- [x] Run focused frontend tests and commit.

### Task 4: Track Capability Profile And Sample List

**Files:**
- Create: `frontend/src/components/results/SampleForecastLinksCard.vue`
- Modify: `frontend/src/pages/TrackPage.vue`
- Test: `frontend/src/tests/ReportPage.test.ts`
- Test: `frontend/src/tests/ResourceLifecycle.test.ts`

- [x] Create a reusable sample link card for track-level sample forecast links.
- [x] Reuse `CapabilityProfile` on track detail using track results payload.
- [x] Add track-level sample list at the bottom of the track page.
- [x] Run focused frontend tests and commit.

### Task 5: Model Comparison Entry

**Files:**
- Create: `frontend/src/components/results/ModelComparisonCard.vue`
- Modify: `frontend/src/pages/TrackPage.vue`
- Test: `frontend/src/tests/ResourceLifecycle.test.ts`

- [x] Write a failing test for selecting two models and highlighting per-dimension primary metric winners.
- [x] Implement the comparison component using `capability_metrics`.
- [x] Do not compute or display a total score.
- [x] Run focused frontend tests and commit.

### Task 6: Final Verification

**Files:**
- No production files expected.

- [ ] Run `cd backend && uv run pytest`.
- [ ] Run `cd frontend && npm test`.
- [ ] Run `git status --short`.
- [ ] Report final commit list and verification output.
