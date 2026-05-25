# Repository Guidelines

## Project Structure & Module Organization

This repository contains a TSBenchmark MVP with a FastAPI backend and Vue frontend. `backend/app/` holds backend code: `api/routes/`, `models/`, `schemas/`, `services/`, and `workers/`. `backend/tests/` contains pytest suites in `unit/`, `api/`, `e2e/`, and `fixtures/`. `frontend/src/` holds Vue code: `api/`, `components/wizard/`, `components/results/`, `pages/`, `stores/`, and `tests/`. `docs/` contains specs, plans, the user manual (`docs/manual/README.md`), the developer manual (`docs/developer/` — `README.md` index, `data-model.md`, `key-flows.md`), and `docs/reference/` (external reference docs synced from Feishu via `scripts/sync-feishu-docs.py`). `scripts/` contains local start/stop/status scripts and script tests.

## Build, Test, and Development Commands

- `./scripts/start-system.sh`: start backend and frontend locally.
- `./scripts/status-system.sh`: show service PID/log status.
- `./scripts/stop-system.sh`: stop both services.
- `./scripts/stub-service.sh {start|stop|status}`: run the local timer-rest-service stub (`backend/stub_service/`) on `127.0.0.1:10810` for offline inference; the backend reaches it via `TSBENCHMARK_TIMER_SERVICE_BASE_URL` (set `TSBENCHMARK_MODEL_ADAPTER=stub` for the in-process stub instead).
- `cd backend && uv run pytest`: run all backend tests.
- `cd frontend && npm test`: run frontend Vitest suite.
- `cd frontend && npm run test:e2e`: run frontend smoke test.
- `bash scripts/tests/test_system_scripts.sh`: verify system scripts.

## Coding Style & Naming Conventions

Use clear boundaries: routes validate and delegate, services own behavior, and models stay persistence-only. Python uses 4-space indentation, type hints, and snake_case names. Vue/TypeScript uses PascalCase components and camelCase functions/state. Keep generated artifacts out of commits; `.venv/`, `node_modules/`, `runtime/`, and `.tsbenchmark-system/` are ignored.

## Testing Guidelines

Backend uses pytest; name files `test_*.py` and keep fixtures under `backend/tests/fixtures/`. Frontend uses Vitest with Vue Testing Library; name files `*.test.ts`. Add or update tests for behavior changes, API contracts, CSV validation, run execution, or UI workflows. Run focused tests first, then the relevant full suite before handoff.

## Agent-Specific Instructions

Multi-agent work is allowed when tasks are independent or split by ownership, such as backend services, frontend components, docs, and tests. Give each agent a clear scope and disjoint write set. Avoid concurrent edits to the same file unless one integration agent owns the final merge. Each agent should report changed paths and verification commands.

## Commit & Pull Request Guidelines

Existing history uses short messages such as `add plans`, `update entity doc`, and Chinese summaries. Keep commits concise and task-scoped. Pull requests should include a summary, tests run, UI screenshots when relevant, and notes for schema/API changes.

## Security & Configuration Tips

The MVP assumes a local trusted environment. Do not commit runtime databases, uploaded CSVs, logs, or secrets. Use `TSBENCHMARK_RUNTIME_DIR`, `TSBENCHMARK_DATABASE_URL`, `TSBENCHMARK_BACKEND_PORT`, and `TSBENCHMARK_FRONTEND_PORT` for isolated local runs.
