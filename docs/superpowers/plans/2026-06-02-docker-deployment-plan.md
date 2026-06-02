# Docker Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TSBenchmark deployable with Docker Compose when the inference service is already running at a configurable address.

**Architecture:** Build the backend as a FastAPI/uvicorn image with a persistent `/var/lib/tsbenchmark` runtime volume. Build the frontend as static Vite output served by nginx, with `/api/*` proxied to the backend. Keep the in-repo stub service as an optional Compose profile for local smoke tests, while the default Compose path points the backend to an external timer service URL.

**Tech Stack:** Docker Compose, Python 3.14 slim, uv, Node 22, nginx, FastAPI, Vue/Vite, SQLite volume.

---

### Task 1: Compose And Image Configuration

**Files:**
- Modify: `docker-compose.yml`
- Modify: `backend/Dockerfile`
- Modify: `frontend/Dockerfile`
- Modify: `frontend/nginx.conf`
- Modify: `frontend/package.json`
- Create: `.env.example`

- [x] Make backend/frontend the default Compose services and move the stub service behind a `stub` profile.
- [x] Configure backend runtime at `/var/lib/tsbenchmark` and publish backend/frontend ports via env defaults.
- [x] Add `host.docker.internal:host-gateway` support for Linux host inference services.
- [x] Raise nginx upload size to support 748M TsFile uploads.
- [x] Add a production frontend build script and use it from the frontend Dockerfile.
- [x] Commit Docker configuration changes.

### Task 2: Deployment And Environment Documentation

**Files:**
- Create: `docs/developer/deployment.md`
- Modify: `docs/developer/README.md`
- Modify: `docs/manual/README.md`
- Modify: `README.md`

- [x] Document Docker quick start with an external timer-rest-service URL.
- [x] Document optional local stub profile for smoke tests.
- [x] Add a full environment variable table with defaults, scope, and deployment notes.
- [x] Link the deployment guide from the developer manual and user manual.
- [x] Fix stale root README wording about full-column ingestion.
- [x] Commit documentation changes.

### Task 3: Verification

**Files:**
- No production file changes expected.

- [x] Run `docker compose config` with required environment variables.
- [x] Run `cd frontend && npm run build`.
- [x] Run `docker compose build backend frontend`.
- [ ] Run a Compose smoke test on non-default host ports against an external/stub timer service URL.
- [x] Run `git diff --check`.
- [x] Record verification result in the final response.

Verification note: `docker compose build backend frontend` was executed, but this
machine's configured Docker registry mirrors failed while resolving official base
image metadata (`python:3.14-slim-bookworm`, `node:22-alpine`, `nginx:1.27-alpine`)
with size-validation/401/EOF errors. The Compose smoke test could not be run
without built images. `docker compose config`, `docker compose --profile stub config`,
and the local frontend production build completed successfully.
