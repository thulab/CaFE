# Docker Deployment And Configuration

This guide is the deployment and environment-variable reference for TSBenchmark.
It assumes the inference service (`timer-rest-service`) is already running in another
container or on a reachable host.

## 1. Docker Compose Quick Start

Copy the example environment file and fill in secrets plus the inference service URL:

```bash
cp .env.example .env
```

Minimum production-oriented `.env` values:

```bash
TSBENCHMARK_AUTH_SECRET=<strong-random-secret-at-least-32-bytes>
TSBENCHMARK_ADMIN_PASSWORD=<strong-admin-password>
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://timer-service:10810
```

Then build and start:

```bash
docker compose up -d --build
```

Default published endpoints:

- Frontend: `http://localhost:5173`
- Backend OpenAPI: `http://localhost:8000/docs`

Stop without deleting data:

```bash
docker compose down
```

Delete the SQLite DB and uploaded/runtime artifacts:

```bash
docker compose down -v
```

## 2. Inference Service Addressing

The backend calls `TSBENCHMARK_TIMER_SERVICE_BASE_URL + TSBENCHMARK_TIMER_SERVICE_API_PREFIX`.
The default API prefix is `/ai/api/v1`.

Common URL choices:

| Scenario | `TSBENCHMARK_TIMER_SERVICE_BASE_URL` |
| --- | --- |
| Inference service is on the same Compose/Docker network | `http://timer-service:10810` |
| Inference service publishes a host port | `http://host.docker.internal:10810` |
| Inference service is on another machine | `http://<gpu-host-or-ip>:10810` |

`docker-compose.yml` includes `host.docker.internal:host-gateway` for Linux hosts,
so the host-published service form works on Linux as well as Docker Desktop.

For a local smoke test without a real inference service, use the optional stub profile:

```bash
TSBENCHMARK_TIMER_SERVICE_BASE_URL=http://stub:10810 \
docker compose --profile stub up -d --build
```

The stub is not part of the default deployment path.

## 3. Runtime Data

The backend stores runtime data in the `tsbenchmark-runtime` Docker volume:

```text
/var/lib/tsbenchmark/
  tsbenchmark.db
  uploads/
  samples/
  forecasts/
  reports/
```

Do not bake uploaded datasets or SQLite DB files into images. Keep them in the
runtime volume or in an external volume mount.

The nginx frontend accepts uploads up to `2g` to cover the current 748M TsFile
inputs with margin.

## 4. Environment Variables

### Backend Runtime

| Variable | Default | Required | Notes |
| --- | --- | --- | --- |
| `TSBENCHMARK_AUTH_SECRET` | none | yes | JWT signing secret. Backend refuses to start without it. Use a strong random value. |
| `TSBENCHMARK_AUTH_TTL_SECONDS` | `28800` | no | JWT lifetime in seconds. |
| `TSBENCHMARK_ADMIN_PASSWORD` | random once if unset | recommended | Initial `admin` password when the User table is empty. Existing users are not overwritten. |
| `TSBENCHMARK_RUNTIME_DIR` | `runtime`; Docker uses `/var/lib/tsbenchmark` | no | Root for uploads, forecasts, reports, and local DB path. |
| `TSBENCHMARK_DATABASE_URL` | `sqlite:///runtime/tsbenchmark.db`; Docker uses `sqlite:////var/lib/tsbenchmark/tsbenchmark.db` | no | SQLModel/SQLite URL. Keep it inside the persistent runtime volume in Docker. |
| `TSBENCHMARK_MODEL_ADAPTER` | `rest` | no | `rest` calls timer-rest-service; `stub` uses the in-process deterministic backend stub. |
| `TSBENCHMARK_TIMER_SERVICE_BASE_URL` | `http://127.0.0.1:10810`; Docker example uses `http://host.docker.internal:10810` | no | External inference service base URL. |
| `TSBENCHMARK_TIMER_SERVICE_API_PREFIX` | `/ai/api/v1` | no | REST API prefix from `docs/reference/rest-api.md`. |
| `TSBENCHMARK_TIMER_SERVICE_MODEL_LOAD_TIMEOUT_SECONDS` | `600` | no | Timeout for model load/unload calls. |
| `TSBENCHMARK_MODEL_LIFECYCLE_MODE` | `sequential_unload` | no | `sequential_unload` reduces peak GPU memory; `keep_loaded` leaves models resident. |
| `TSBENCHMARK_SAMPLE_FORECAST_TIMEOUT_SECONDS` | `300` | no | Timeout for a single sample forecast request. |

### Docker Compose

| Variable | Default | Notes |
| --- | --- | --- |
| `TSBENCHMARK_FRONTEND_PUBLISHED_PORT` | `5173` | Host port mapped to nginx port `80`. |
| `TSBENCHMARK_BACKEND_PUBLISHED_PORT` | `8000` | Host port mapped to backend port `8000`. |

Compose reads `.env` automatically from the repository root.

### Local Development Scripts

These variables affect `./scripts/start-system.sh`, `./scripts/stop-system.sh`,
`./scripts/status-system.sh`, and `./scripts/stub-service.sh`. They are not needed
for Docker unless you explicitly reuse the scripts.

| Variable | Default | Notes |
| --- | --- | --- |
| `TSBENCHMARK_BACKEND_HOST` | `127.0.0.1` | Local backend bind host. |
| `TSBENCHMARK_BACKEND_PORT` | `8000` | Local backend port. |
| `TSBENCHMARK_FRONTEND_HOST` | `127.0.0.1` | Local Vite bind host. |
| `TSBENCHMARK_FRONTEND_PORT` | `5173` | Local Vite port. |
| `TSBENCHMARK_SYSTEM_DIR` | `.tsbenchmark-system` | PID and log directory for local scripts. |
| `TSBENCHMARK_START_STUB` | `1` | `0` skips automatic local REST stub startup. |
| `TSBENCHMARK_STUB_HOST` | `127.0.0.1` | Local REST stub bind host. |
| `TSBENCHMARK_STUB_PORT` | `10810` | Local REST stub port. |
| `TSBENCHMARK_BACKEND_CMD` | generated uvicorn command | Override backend command for tests/debugging. |
| `TSBENCHMARK_FRONTEND_CMD` | generated Vite command | Override frontend command for tests/debugging. |
| `TSBENCHMARK_STUB_CMD` | generated stub uvicorn command | Override stub command for tests/debugging. |
| `TSBENCHMARK_START_GRACE_SECONDS` | `1` | Startup grace before scripts check the process is alive. |
| `TSBENCHMARK_TIMER_PROBE_CONNECT_TIMEOUT_SECONDS` | `1` | Local start script readiness probe connect timeout. |
| `TSBENCHMARK_TIMER_PROBE_TIMEOUT_SECONDS` | `3` | Local start script readiness probe total timeout. |

### Baseline Script

| Variable | Default | Notes |
| --- | --- | --- |
| `TSBENCHMARK_BASELINE_PORT` | `18900` | Isolated backend port for `scripts/baseline-run.sh`. |
| `TSBENCHMARK_AUTH_SECRET` | baseline default | Baseline backend JWT secret. |
| `TSBENCHMARK_ADMIN_PASSWORD` | `baseline-admin-pw` | Baseline admin password. |

## 5. Image Build And Publish

Build locally:

```bash
docker compose build backend frontend
```

Tag and push:

```bash
docker tag tsbenchmark-backend:latest  <registry>/tsbenchmark-backend:<tag>
docker tag tsbenchmark-frontend:latest <registry>/tsbenchmark-frontend:<tag>
docker push <registry>/tsbenchmark-backend:<tag>
docker push <registry>/tsbenchmark-frontend:<tag>
```

Multi-platform publish:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f backend/Dockerfile -t <registry>/tsbenchmark-backend:<tag> --push .

docker buildx build --platform linux/amd64,linux/arm64 \
  -f frontend/Dockerfile -t <registry>/tsbenchmark-frontend:<tag> --push .
```
