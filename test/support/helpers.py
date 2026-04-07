from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from backend.app.data_management.domain import SeriesSample, SeriesTruth, TrackKind
from backend.app.huggingface import HuggingFaceForecast
from frontend.app import BackendProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / "test" / ".tmp"


@contextmanager
def temporary_runtime_dir(prefix: str = "runtime-") -> Iterator[Path]:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=TEST_TMP_ROOT, prefix=prefix) as tmpdir:
        yield Path(tmpdir)


class AsgiBackendProvider(BackendProvider):
    def __init__(self, app) -> None:
        self.app = app

    def _request(self, method: str, path: str, payload=None):
        async def once():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.request(method, path, json=payload)
                response.raise_for_status()
                return response.json()

        return asyncio.run(once())

    def fetch_user_overview(self):
        return self._request("GET", "/api/v1/overview/user")

    def fetch_admin_overview(self):
        return self._request("GET", "/api/v1/overview/admin")

    def fetch_report(self, report_id):
        return self._request("GET", f"/api/v1/reports/{report_id}")

    def generate_batch(self, payload):
        return self._request("POST", "/api/v1/datasets/generate", payload)

    def load_csv_batch(self, payload):
        return self._request("POST", "/api/v1/datasets/load/csv", payload)

    def run_task(self, payload):
        return self._request("POST", "/api/v1/tasks/run", payload)

    def register_model(self, payload):
        return self._request("POST", "/api/v1/models/register", payload)

    def submit_huggingface_model(self, payload):
        return self._request("POST", "/api/v1/models/register/huggingface", payload)

    def load_model(self, model_id):
        return self._request("POST", f"/api/v1/models/{model_id}/load")


class FakeHuggingFaceRunner:
    def __init__(self, config) -> None:
        self.config = config

    def load(self) -> None:
        if self.config.repo_id == "org/broken-forecast-model":
            raise RuntimeError("intentional runner failure")

    def forecast(self, sample, track) -> HuggingFaceForecast:
        return HuggingFaceForecast(
            prediction=[round(sample.history[-1], 4)] * len(sample.target),
            latency_ms=7.5,
            token_count=160,
            notes={"decision": "fake_huggingface", "repo_id": self.config.repo_id, "track": track.value},
        )

    def forecast_batch(self, samples, track):
        return [self.forecast(sample, track) for sample in samples]


def backend_request(app, method: str, path: str, payload=None):
    async def once():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method, path, json=payload)
            response.raise_for_status()
            return response

    return asyncio.run(once())


def backend_request_raw(app, method: str, path: str, payload=None):
    async def once():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(once())


def build_sample(
    *,
    sample_id: str = "sample-1",
    history: list[float] | None = None,
    target: list[float] | None = None,
    covariates: dict[str, list[float]] | None = None,
    track: TrackKind = TrackKind.FORECAST_ACCURACY,
    truth: SeriesTruth | None = None,
    track_tags: list[str] | None = None,
    notes: dict[str, object] | None = None,
) -> SeriesSample:
    history_values = history or [1.0, 2.0, 3.0, 4.0]
    target_values = target or [5.0, 6.0]
    return SeriesSample(
        sample_id=sample_id,
        history=history_values,
        target=target_values,
        covariates=covariates or {},
        track_tags=track_tags or [track.value],
        truth=truth
        or SeriesTruth(
            trend_type="linear",
            periods=[2, 4],
            dominant_period=2,
            amplitude_mode="stable",
            phase_shift=False,
            noise_level=0.1,
            difficulty="easy",
        ),
        notes=notes or {},
    )


def write_demo_csv(directory: Path) -> Path:
    path = directory / "demo.csv"
    path.write_text(
        "\n".join(
            ["sample_id,step,target,calendar_signal,noise_signal"]
            + [f"series_a,{index},{10 + index * 0.1:.1f},{(index % 24) / 24:.4f},{index * 0.01:.4f}" for index in range(12)]
            + [f"series_b,{index},{20 + index * 0.2:.1f},{(index % 24) / 24:.4f},{index * 0.02:.4f}" for index in range(12)]
        ),
        encoding="utf-8",
    )
    return path
