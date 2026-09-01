from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NATIVE_CATALOG_SCHEMA = "cafe.native_model_catalog.v1"


@dataclass(frozen=True)
class NativeModelSpec:
    model_id: str
    storage_key: str
    repository_id: str | None
    provider: str
    license: str
    max_input_length: int | None
    max_output_length: int | None
    max_future_covariate_length: int | None
    max_target_count: int | None
    max_history_covariate_count: int | None
    supports_future_covariates: bool
    requires_paired_covariates: bool = False

    def catalog_row(self, model_root: Path) -> dict[str, Any]:
        maximum_targets = -1 if self.max_target_count is None else self.max_target_count
        maximum_covariates = (
            -1
            if self.max_history_covariate_count is None
            else self.max_history_covariate_count
        )
        return {
            "schema_version": NATIVE_CATALOG_SCHEMA,
            "model_id": self.model_id,
            "native_runtime": {
                **asdict(self),
                "weight_path": str(
                    (model_root / "builtin" / self.storage_key).resolve()
                ),
            },
            "forecast_limits": {
                "max_input_length": (
                    -1 if self.max_input_length is None else self.max_input_length
                ),
                "max_future_covs_length": (
                    -1
                    if self.max_future_covariate_length is None
                    else self.max_future_covariate_length
                ),
                "min_input_length": 1,
                "max_output_length": (
                    -1 if self.max_output_length is None else self.max_output_length
                ),
                "input_mode": {
                    "max_target_count": maximum_targets,
                    "max_history_covariate_count": maximum_covariates,
                    "supports_future_covariates": self.supports_future_covariates,
                    "max_static_covariate_count": 0,
                    "requires_future_covariates_with_history": (
                        self.requires_paired_covariates
                    ),
                },
            },
        }


NATIVE_MODEL_SPECS: dict[str, NativeModelSpec] = {
    "Timer-4.0": NativeModelSpec(
        model_id="Timer-4.0",
        storage_key="timer_4p0",
        repository_id=None,
        provider="cafe_vendored",
        license="apache-2.0",
        max_input_length=8192,
        max_output_length=960,
        max_future_covariate_length=960,
        max_target_count=None,
        max_history_covariate_count=None,
        supports_future_covariates=True,
    ),
    "Timer-3.5": NativeModelSpec(
        model_id="Timer-3.5",
        storage_key="timer_3p5",
        repository_id="thuml/Timer-S1",
        provider="cafe_vendored",
        license="apache-2.0",
        max_input_length=11520,
        max_output_length=None,
        max_future_covariate_length=None,
        max_target_count=1,
        max_history_covariate_count=0,
        supports_future_covariates=False,
    ),
    "Chronos-2": NativeModelSpec(
        model_id="Chronos-2",
        storage_key="chronos_2",
        repository_id="amazon/chronos-2",
        provider="chronos-forecasting",
        license="apache-2.0",
        max_input_length=8192,
        max_output_length=1024,
        max_future_covariate_length=1024,
        max_target_count=None,
        max_history_covariate_count=None,
        supports_future_covariates=True,
    ),
    "moirai2": NativeModelSpec(
        model_id="moirai2",
        storage_key="moirai2",
        repository_id="Salesforce/moirai-2.0-R-small",
        provider="cafe_vendored",
        license="cc-by-nc-4.0",
        max_input_length=None,
        max_output_length=None,
        max_future_covariate_length=None,
        max_target_count=1,
        max_history_covariate_count=0,
        supports_future_covariates=False,
    ),
    "toto2.0": NativeModelSpec(
        model_id="toto2.0",
        storage_key="toto_2p0",
        repository_id="Datadog/Toto-2.0-2.5B",
        provider="toto-2",
        license="apache-2.0",
        max_input_length=None,
        max_output_length=None,
        max_future_covariate_length=None,
        max_target_count=None,
        max_history_covariate_count=0,
        supports_future_covariates=False,
    ),
    "timesfm2.5": NativeModelSpec(
        model_id="timesfm2.5",
        storage_key="timesfm2p5",
        repository_id="google/timesfm-2.5-200m-pytorch",
        provider="timesfm",
        license="apache-2.0",
        max_input_length=15360,
        max_output_length=1024,
        max_future_covariate_length=1024,
        max_target_count=1,
        max_history_covariate_count=None,
        supports_future_covariates=True,
        requires_paired_covariates=True,
    ),
    "tirex2": NativeModelSpec(
        model_id="tirex2",
        storage_key="tirex2",
        repository_id="NX-AI/TiRex-2",
        provider="cafe_vendored",
        license="apache-2.0",
        max_input_length=2048,
        max_output_length=320,
        max_future_covariate_length=320,
        max_target_count=None,
        max_history_covariate_count=None,
        supports_future_covariates=True,
    ),
}


def native_catalog(model_root: Path) -> dict[str, dict[str, Any]]:
    return {
        model_id: spec.catalog_row(model_root)
        for model_id, spec in NATIVE_MODEL_SPECS.items()
    }


def model_weight_path(model_root: Path, model_id: str) -> Path:
    try:
        spec = NATIVE_MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"unknown native model: {model_id}") from error
    return (model_root / "builtin" / spec.storage_key).resolve()
