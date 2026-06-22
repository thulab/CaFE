"""Build a model-input view from a full materialized sample.

The model must never receive ground-truth future values (``target_future``).
``build_model_input`` strips those values and replaces them with ``horizon``
(an int) so adapters can size their ``output_length`` without peeking at the
answer.
"""
from __future__ import annotations

_REQUIRED_KEYS = (
    "sample_id",
    "target_column_names",
    "history_timestamps",
    "future_timestamps",
    "target_history",
)

_OPTIONAL_KEYS = (
    "covariate_column_names",
    "history_cov",
    "future_cov",
)


def build_model_input(sample: dict) -> dict:
    """Return a model-input view of *sample* with ``target_future`` removed.

    The returned dict contains:
    - All keys from ``_REQUIRED_KEYS``.
    - ``horizon``: int derived from ``len(sample["target_future"])``.
    - ``covariate_column_names`` / ``history_cov`` / ``future_cov`` if present in *sample*.

    ``target_future`` (the ground-truth answer) is intentionally excluded so
    that it cannot leak into model inference.
    """
    model_input: dict = {key: sample[key] for key in _REQUIRED_KEYS}
    model_input["horizon"] = len(sample["target_future"])

    for key in _OPTIONAL_KEYS:
        if key in sample:
            model_input[key] = sample[key]

    return model_input
