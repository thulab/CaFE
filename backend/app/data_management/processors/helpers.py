from __future__ import annotations

import math

from ...config import get_settings, infer_difficulty, infer_periods_for_track, infer_trend_type
from ..domain import SeriesSample, SeriesTruth, TrackKind


def infer_truth(track: TrackKind, series: list[float]) -> SeriesTruth:
    trend_type = infer_trend_type(series)
    difficulty = infer_difficulty(series)
    noise_level = estimate_noise(series)
    periods = infer_periods(track=track, length=len(series))
    return SeriesTruth(
        trend_type=trend_type,
        periods=periods,
        dominant_period=periods[-1],
        amplitude_mode="stable",
        phase_shift=False,
        noise_level=noise_level,
        difficulty=difficulty,
    )


def infer_periods(track: TrackKind, length: int) -> list[int]:
    return infer_periods_for_track(track, length, settings=get_settings())


def estimate_noise(series: list[float]) -> float:
    if len(series) < 3:
        return 0.0
    second_diffs = [series[index] - 2 * series[index - 1] + series[index - 2] for index in range(2, len(series))]
    variance = sum(value * value for value in second_diffs) / max(len(second_diffs), 1)
    return round(math.sqrt(max(variance, 0.0)), 4)


def rebuild_sample(
    sample: SeriesSample,
    track: TrackKind,
    history: list[float],
    target: list[float],
    covariates: dict[str, list[float]],
    *,
    extra_tags: list[str] | None = None,
    processor_note: dict[str, object] | None = None,
) -> SeriesSample:
    notes = dict(sample.notes)
    processors_applied = list(notes.get("processors_applied", []))
    if processor_note is not None:
        processor_name = processor_note.get("name")
        if processor_name is not None:
            processors_applied.append(str(processor_name))
        notes.update({"last_processor": processor_note})
    if processors_applied:
        notes["processors_applied"] = processors_applied

    track_tags = list(sample.track_tags)
    if extra_tags:
        track_tags.extend(extra_tags)

    return sample.model_copy(
        update={
            "history": history,
            "target": target,
            "covariates": covariates,
            "track_tags": track_tags,
            "truth": infer_truth(track=track, series=history + target),
            "notes": notes,
        }
    )
