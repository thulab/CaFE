from __future__ import annotations

import math
import random

from ..config import AppSettings, infer_periods_for_track
from ..domain import SeriesSample, SeriesTruth, TrackKind


class SyntheticDatasetGenerator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def generate_sample(
        self,
        rng: random.Random,
        sample_id: str,
        track: TrackKind,
        context_length: int,
        horizon: int,
    ) -> SeriesSample:
        generation = self.settings.benchmark.synthetic_generation
        total_length = context_length + horizon
        periods = infer_periods_for_track(track, context_length, settings=self.settings)
        dominant_period = periods[-1]
        phase_shift = rng.random() < generation.phase_shift_probability
        amplitude_mode = rng.choice(generation.amplitude_modes)
        trend_type = rng.choice(generation.trend_types)
        difficulty = rng.choice(generation.difficulties)
        noise_level = self._noise_level_for(track, difficulty)

        series = []
        for step in range(total_length):
            value = self._trend_value(step, total_length, trend_type)
            for order, period in enumerate(periods, start=1):
                amplitude = generation.amplitude_base * order
                if amplitude_mode == "slow_drift":
                    amplitude *= 1.0 + generation.slow_drift_strength * math.sin(step / max(period, 2))
                elif amplitude_mode == "mid_spike" and step > total_length // 2:
                    amplitude *= generation.mid_spike_multiplier
                phase = generation.phase_shift_radians if phase_shift and step > context_length else 0.0
                value += amplitude * math.sin(2 * math.pi * step / period + phase)
            value += rng.gauss(0.0, noise_level)
            series.append(round(value, 4))

        history = series[:context_length]
        target = series[context_length:]
        covariates: dict[str, list[float]] = {}
        track_tags = [track.value, difficulty]
        notes: dict[str, object] = {"total_length": total_length}

        if track == TrackKind.COVARIATE_ROBUSTNESS:
            helpful = [
                round(
                    value * generation.covariate_helpful_scale
                    + rng.gauss(0.0, noise_level / generation.covariate_helpful_noise_divisor),
                    4,
                )
                for value in series
            ]
            distractors = {
                f"distractor_{index + 1}": [
                    round(
                        generation.covariate_distractor_amplitude
                        * math.sin(2 * math.pi * step / rng.choice(generation.covariate_distractor_period_choices))
                        + rng.gauss(0.0, generation.covariate_distractor_noise_std),
                        4,
                    )
                    for step in range(total_length)
                ]
                for index in range(generation.covariate_distractor_count)
            }
            ordered_items = [("helpful_covariate", helpful), *distractors.items()]
            rng.shuffle(ordered_items)
            covariates = {key: values for key, values in ordered_items}
            notes["covariate_order"] = list(covariates.keys())
            notes["future_known_covariates"] = ["helpful_covariate"]
            track_tags.append("order_shuffle")
        elif track == TrackKind.NOISE_ROBUSTNESS:
            history = [
                round(value + rng.gauss(0.0, noise_level * generation.noise_history_multiplier), 4)
                for value in history
            ]
            covariates["noise_probe"] = [round(rng.gauss(0.0, generation.noise_probe_std), 4) for _ in range(total_length)]
            track_tags.append("noise_augmented")
        elif track == TrackKind.COST_INTENSIVE:
            covariates["calendar_signal"] = [
                round(math.sin(2 * math.pi * step / generation.calendar_signal_period), 4)
                for step in range(total_length)
            ]
            covariates["load_signal"] = [
                round(generation.load_signal_trend_scale * step / total_length + rng.random(), 4)
                for step in range(total_length)
            ]
            notes["future_known_covariates"] = ["calendar_signal", "load_signal"]
            track_tags.extend(["long_context", "cost_sensitive"])

        truth = SeriesTruth(
            trend_type=trend_type,
            periods=periods,
            dominant_period=dominant_period,
            amplitude_mode=amplitude_mode,
            phase_shift=phase_shift,
            noise_level=noise_level,
            difficulty=difficulty,
        )
        return SeriesSample(
            sample_id=sample_id,
            history=history,
            target=target,
            covariates=covariates,
            track_tags=track_tags,
            truth=truth,
            notes=notes,
        )

    def _noise_level_for(self, track: TrackKind, difficulty: str) -> float:
        generation = self.settings.benchmark.synthetic_generation
        return round(generation.noise_base_levels[track.value] * generation.difficulty_factors[difficulty], 4)

    def _trend_value(self, step: int, total_length: int, trend_type: str) -> float:
        generation = self.settings.benchmark.synthetic_generation
        ratio = step / max(total_length - 1, 1)
        if trend_type == "linear":
            return generation.trend_linear_scale * ratio
        if trend_type == "piecewise_linear":
            if ratio < generation.trend_piecewise_first_ratio:
                return generation.trend_piecewise_first_slope * ratio
            if ratio < generation.trend_piecewise_second_ratio:
                return generation.trend_piecewise_second_base + generation.trend_piecewise_second_slope * (
                    ratio - generation.trend_piecewise_first_ratio
                )
            return generation.trend_piecewise_third_base + generation.trend_piecewise_third_slope * (
                ratio - generation.trend_piecewise_second_ratio
            )
        return (
            generation.trend_smooth_base
            + generation.trend_smooth_slope * ratio
            + math.sin(ratio * math.pi) * generation.trend_smooth_wave
        )
