from __future__ import annotations

import math
import random

from ..config import AppSettings, infer_periods_for_track
from .domain import ChannelLayout, NoiseMode, SeriesSample, SeriesTruth, TrackSpec


class SyntheticDatasetGenerator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def generate_sample(
        self,
        rng: random.Random,
        sample_id: str,
        track_spec: TrackSpec,
        input_length: int,
        prediction_length: int,
    ) -> SeriesSample:
        generation = self.settings.benchmark.synthetic_generation
        total_length = input_length + prediction_length
        periods = infer_periods_for_track(track_spec.track, input_length, settings=self.settings)
        dominant_period = periods[-1]
        phase_shift = rng.random() < generation.phase_shift_probability
        amplitude_mode = rng.choice(generation.amplitude_modes)
        trend_type = rng.choice(generation.trend_types)
        difficulty = rng.choice(generation.difficulties)
        base_noise = self._noise_level_for(track_spec.track.value, difficulty)
        noise_level = round(base_noise * (1.4 if track_spec.noise_mode == NoiseMode.NOISY else 1.0), 4)

        primary_channel = track_spec.target_channels[0] if track_spec.target_channels else "target"
        target_sequences = self._build_channel_sequences(
            rng=rng,
            channel_names=track_spec.target_channels or [primary_channel],
            total_length=total_length,
            periods=periods,
            amplitude_mode=amplitude_mode,
            trend_type=trend_type,
            phase_shift=phase_shift,
            input_length=input_length,
            noise_level=noise_level,
        )
        primary_series = target_sequences[primary_channel]

        input_sequences = self._build_input_sequences(
            rng=rng,
            track_spec=track_spec,
            total_length=total_length,
            primary_series=primary_series,
            target_sequences=target_sequences,
            noise_level=noise_level,
        )
        future_sequences = self._build_future_known_sequences(
            rng=rng,
            track_spec=track_spec,
            total_length=total_length,
            input_sequences=input_sequences,
            noise_level=noise_level,
        )

        history = primary_series[:input_length]
        target = primary_series[input_length:]
        if track_spec.noise_mode == NoiseMode.NOISY:
            history = [
                round(value + rng.gauss(0.0, noise_level * generation.noise_history_multiplier), 4)
                for value in history
            ]
            input_sequences[primary_channel] = list(history)

        covariates = {
            name: list(values)
            for name, values in {
                **{name: sequence for name, sequence in input_sequences.items() if name != primary_channel},
                **future_sequences,
            }.items()
        }

        notes: dict[str, object] = {
            "total_length": total_length,
            "track_variant_id": track_spec.track_variant_id,
            "future_known_covariates": list(track_spec.future_known_channels),
        }
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
            input_channel_values={name: sequence[:input_length] for name, sequence in input_sequences.items()},
            target_channel_values={name: sequence[input_length:] for name, sequence in target_sequences.items()},
            future_known_channel_values=future_sequences,
            channel_layout=ChannelLayout(
                primary_target_channel=primary_channel,
                input_channels=list(track_spec.input_channels),
                target_channels=list(track_spec.target_channels),
                future_known_channels=list(track_spec.future_known_channels),
            ),
            track_tags=[track_spec.track_variant_id, track_spec.track_template_kind.value, track_spec.noise_mode.value, difficulty],
            truth=truth,
            notes=notes,
        )

    def _build_channel_sequences(
        self,
        *,
        rng: random.Random,
        channel_names: list[str],
        total_length: int,
        periods: list[int],
        amplitude_mode: str,
        trend_type: str,
        phase_shift: bool,
        input_length: int,
        noise_level: float,
    ) -> dict[str, list[float]]:
        sequences: dict[str, list[float]] = {}
        for index, channel in enumerate(channel_names, start=1):
            sequences[channel] = self._generate_series(
                rng=rng,
                total_length=total_length,
                periods=periods,
                amplitude_mode=amplitude_mode,
                trend_type=trend_type,
                phase_shift=phase_shift,
                input_length=input_length,
                noise_level=noise_level,
                scale=1.0 + index * 0.12,
                bias=(index - 1) * 0.4,
            )
        return sequences

    def _build_input_sequences(
        self,
        *,
        rng: random.Random,
        track_spec: TrackSpec,
        total_length: int,
        primary_series: list[float],
        target_sequences: dict[str, list[float]],
        noise_level: float,
    ) -> dict[str, list[float]]:
        sequences: dict[str, list[float]] = {}
        for index, channel in enumerate(track_spec.input_channels, start=1):
            if channel in target_sequences:
                sequences[channel] = list(target_sequences[channel])
                continue
            sequences[channel] = [
                round(primary_series[step] * (0.65 + index * 0.08) + rng.gauss(0.0, noise_level * 0.3), 4)
                for step in range(total_length)
            ]
        return sequences

    def _build_future_known_sequences(
        self,
        *,
        rng: random.Random,
        track_spec: TrackSpec,
        total_length: int,
        input_sequences: dict[str, list[float]],
        noise_level: float,
    ) -> dict[str, list[float]]:
        sequences: dict[str, list[float]] = {}
        for channel in track_spec.future_known_channels:
            base = input_sequences.get(channel)
            if base is None:
                base = [round(math.sin(2 * math.pi * step / max(total_length // 3, 2)), 4) for step in range(total_length)]
            sequences[channel] = [round(value + rng.gauss(0.0, noise_level * 0.15), 4) for value in base]
        return sequences

    def _generate_series(
        self,
        *,
        rng: random.Random,
        total_length: int,
        periods: list[int],
        amplitude_mode: str,
        trend_type: str,
        phase_shift: bool,
        input_length: int,
        noise_level: float,
        scale: float,
        bias: float,
    ) -> list[float]:
        generation = self.settings.benchmark.synthetic_generation
        series: list[float] = []
        for step in range(total_length):
            value = self._trend_value(step, total_length, trend_type)
            for order, period in enumerate(periods, start=1):
                amplitude = generation.amplitude_base * order * scale
                if amplitude_mode == "slow_drift":
                    amplitude *= 1.0 + generation.slow_drift_strength * math.sin(step / max(period, 2))
                elif amplitude_mode == "mid_spike" and step > total_length // 2:
                    amplitude *= generation.mid_spike_multiplier
                phase = generation.phase_shift_radians if phase_shift and step > input_length else 0.0
                value += amplitude * math.sin(2 * math.pi * step / period + phase)
            value += bias
            value += rng.gauss(0.0, noise_level)
            series.append(round(value, 4))
        return series

    def _noise_level_for(self, track_key: str, difficulty: str) -> float:
        generation = self.settings.benchmark.synthetic_generation
        return round(generation.noise_base_levels[track_key] * generation.difficulty_factors[difficulty], 4)

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
