# CaFE v15 structural randomness

CaFE v15 keeps the official benchmark instances and five capability levels
fixed, but upgrades `augmentation_seed` from a dose-only draw to a mechanism
realization seed. For one official instance and capability, the selected
structure is shared by all five levels. The level-specific draw continues to
control only the declared coordinate: treatment RMS, period count, change
location, or event spacing.

The implementation deliberately retains the existing compact set of gates:
history-only capability qualification, treatment-to-source distance bounds,
and a future-effect or horizon-support check where the mechanism needs one.
No future target value participates in structure selection.

## Seeded structures

| Capability | Seed-selected structure |
|---|---|
| trend | stable target subset; whole-history linear, delayed linear, slow curvature, or recurring piecewise-linear profile; onset, curvature, knot interval, knot phase, and recurring slope pattern |
| multi-seasonal | eligible real anchor, generated period order and phase, and sinusoidal or two-harmonic generated waveform |
| time-varying seasonality | qualified top-pool carrier/modulation pair, amplitude modulation or periodic phase drift, and affected target subset |
| regime switching | affected target subset, direction, step/ramp/sigmoid transition, and transition width; level still controls change location |
| predictable intermittency | affected target subset, sign, rectangular/triangular/exponential pulse, width, and bounded alternating gap jitter; level still controls mean gap |
| common factor | qualified PC1/PC2/PC3 loading mixture, loading subset, and stable latent harmonic carrier |
| cross-series dependence | one edge sampled from the top qualified driver/responder/lag pool rather than the single strongest edge |
| covariate impulse response | legal continuous covariate, target, covariate and response signs, event timing, period, exponential/delayed-gamma/triangular kernel, delay, and half-life |

For recurring piecewise trend, one profile law is evaluated across the complete
history and forecast horizon. A knot may therefore occur in the forecast
period; it is the continuation of the same knot interval, phase, and slope
cycle visible in the treatment history, rather than a forecast-only change.

## Replay contract

Each available capability group records:

- `randomness_schema = cafe.structural_randomness.v1`;
- the frozen `structure_metadata`;
- `structure_draw_sha256`;
- `structure_shared_across_levels = true`.

Generation schema v13 and contract schema v10 store these fields while keeping
the source benchmark artifacts as the only dense copy. Publication validation
regenerates the same structure and compares the compact contract exactly.

