# CaFE

CaFE (Capability-Focused Extension) extends existing time-series benchmarks
with controlled capability treatments. Its active implementation starts from
GIFT-Eval's official test instances, modifies their authentic paths, and asks
forecasting models to predict the corresponding treated futures.

## Pipeline

```text
GIFT-Eval official native test instances
  → capability generation
  → pair and provenance validation
  → model inference
  → official-accuracy and capability-effect analysis
```

There is no calibration stage and no standalone synthetic-curve generator.
The adapter reproduces GIFT-Eval's prediction length, test-window count,
split offset, and rolling origin distance. Native multivariate targets remain
one benchmark sample. A univariate-only model is adapted during inference and
its channel forecasts are reassembled.

Every supported instance×capability cell produces five treatments. Parameter
draws come from ordered, non-overlapping level intervals and are reproducible
from `official_instance_id`, capability, level, and `augmentation_seed`.
Changing the augmentation seed creates another treatment batch over the same
official samples.

Treatments modify the complete retained official history. Model-specific
context truncation happens afterward. Each treatment is at least 0.10
source-scale normalized RMS from its authentic source across standard context
suffixes, while upper-distance limits keep the perturbation local.

## Capabilities

The GIFT-Eval adapter attempts these nine generatable mechanisms per official
instance:

- whole-history linear trend in the sample's own trend direction;
- independent secondary seasonality;
- constrained time-varying seasonal amplitude;
- regime change with level-controlled change location;
- nonlinear persistence;
- predictable intermittency with level-controlled event sparsity;
- native-panel common factor;
- directed predictive cross-series transfer;
- response to deterministic known-future calendar covariates.

Availability is instance-specific. Short or structurally unsuitable samples
remain in the official baseline table and record a capability-unavailable
reason. Hierarchical coherence is currently qualification-only and produces
no ranked treatments.

## Install and test

```bash
uv sync --extra test
uv run pytest
```

## Smoke preparation

This runs generation and validation without starting model services:

```bash
uv run cafe run \
  --experiment-id gift-v6-smoke \
  --dataset-id gift_ett1_h \
  --max-instances 2 \
  --augmentation-seed 2026081601 \
  --stop-after validation
```

`--max-instances` selects a non-formal source-order prefix. Omitting it uses
all official GIFT-Eval test instances.

## Formal run

```bash
uv run cafe run \
  --experiment-id gift-v6-formal \
  --dataset-ids gift_electricity_h gift_ett1_h gift_jena_weather_h \
  --augmentation-seed 2026081601 \
  --models Timer-4.0 Chronos-2 timesfm2.5 tirex2 moirai2 Timer-3.5 toto2.0 \
  --endpoints http://100.102.176.45:10810
```

Artifacts use this layout:

```text
<experiment>/
├── experiment.json
├── stage_contracts/
└── <dataset_id>/
    ├── 01_generation/
    ├── 02_validation/
    ├── 03_inference/
    └── 04_analysis/
```

See [docs/protocol.md](docs/protocol.md) for the frozen scientific protocol
and [docs/real_anchored_ten_capability_design.md](docs/real_anchored_ten_capability_design.md)
for mechanism formulas.

## History

CaFE was extracted from TSBenchmark at commit `21b8452`. The
`monorepo-cutover-2026-07-28` tag and ancestor history remain available.
