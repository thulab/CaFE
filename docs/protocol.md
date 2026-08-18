# CaFE v7 scientific protocol

## 1. Estimand

CaFE measures model behavior on capability-focused extensions of existing
benchmark tasks. A source task is a GIFT-Eval official test instance. Its
history, forecast origin, horizon, native target dimension, and observed
future are retained. A treatment modifies the authentic history and adds a
history-only or legally known-future component to the authentic future.

The analysis reports three separate views:

1. MASE and MAE on both unchanged official instances and treated instances;
2. agreement between the model's forecast change and the treatment's truth
   change;
3. for common-factor and cross-series treatments, prediction degradation when
   the auxiliary input histories are temporally misaligned while the assessed
   target history and scored future stay unchanged.

## 2. Official GIFT-Eval instances

For frequency-specific base horizon `P`, term multiplier `m`, and
`H = mP`, the short-term window count is

\[
W=\operatorname{clip}\left(\left\lceil
0.1\,L_{\min}/H\right\rceil,1,20\right).
\]

M4 uses its frequency-specific horizon and `W=1`. For a record of length `L`,
the official origins are

\[
o_i=L-HW+iH,\qquad i=0,\ldots,W-1.
\]

This is the split and `generate_instances` rule in GIFT-Eval. Records remain
native: a `[D,T]` target produces one `D`-target instance. Channel splitting,
when required by a model, is an inference adaptation and is reassembled before
scoring.

The parity source is pinned to GIFT-Eval revision
`26df7582a5a2a2ef7602b5ded3a9a12fd4da74b2`,
`src/gift_eval/data.py`.

Missing history values use history-only linear interpolation with edge hold.
The future observed mask is retained and metrics use observed cells only.

## 3. Treatment batches

All official instances are used in formal mode. `--max-instances` creates a
non-formal source-order prefix for smoke tests. Instance selection has no
generation seed.

Transformation randomness is counter-based. A draw is keyed by:

```text
official_instance_id × capability_id × capability_level × augmentation_seed
```

The five levels use ordered, non-overlapping intervals. Shared nuisance values
such as direction, pulse shape, or regime amplitude are keyed without the
level and remain fixed across the group.

The treatment is first applied to the complete official history. Inference
then takes the suffix allowed by each model's maximum input context.

## 4. Treatment-to-source distance

The anti-contamination check compares every treatment directly with its
authentic source; it does not compare adjacent levels. Let `A` be the affected
targets, `s_d` the source-history standard deviation for target `d`, and `C`
a context suffix. The distance is

\[
d_C=\frac1{|A|}\sum_{d\in A}
\sqrt{\frac1{|C|}\sum_{t\in C}
\left(\frac{x'_{t,d}-x_{t,d}}{s_d}\right)^2}.
\]

The checked suffixes are the available members of
`{96,168,336,512,1024,L}`. Every treatment satisfies

\[
\min_C d_C\ge 0.10.
\]

For amplitude-controlled mechanisms the physical gain maps the sampled level
coordinate to the distance range. For regime and intermittency, one shared
amplitude is solved from the weakest of the five location/sparsity treatments.
There is no upper distance threshold. The whole five-level group is unavailable
when any treatment does not reach the lower bound.

## 5. Capability availability

Availability is resolved per official instance and capability. Short histories
use the same formulas with their actual length. A cell records an unavailable
reason when it cannot support the required cycles, events, segments, stable
trend direction, native panel dimension, predictive gain, or all five levels.
The unchanged official baseline remains available.

Hierarchical coherence is qualification-only. Covariate response uses the
deterministic periodic calendar path implied by the source frequency; that
path is shared by baseline and treatment and is known over the horizon.

## 6. Validation

Validation binds the source Arrow files, generation manifest, and every
Parquet artifact hash. It streams the compact rows and independently rebuilds
every official instance, five-level treatment, and input ablation. Dense
targets, futures, and calendar covariates are transient replay values and are
not generation artifacts.

## 7. Inference

Inference reads source Arrow and compact contracts directly. It rebuilds a
bounded number of samples in memory, applies model maximum-context truncation,
groups homogeneous shapes, and sends MessagePack bulk requests. No
model-specific task dataset is stored. A model is loaded across all compatible
configured endpoints and GPU devices. Native multivariate requests are used
when supported; otherwise channel requests are reassembled before the float32
forecast is written to source-sharded ZSTD Parquet.

## 8. Analysis

Accuracy is reported with observed-cell MAE and MASE for the official baseline
and every treatment, using the authentic baseline-history MASE scale. For
treatment `a` and baseline `0`, the mechanism estimand is

\[
\Delta y= y^{(a)}-y^{(0)},\qquad
\Delta\hat y=\hat y^{(a)}-\hat y^{(0)}.
\]

On affected targets, CaFE reports effect NRMSE, correlation, and amplitude
ratio. Ranks are separate by capability and level; lower effect NRMSE ranks
better. Levels are neutral capability coordinates: for regime and
intermittency a higher level means less evidence, not a larger amplitude.

For common factor and cross-series dependence, an additional attribution table
compares the full treatment forecast with a forecast for the same treatment
after auxiliary histories are temporally misaligned. The primary statistic is

\[
\Delta\operatorname{MASE}_{\mathrm{ablate}}=
\operatorname{MASE}_{\mathrm{ablated\ input}}-
\operatorname{MASE}_{\mathrm{full\ input}}.
\]

A positive value means the intact auxiliary inputs improved the assessed
forecast. A univariate inference adaptation sees the same assessed-target
history in both tasks and should therefore produce a value near zero. This
attribution audit is reported separately and is not folded into effect NRMSE.

Analysis loads one source shard and its prediction shard at a time. Per-sample
metrics use compressed Parquet; summaries and manifests use JSON.

## 9. Artifacts and stage contracts

The active stages are generation, validation, inference, and analysis. Research
validation scans every treatment contract in parallel and requires its stored
treatment-to-source distance evidence to be internally consistent and above the
protocol minimum. Publication validation additionally verifies manifests,
source and artifact hashes, storage policy, row coverage, and an exact replay
of every compact contract from the official Arrow data. Publication mode is
opt-in through `--validation-mode publication`; research mode is the default.
`experiment.json` stores identity. Each stage contract records config, Git
provenance, and upstream artifact hashes. A protocol change starts a new
experiment id.

The original GIFT-Eval Arrow files are the sole persistent copy of authentic
series. Generation stores replay contracts in ZSTD Parquet, inference stores
float32 prediction shards, and analysis stores scalar metric Parquet. A
preflight estimates the complete experiment footprint and enforces the
configured disk budget before generation begins.
