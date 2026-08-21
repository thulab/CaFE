# CaFE v11 scientific protocol

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
3. for common-factor, cross-series, and covariate treatments, prediction
   degradation when the constructed auxiliary input signal is temporally
   misaligned while the assessed target history and scored future stay
   unchanged.

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

Amplitude-controlled levels are calibrated only on the complete official
history, so the sampled coordinate is the full-history macro distance. State-
dependent persistence is the exception: its coordinate is the ordered fraction
of stability headroom used in the history-detected direction, while source
distance remains a visibility and safety gate. Distance
qualification then checks the distinct contexts that the seven evaluated models
actually receive:

`{min(L,2048), min(L,8192), min(L,11520), min(L,15360), min(L,16384)}`.

Duplicate contexts are evaluated once and retain the corresponding model ids.
Inference refuses to run when a service advertises a different maximum input
length from the one frozen in the generation distance contract.
Every treatment satisfies

\[
\min_C d_C\ge 0.10.
\]

It also satisfies a maximum model-context macro distance of `2.0` and a maximum
single-channel normalized RMS of `3.0`. These bounds are qualification checks;
they never feed back into the physical gain. For regime and intermittency, one
shared amplitude is solved from the weakest full-history distance of the five
location/sparsity treatments. Intermittency additionally chooses a deterministic
phase whose scheduled future pulse intersects the official observed mask, then
uses the same shared amplitude to make every level's future truth-effect RMS at
least `0.05` in authentic-history MASE units. No future target value or model
output is used. The whole five-level group is unavailable when any treatment
violates a lower or upper history-distance bound.
Generation records unavailable-reason counts by capability in the dataset
manifest and preserves the failed level plus its complete distance evidence in
the availability contract for later mechanism audits.

Horizon support is capability-specific rather than a universal partition rule.
TVS fits both coefficients of a slow harmonic envelope from history and requires
each affected target to spend at least 25% of its observed forecast positions
outside 25% of the fitted envelope amplitude. Common factor replaces recursive
AR(1) continuation with a stable harmonic fitted to the authentic PCA factor;
its forecast horizon is split into three relative sections only to verify that
the tail-to-head macro RMS ratio is at least 0.50. The middle section is retained
as diagnostic evidence. Other capabilities use their own support conditions or
construction invariants and do not inherit these two gates.
State-dependent persistence requires a non-flat, decaying future effect: its
peak must occur in the first half of the horizon, tail RMS must be at most 0.90
of the peak, and the relative profile range must be at least 0.10.

## 5. Capability availability

Availability is resolved per official instance and capability. Short histories
use the same formulas with their actual length. A cell records an unavailable
reason when it cannot support the required cycles, events, segments, stable
trend direction, native panel dimension, predictive gain, or all five levels.
The unchanged official baseline remains available.

State-dependent persistence requires at least 96 history points. A nonlinear
AR(1) structure using `z*abs(z)/(1+abs(z))` is fitted once on a prefix and must
beat linear AR(1) by at least 0.005 incremental R-squared on the contiguous
held-out suffix. Ordinary and extreme states must appear in both partitions,
the coefficient direction must agree across history halves, and every level
must remain dynamically stable. Treatment history is generated recursively
with the authentic linear innovations. The nonlinear model must also retain
positive predictive gain over linear AR in rolling multi-step pseudo-forecasts
inside the held-out history. Future innovations use 128 centered circular
moving-block bootstrap paths drawn only from historical residuals; each path is
shared by the linear and nonlinear branches, and their paired differences are
averaged into the conditional-mean truth effect.

Hierarchical coherence is qualification-only. Covariate impulse response is
available only when the source record has a native dynamic-real covariate.
`past_feat_dynamic_real` is visible only in history; `feat_dynamic_real` is
also visible over the forecast horizon. CaFE never upgrades a past-only field
to known-future and does not synthesize calendar covariates from frequency.
Repeated historical impulses and a terminal pre-origin impulse use one fixed
causal kernel. The terminal amplitude is chosen from history scales and the
observed mask so every level has future MASE-standardized effect RMS of at
least `0.05`; the scoring gate verifies this construction invariant.

## 6. Validation

Generation performs mechanism qualification before it writes a treatment.
Default research validation then scans every compact treatment contract and
checks its source-distance, mechanism-scoring, nonlinear-identifiability, and
applicable capability-specific horizon-support evidence in parallel. Publication validation additionally binds
the source Arrow files, generation manifest, and every Parquet artifact hash,
then independently rebuilds every official instance, five-level treatment, and
input ablation. Dense targets, futures, and native covariates are transient
replay values and are not generation artifacts.

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

For target scale (s_{i,j}), mechanism scoring first computes

\[
q_i=\operatorname{RMS}\left(\Delta y_{i,t,j}/s_{i,j}\right).
\]

When (q_i<0.05), treatment MASE remains valid but the mechanism score is
`unavailable_low_truth_effect`; CaFE never replaces a missing effect with an
epsilon denominator. On the remaining affected and observed cells, the primary
mechanism score is

\[
\operatorname{NRMSE}_{pooled}=\sqrt{\frac{
\sum_{i,t,j}((\Delta\hat y-\Delta y)/s_{i,j})^2}{
\sum_{i,t,j}(\Delta y/s_{i,j})^2}}.
\]

CaFE also reports coverage, low-signal count, valid-sample mean NRMSE,
correlation, and amplitude ratio. Ranks are separate by capability and level;
lower pooled effect NRMSE ranks better. Levels are neutral capability
coordinates: for regime and intermittency a higher level means less evidence,
not a larger amplitude.

For state-dependent persistence, analysis additionally reports peak-normalized
effect-profile NRMSE and the absolute error in the first post-peak half-life.
When truth or forecast does not halve inside the requested horizon, the row is
explicitly censored. These diagnostics do not change the primary ranking.

For common factor, cross-series dependence, and covariate impulse response, an
additional attribution table compares the full treatment forecast with a
forecast for the same treatment after the relevant constructed auxiliary
history is temporally misaligned. For the covariate capability, the target
history and future remain byte-identical and the authentic covariate path is
unchanged; only the injected covariate impulse is shifted. The primary
statistic is

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
treatment-to-source distance evidence to be internally consistent and within the
protocol bounds. Publication validation additionally verifies manifests,
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
