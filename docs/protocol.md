# CaFE v6 scientific protocol

## 1. Estimand

CaFE measures model behavior on capability-focused extensions of existing
benchmark tasks. A source task is a GIFT-Eval official test instance. Its
history, forecast origin, horizon, native target dimension, and observed
future are retained. A treatment modifies the authentic history and adds a
history-only or legally known-future component to the authentic future.

The analysis reports two separate estimands:

1. forecast accuracy on the unchanged official GIFT-Eval future;
2. the agreement between the model's forecast change and the treatment's
   truth change.

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
\min_C d_C\ge 0.10,
\qquad \max_C d_C\le1.0,
\qquad \max_{C,d} d_{C,d}\le1.5.
\]

For amplitude-controlled mechanisms the physical gain maps the sampled level
coordinate to the distance range. For regime and intermittency, one shared
amplitude is solved from the weakest of the five location/sparsity treatments.
The whole five-level group is unavailable when the lower and upper conditions
cannot both be met.

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

Validation binds the generation manifest and all JSONL file hashes. It checks
native shapes, source links, five-level completeness, coordinate intervals,
delta hashes, treatment-only distance, regime-location ordering, event-gap
ordering, and availability agreement.

## 7. Inference

The model task retains the generated full-history provenance. If the service
advertises a maximum input length, the task target and covariates are sliced
to that suffix after treatment. Native multivariate requests are used when
supported. Otherwise the inference adapter issues independent univariate
requests and reassembles the native forecast tensor.

## 8. Analysis

Official accuracy is reported with observed-cell MAE and MASE. For treatment
`a` and baseline `0`, the mechanism estimand is

\[
\Delta y= y^{(a)}-y^{(0)},\qquad
\Delta\hat y=\hat y^{(a)}-\hat y^{(0)}.
\]

On affected targets, CaFE reports effect NRMSE, correlation, and amplitude
ratio. Ranks are separate by capability and level; lower effect NRMSE ranks
better. Levels are neutral capability coordinates: for regime and
intermittency a higher level means less evidence, not a larger amplitude.

## 9. Artifacts and stage contracts

The active stages are generation, validation, inference, and analysis.
`experiment.json` stores identity. Each stage contract records config, Git
provenance, and upstream artifact hashes. A protocol change starts a new
experiment id.
