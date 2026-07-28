# Paper v8 Common-Factor and Cross-Series Diagnosis

Date: 2026-07-26

## Scope

This note records the diagnosis obtained by comparing the formal experiment
`v8_formal_411c2c0_20260726_prep20` across:

- foundation-model inference;
- multivariate input ablations;
- strict counterfactual audits;
- structured positive controls;
- post-hoc seed-variance and selectivity analysis.

The relevant capabilities are `common_factor` and
`cross_series_dependence`.

## Common-Factor Diagnosis

The current generator reliably creates shared covariance and an identifiable
first factor, but it does not reliably create a forecasting problem that
requires information from the auxiliary channels.

At context 168, the structured dynamic-factor control has a median factor
trajectory correlation of about 0.716 and a median input-ablation degradation
of about 50.9%. However, its main-mechanism error is slightly worse than the
diagonal AR control (0.591 versus 0.579). The current analysis still accepts
this result because absence of a 10% advantage over diagonal AR is explicitly
removed from the hard failure codes for `common_factor`.

The strict common-factor counterfactual is also not recovered by the generic
history-only structured control: median effect NRMSE is about 1.014, median
effect correlation is approximately zero, and median amplitude ratio is about
0.185. The generation-time identifiability gate is metadata-assisted: it uses
the hidden teaching-episode layout, code matrix, response basis, and response
loadings. It therefore demonstrates oracle solvability rather than blind
learnability from history.

Foundation-model protected-target accuracy changes little under auxiliary
input ablation for most models. The current `common_component_nmae` compares
future principal components and can reward a rank-one forecast shape without
showing that auxiliary histories were used.

Conclusion: the present mechanism supports a claim of shared-factor
reconstruction, but not yet a strong claim of cross-series information use.

## Cross-Series Diagnosis

The delayed linear SCM is structurally identifiable and no pairing,
normalization, or counterfactual leakage bug was found. At context 168, the
blind ridge/ARDL structured control passes all 20 datasets, improves over the
diagonal control by about 22%, and recovers the strict effect almost exactly.

The current intensity construction nevertheless makes the high-dose task too
clean:

- increasing intensity raises the causal gain while also reducing responder
  background variance;
- the I5 lead-lag feature is nearly saturated at one;
- the strict intervention affects only the first `delay` points of the
  48-point future;
- the main responder error is evaluated over the full horizon and can be good
  for an independent-univariate model.

Conclusion: the mechanism is valid, but its dose changes more than one causal
quantity and its strongest levels are too deterministic and narrow to support
all interpretations currently attached to the model ranking.

## Protocol Repair

The repair should:

1. Replace the hidden common-factor code relay with a repeated, blind-learnable
   symmetric shared-state observation construction, with no privileged driver
   or channel-specific lag.
2. Make the common-factor generation gate history-only and independent of
   generator-private metadata.
3. Require the common structured control at context 168 to:
   - beat diagonal AR by at least 10%;
   - degrade by at least 10% under auxiliary input ablation;
   - recover the strict effect with NRMSE at most 0.70, correlation at least
     0.60, and amplitude ratio between 0.30 and 1.70.
4. Hold cross-series responder nuisance variance fixed across intensity and
   vary the causal gain independently.
5. Prefer holdout incremental predictive value over a saturated peak
   correlation when calibrating or validating cross-series dose.
6. Report strict cross-series effect quality on the actually affected future
   prefix as well as on the full horizon.
7. Keep independent-univariate adapters as reference baselines in a separate
   cross-series-utilization audit.

## Single-Dataset Acceptance Test

Run a new immutable test experiment under `runtime/paper_exp/v8_test` through:

```text
calibration -> generation -> validation -> inference -> analysis
```

The pilot must use one dataset, all five intensities, paired seeds, the
96/168/336 views, and enough seeds to exercise the structured gates. At
context 168:

- both structured positive controls must pass;
- common factor must satisfy all three hard criteria listed above;
- cross-series strict recovery must remain within its existing hard limits;
- cross-series background variance must not systematically shrink with dose;
- the cross-series dose curve must remain monotonic without I5 collapsing to
  an effectively deterministic task;
- generated manifests must record the revised protocol and no formal runtime
  experiment may be overwritten.

## Implemented Repair and Pilot Result

The implementation uses generator version
`capts-paper-v8-family-calibrated-v5`, pipeline schema
`paper_v8_pipeline.v15`, and experiment protocol
`paper_v8_experiment_protocol.v8`.

Common factor now combines:

- a symmetric rank-one state with dense loadings and no driver or
  channel-specific lag;
- paired, history-only idiosyncratic observation texture with a clean
  deterministic future;
- a final 48-point auxiliary evidence suffix whose paired state differs while
  the protected channel remains exactly invariant;
- a blind generation gate and a blind shared-fit DFM counterfactual control
  that do not use generator-private code or episode metadata.

Cross-series dependence now:

- keeps responder background magnitude fixed for every intensity of a paired
  seed;
- varies only the directed transfer gain;
- uses `cross_series_incremental_r2` as its calibration coordinate;
- evaluates strict recovery on the history-covered active future prefix and
  audits leakage in the unaffected tail.

The accepted end-to-end pilot is:

```text
runtime/paper_exp/v8_test/
  v8_test_common_cross_v5_20260726_s16_a2
```

It used `gift_ett2_h`, 16 seeds, 64 real anchors, and the models Chronos-2,
tirex2, and timesfm2.5. Calibration, generation, validation, inference, local
analysis, and experiment aggregation all completed. Generation validation
accepted 200 clean samples, 160 input-ablation samples, and 20 robustness
samples. Inference produced 3,612 predictions with no missing rows.

At context 168, the structured controls reported:

| Capability | DFM/ARDL advantage over diagonal | Input-ablation degradation | Strict effect NRMSE | Strict correlation | Strict amplitude | Result |
|---|---:|---:|---:|---:|---:|---|
| common factor | 15.7% | 124.4% | 0.692 | 0.787 | 0.480 | pass |
| cross-series | 10.7% | 17.8% | 0.037 | 1.000 | 0.980 | pass |

The common-factor trajectory correlation was 0.889. Cross-series background
span across intensity was exactly zero within every seed, while mean
incremental R² increased from 0.114 at I1 to 0.339 at I5 rather than
saturating near one.

This is a one-dataset protocol acceptance test, not evidence that all formal
datasets will pass. The common strict NRMSE has limited margin below the 0.70
threshold, so a multi-dataset preparation pilot remains necessary before
starting a new formal experiment.
