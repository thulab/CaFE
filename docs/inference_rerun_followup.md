# Inference rerun follow-up

Recorded on 2026-08-04 for review before making one coordinated repair and
starting a new immutable experiment chain.  This file is a diagnosis and work
record only; the completed runtime artifacts remain unchanged.

## Confirmed execution order

The multi-dataset inference entry point is model-major.  It prepares and loads
one model, runs that model over every requested dataset, unloads it, and then
moves to the next model.  Dataset work inside one model phase may be processed
in deterministic parallel batches, with the configured total HTTP concurrency
divided across the active datasets.

## 1. Service input-mode compatibility

CaFE currently reads the legacy top-level `forecast_limits.max_target_count`
and `forecast_limits.max_covariate_count` fields.  The current service publishes
the following fields under `forecast_limits.input_mode`:

- `max_target_count`
- `max_history_covariate_count`
- `supports_future_covariates`
- `max_static_covariate_count`

The service uses `-1` as the unbounded count sentinel.  Merely changing the
field path is insufficient because the existing numeric comparison would treat
`-1` as smaller than every real target or covariate count.

Required repair:

1. Normalize the current nested schema and retain a tested legacy fallback.
2. Interpret `-1` as unbounded and `0` as unsupported where the service
   contract permits zero.
3. Check history-covariate count separately from future-covariate support;
   retain the top-level future-horizon length limit.
4. Freeze the resolved input capability and adaptation decision in the new
   inference contract and each prediction row.
5. Add fixtures for Timer-4.0, Chronos-2, TiRex2, Toto2.0, TimesFM2.5,
   Moirai2, and Timer-3.5 using the live catalog shape.

Impact on the completed run: native multi-target models were recorded as
`independent_univariate`, and supported covariates would have been recorded as
`omitted_unsupported`.  Existing univariate results remain reference results,
but native multivariate or covariate-utilization claims require a new inference
and analysis chain.

## 2. Hierarchical Sales calibration supplement

Add `hierarchical_sales/D` from GIFT-Eval under the ignored data directory and
place the original `hierarchical_sales_data.csv` beside its Arrow file.  The
original CSV supplies the promotion indicators omitted from the GIFT-Eval
Arrow schema.

The existing `gift_hierarchical_sales` adapter already validates the exact 118
SKU leaves and four brand counts, reconstructs deterministic same-brand sibling
projections for `hierarchical_coherence`, verifies CSV quantities against the
Arrow targets, and aligns the paired `PROMO_*` indicators as known-future
covariates for `covariate_response`.

Required execution:

1. Download and checksum both source assets without committing either file.
2. Run the focused adapter tests and a real-calibration smoke check.
3. Create new calibration and generation stage contracts; never add the
   dataset retrospectively to the completed contracts.
4. Preserve the current history-only feature and normalization rules.

The current adapter evaluates local two-child additive projections.  A future
full three-level reconciliation benchmark would require an explicit summing
matrix for total, brand, and all SKU nodes and is outside this repair.

## 3. TimesFM2.5 throughput diagnosis

The deployed service contains commit `bc6bbce` (`optimize benchmark bulk
throughput`).  Its TimesFM change rebatches the model to the actual worker batch
size, so the old per-item serialized forward path is no longer the explanation
for the current result.

The 2026-08-03 formal bulk benchmark nevertheless measured about 97.1 original
views/s at one replica per device, total concurrency 8, and request batch 64.
The formal model-major phase shows the same order of throughput.

The remaining slow path is shape inflation in the current TimesFM pipeline:

- every context is padded to the compiled `max_context` of 15,360, although
  CaFE supplies only 96, 168, or 336 history steps;
- the model decodes the compiled `max_horizon` of 1,024 and slices the result
  to the requested horizon of 48 afterward;
- flip invariance performs a second full decode;
- switching between 64-input and 60-input worker batches also rebinds the
  decode closure repeatedly.

Provenance of this regression:

- the 2026-05-28 TimesFM2.5 integration (`98e9e50`) already padded every input
  to the compiled `max_context` and decoded the compiled `max_horizon` before
  slicing to the requested horizon;
- on 2026-07-21, commit `f7d3494` changed the compiled envelope from the REST
  constants `2,880 / 720` to the checkpoint-native `15,360 / 1,024`.  This is
  the commit that introduced the severe request-shape amplification;
- the 2026-08-03 commit `bc6bbce` changed `per_core_batch_size` to the actual
  worker batch and removed the older per-item GPU serialization, but retained
  the inflated context and horizon dimensions;
- the currently supervised service workers started on 2026-08-03 at 18:03 UTC.
  A successful test against the previously running deployment can therefore
  predate the activation of the 2026-07-21 code even if the commit was already
  present in the checkout.

Required repair and verification:

1. Keep `15,360 / 1,024` as public validation ceilings, but derive the execution
   shape inside each homogeneous worker call.  Round the longest real context
   only to the 32-step input patch boundary and pass the requested horizon to
   `decode`; its 128-step output patching already determines the minimum work.
2. Make batch size and execution context call-local.  Do not mutate the model's
   global forecast configuration or rebuild its decode closure when worker
   batch sizes alternate.  The worker already groups calls by input and output
   length, so no additional cross-shape batching policy is required.
3. In continuous-quantile postprocessing, use the point forecast's actually
   generated length instead of `fc.max_horizon`, then retain the existing final
   slice to the requested horizon.
4. Preserve `normalize_inputs`, continuous quantiles, flip invariance,
   positivity inference, quantile-crossing repair, float32, and XReg behavior.
   Disabling any of these is a model-semantics change and is not part of the
   performance repair.
5. Gate the change on old-versus-new checkpoint output comparisons for context
   lengths 96, 168, and 336 at horizon 48, plus patch-boundary cases.  A
   same-code randomized-model probe found maximum absolute differences below
   `7.2e-7` when reducing `384 / 256` to `96 / 128`; require the real checkpoint
   to pass explicit `rtol/atol` tolerances and the existing precision fixtures.
6. Benchmark point-only and covariate/XReg workloads separately; XReg includes
   CPU regression work and must not inherit the target-only throughput number.
7. Re-run the replica/concurrency sweep after the shape fix.  The old projected
   memory rejection of two replicas was based on the inflated 14.5 GiB per-card
   footprint and must not be reused.
8. Confirm results with a longer steady-state trial and report requests/s,
   original views/s, and actual model inputs/s distinctly.

The benchmark helper also increments `successful_model_inputs` twice per
successful request.  This does not change its `views_per_second` selection, but
the model-input throughput field must be fixed before the next report.

## Coordinated rerun boundary

After approval, make the code and data repairs together, run focused tests,
re-benchmark affected native multi-target and covariate workloads, and create a
new immutable calibration -> generation -> inference -> analysis chain.  Keep
the completed 2026-08-03 artifacts as the independent-univariate reference.

## Completion record (2026-08-04)

The coordinated repair was completed before starting the replacement
experiment:

- The service now retains the public TimesFM ceilings while executing only the
  patch-rounded local context and requested horizon.  Real-checkpoint old/new
  comparisons passed at `rtol=1e-5, atol=1e-6`, including L96/L168/L336,
  boundary lengths, B1/B60/B64, ordinary/bulk, and XReg requests.
- CaFE normalizes the live `input_mode`, freezes it in the inference stage
  contract, checks it again before inference, and stores the exact resolved
  capability and adaptation in every prediction row.  The pipeline service
  API prefix and live-contract query path have dedicated regression coverage.
- The ignored Hierarchical Sales Arrow and original CSV assets were checked by
  SHA-256, loaded together, and exercised through real calibration.  The CSV
  quantity channels match the Arrow targets and promotion indicators remain
  known-future covariates.
- The benchmark helper counts model inputs once per successful request.  Both
  7-model sweeps completed with zero request or output-validation errors.

The formal four-card configuration selected for the 20-dataset run is:

| model | replicas/card | total HTTP concurrency | task batch |
| --- | ---: | ---: | ---: |
| Timer-4.0 | 4 | 32 | 192 |
| Chronos-2 | 2 | 16 | 192 |
| timesfm2.5 | 4 | 32 | 64 |
| tirex2 | 1 | 8 | 512 |
| moirai2 | 1 | 8 | 256 |
| Timer-3.5 | 1 | 8 | 1024 |
| toto2.0 | 2 | 16 | 4 |

The primary-shape sweep selects the global configuration because it represents
19 of the 20 datasets.  The separate hierarchy/covariate sweep validates the
native paths and memory boundary.  Its isolated Timer-4.0 and TiRex2 optima
would trade away more throughput on the dominant primary workload than they
recover on the one hierarchical dataset.
