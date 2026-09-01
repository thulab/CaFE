# Fine-tuning experiment audit notes

## Scope and source

This audit is based on read-only inspection of two runs on `timecho92`:

- Chronos default objective: `/data/xmy/chronos-forecasting/chronos-2-finetuned/cafe-v15-qf-moirai16k-window10-replacement-40k`
- Paired effect objective: `/data/xmy/chronos-forecasting/chronos-2-finetuned/cafe-v15-qf-moirai16k-window10-nrmse-replacement-40k`

The local `raw/` directory is a snapshot of the curve files, all metric parts,
training manifests, final trainer states, and data-selection manifests. The
remote experiments were not modified. All 44 `(objective, corpus, checkpoint)`
groups are complete, and independent aggregation of their rank-local metric
parts reproduces the provided curves to floating-point tolerance.

The inspected remote repository was at Git HEAD
`8589d1988e9676817548e9626738ff06b6ca6370`, but the run scripts were untracked
and `dataset.py`/`pipeline.py` had uncommitted changes. The experiment manifests
do not contain code hashes. `provenance.json` therefore records hashes of the
post-run files inspected here, but those hashes are not an immutable proof of
the exact run-time source. This is a reproducibility issue to fix before the
camera-ready release.

## What was actually trained

Both experiments adapt the same Chronos-2 base model with LoRA (`r=8`,
`alpha=16`) and the same target modules. Both use horizon 48, training context
2,048, a series budget of 32, random sampling with replacement, seed
2026082701, 40,000 optimizer steps, and checkpoints every 4,000 steps.
Inference is float32 with context 8,192 and reports the median forecast.

They are **not a loss-only controlled comparison**:

| Property | Chronos default | Paired effect NRMSE |
|---|---:|---:|
| Training object | treatment sequence only | official/treatment pair |
| Forecast origin in training | random internal origin | fixed official origin |
| Loss | multi-quantile pinball loss after context mean/std normalization | batch-pooled squared error ratio for the paired treatment effect |
| Cells used | all forecast target cells available to the model loss | affected targets only; weak effects excluded |
| Learning rate | `1e-4` | `1e-5` |
| Nominal epoch-equivalent steps | 3,917 | 7,407 |
| 40k epoch equivalents | 10.212 | 5.400 |

The default loss is therefore not MASE. It standardizes each series by the
context standard deviation and sums pinball losses over all predicted
quantiles. Its evaluation MASE happens to be the closer of the two reported
metrics because both are level-error criteria with an absolute-error geometry.
The effect objective uses only the median forecast and minimizes the squared
batch-pooled ratio (NRMSE squared); evaluation instead computes an NRMSE within
each stratum and then macro-averages strata. It is aligned with, but is not
identical to, the reported effect NRMSE.

## Data split and seed transfer

The “10%” refers to a deterministic 1-of-10 fold over official instance IDs;
all eligible treatments of selected instances are materialized. The fold salt
contains the augmentation seed, so seed A (`2026082701`) and seed B
(`2026082702`) select mostly different official instances in addition to
generating different treatments.

| Audit quantity | Seed A / train | Seed B / cross | Intersection |
|---|---:|---:|---:|
| Treatment rows | 50,535 | 48,365 | 0 identical `sample_id`s |
| Unique official instances | 2,382 | 2,309 | 240 |
| `(official instance, capability, level)` keys | 50,535 | 48,365 | 5,035 |

The 240 shared official instances are 10.08% of seed A and 10.39% of seed B
(Jaccard 5.39%). Thus the cross-seed corpus contains no identical synthesized
treatment samples and is mostly new at the official-instance level, but it is
not a strictly instance-disjoint test.

## Metric definitions

For a treatment forecast, MASE is computed over all observed target cells as
the absolute forecast error divided by the target-specific MASE scale. Samples
are first averaged within each `(dataset, capability, level)` stratum, after
which 390 strata are macro-averaged.

Effect NRMSE compares the change induced by a treatment:

`truth effect = treated truth - official truth`

`forecast effect = treated forecast - official forecast`

Both effects are divided by target-specific MASE scales. Only affected target
dimensions are assessed. A treatment is excluded when the RMS standardized
truth effect is below 0.05. Squared errors and squared truth effects are pooled
within a stratum, their ratio is square-rooted, and scoreable strata are
macro-averaged. Seed A contains 50,406 scoreable effects and 388 scoreable
effect strata; seed B contains 48,268 and 390, respectively.

This matters for interpretation: MASE is an all-target L1 level metric, whereas
effect NRMSE is an affected-target L2 relative-change metric. A common forecast
error added to both official and treated predictions cancels in the effect
difference but still harms MASE. Conversely, improving absolute levels need not
improve sensitivity to the synthetic change.

## Main numerical findings

### Chronos default objective

- Seed A MASE falls from 1.0199 to its minimum/final value 0.9605 at 40k
  (`-5.82%`). Effect NRMSE improves transiently to 0.4306 at 16k (`-7.14%`)
  but finishes at 0.4770 (`+2.88%` from baseline).
- Seed B MASE falls from 0.9489 to its minimum/final value 0.8546 at 40k
  (`-9.94%`). Effect NRMSE improves transiently to 0.4395 at 8k (`-4.33%`)
  but finishes at 0.4865 (`+5.90%`).
- Relative MASE trajectories are strongly similar across seeds (Pearson
  `r=0.903`, Spearman `rho=0.952`, nonzero checkpoints). At 40k, the cross-seed
  MASE gain is 4.12 percentage points **larger** than the training-seed gain.
  This pattern is inconsistent with a simple “training examples improve but
  unseen seeded examples do not” memorization signature.

### Paired effect-NRMSE objective

- Seed A effect NRMSE falls from 0.4637 to 0.3441 at 36k (`-25.80%`) and ends
  at 0.3447 (`-25.66%`). MASE has no improving checkpoint: its least-bad
  nonzero checkpoint is 1.1753 at 8k (`+15.23%`) and it ends at 1.2357
  (`+21.15%`).
- Seed B effect NRMSE falls from 0.4594 to 0.3852 at 28k (`-16.15%`) and ends
  at 0.3887 (`-15.38%`). MASE again has no improving checkpoint: the least-bad
  nonzero value is 1.0782 at 8k (`+13.63%`) and the final value is 1.1379
  (`+19.91%`).
- Seed A and B agree strongly on the direction of MASE degradation (Pearson
  `r=0.995`, Spearman `rho=0.976`). Their detailed NRMSE paths are less similar
  (Pearson `r=0.521`, Spearman `rho=0.624`) because seed B improves
  non-monotonically and later. The final seed-B NRMSE gain retains about 60.0%
  of the seed-A gain; the train-minus-cross gain gap is 10.27 percentage points.

### Heterogeneity

At 40k on seed B, default training improves capability-level MASE for seven of
eight capabilities; `common_factor` is the exception (`+2.4%`). Its effect
NRMSE improves only for `multi_seasonal` (`-5.4%`) and worsens most sharply for
`trend` (`+63.2%`). Effect-objective training worsens capability-level MASE for
all eight capabilities. It improves effect NRMSE for five of eight capabilities
but worsens `trend` (`+31.7%`), `common_factor` (`+9.6%`), and
`covariate_impulse_response` (`+33.2%`). The global result is therefore not
uniform across mechanisms.

The descriptive paired-stratum bootstrap supports the direction of the main
final changes, but it is not a formal confidence interval over datasets or
training randomness. For seed B at 40k, the macro-stratum absolute changes are:

- default MASE: `-0.0943`, cell-bootstrap 95% interval `[-0.1174, -0.0734]`;
- default effect NRMSE: `+0.0271`, `[+0.0119, +0.0430]`;
- effect-objective MASE: `+0.1890`, `[+0.1682, +0.2108]`;
- effect-objective effect NRMSE: `-0.0707`, `[-0.0904, -0.0499]`.

## Claim audit

### Supported directly by these runs

1. **The two evaluation criteria are non-redundant under adaptation.** The
   default objective eventually improves MASE while degrading effect NRMSE;
   the paired objective improves effect NRMSE while degrading MASE, on both
   seeds.
2. **Targeted gains transfer to a new augmentation seed.** Seed B has no
   identical treatment sample IDs, yet default training improves MASE by 9.94%
   and paired training improves effect NRMSE by 15.38% at 40k.
3. **Checkpoint choice matters.** Default training briefly improves effect
   NRMSE before reversing, so an endpoint-only report would hide the early
   joint improvement region.
4. **There is a seed-specific component for the paired effect objective.** Its
   final effect-NRMSE gain is 10.27 percentage points larger on seed A than B,
   although the cross-seed gain remains substantial.

### Suggestive, but requires a dedicated experiment

1. **Joint training may be useful.** The Pareto trajectories motivate a loss
   combining absolute level accuracy and paired effect fidelity, but these runs
   do not show that a joint objective improves the frontier.
2. **Withholding original/treatment pairings can raise the engineering cost of
   direct leaderboard fitting.** The implemented effect objective explicitly
   requires both forecasts, so publishing only treated series would remove the
   exact supervision used here. This is an access-control argument, not an
   empirically measured guarantee.
3. **Seed randomization blocks literal treatment-sample replay.** There are no
   shared sample IDs. It does not block distributional learning or reuse of the
   underlying forecasting structure.

### Unsupported or contradicted by the current evidence

1. **“MASE improved because the model only memorized rather than learned a
   rule.”** The runs contain no representation, nearest-neighbor, or controlled
   instance-overlap test. Default MASE improves more on seed B than seed A,
   which directly cuts against the simplest memorization-only account.
2. **“The method is resistant to training contamination.”** This experiment
   demonstrates meaningful cross-seed adaptation, not resistance to it. It is
   best described as a *cross-seed contamination stress test* that exposes
   objective dependence.
3. **“MASE and effect NRMSE are fundamentally antagonistic.”** Objective,
   paired-vs-unpaired input, forecast origin, learning rate, and effective
   exposure all change together. The observed trade-off is real for these two
   protocols, but its cause is not isolated.
4. **“Keeping the original sequence private prevents NRMSE optimization.”** It
   prevents use of this exact paired loss, but an attacker may approximate a
   baseline, exploit other supervision, or optimize a surrogate. No access
   ablation was run.
5. **“Chronos-2 is strong on effect NRMSE because it was pretrained on diverse
   synthetic data.”** That is a plausible hypothesis, but these fine-tuning
   runs do not compare pretraining mixtures and cannot establish the cause.

## Experiments needed for the stronger story

1. Hold the paired data loader, fixed official origin, learning rate schedule,
   batch construction, optimizer steps, and LoRA configuration fixed; vary only
   `L = L_level + lambda * L_effect` over a lambda grid and report its Pareto
   frontier.
2. Make seed B strictly official-instance-disjoint (and ideally hold out entire
   datasets) to separate augmentation transfer from source-instance reuse.
3. Cross training seed, augmentation seed, and access condition (treated-only
   versus paired original+treatment) with at least three optimization repeats.
4. Reserve a third seed for checkpoint/lambda selection. Current “best” values
   are descriptive minima selected on the reported corpus; 40k endpoints are
   the safer pre-specified comparison.
5. Add a contamination attack baseline that receives only the information a
   leaderboard participant would see. This directly tests the proposed
   engineering barrier instead of inferring it from metric conflict.

## Output map

- `summary.json`: compact machine-readable audit and key checkpoint values.
- `checkpoint_summary_wide.csv`: baseline, best (including and excluding step
  0), and final values for every objective/corpus/metric.
- `seed_transfer_summary.csv`: cross-seed trajectory correlations and gain gaps.
- `pareto_trajectory.csv`: MASE/effect-NRMSE coordinates and nondominance flags.
- `capability_final_changes.csv`: capability-wise final changes.
- `paired_stratum_bootstrap.csv`: descriptive cell-bootstrap intervals.
- `metric_parts_audit.csv`: completeness and reconstructed stratum counts.
- `training_loss.csv`, `training_loss_summary.csv`: trainer logs.
- `finetuning_*.png/.pdf`: publication figures.
