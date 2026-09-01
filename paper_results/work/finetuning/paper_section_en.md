## Cross-seed fine-tuning exposes objective-dependent capability fitting

We study whether direct exposure to CaFE instances enables metric-directed
adaptation. We deterministically select one tenth of the official GIFT-Eval
short-term instances under two augmentation seeds. Seed A (`2026082701`) is
used for fine-tuning and in-seed evaluation, whereas seed B (`2026082702`) is
used only for cross-seed evaluation. The two corpora contain 50,535 and 48,365
treatments, respectively, with no shared treatment sample IDs. They are not
strictly source-instance-disjoint: 240 official instances are shared (10.1% of
seed A and 10.4% of seed B).

We compare two LoRA adaptation protocols. The first uses the standard Chronos-2
multi-quantile loss and samples random forecast origins within each complete
treatment series. The second uses paired official and treatment examples at the
fixed official forecast origin and minimizes a batch-pooled squared effect
NRMSE objective on affected dimensions. These are complete protocols rather
than a loss-only ablation: input pairing, forecast-origin sampling, learning
rate (`1e-4` versus `1e-5`), and effective data exposure also differ. Both are
run for 40k optimizer steps and evaluated every 4k steps using macro-stratum
MASE and treatment-effect NRMSE.

The two protocols move the evaluation metrics in opposite directions. Under
the standard objective, seed-A MASE decreases from 1.0199 to 0.9605 (-5.82%)
and seed-B MASE decreases from 0.9489 to 0.8546 (-9.94%) at 40k. Effect NRMSE
improves only transiently, reaching 0.4306 at 16k on seed A and 0.4395 at 8k on
seed B, before ending above baseline at 0.4770 (+2.88%) and 0.4865 (+5.90%).
Conversely, paired-effect training reduces final effect NRMSE by 25.66% on seed
A (0.4637 to 0.3447; observed minimum 0.3441 at 36k) and by 15.38% on seed B
(0.4594 to 0.3887; minimum 0.3852 at 28k), while increasing final MASE by
21.15% and 19.91%, respectively. Thus, level accuracy and treatment-response
fidelity provide non-redundant signals under adaptation.

The targeted effects transfer to a new augmentation seed. Relative MASE curves
under standard training are strongly correlated across seeds (Pearson
`r=0.903`, Spearman `rho=0.952` over nonzero checkpoints), and the final
cross-seed improvement is 4.12 percentage points larger than the in-seed
improvement. Under paired-effect training, MASE degradation is likewise highly
correlated (`r=0.995`, `rho=0.976`). Cross-seed effect-NRMSE improvement remains
substantial but weaker: its final 15.38% gain retains approximately 60% of the
25.66% in-seed gain, with a 10.27-point gap and lower checkpoint-wise
correlation (`r=0.521`). Exact treatment randomization therefore prevents
literal sample replay but does not prevent distribution-level metric-directed
adaptation.

These results should not be interpreted as evidence that the standard loss
induces memorization rather than rule learning. The standard objective is an
instance-normalized quantile loss over absolute levels, whereas effect NRMSE is
an affected-target, L2, paired-difference criterion. Common-mode errors can
cancel in the latter while harming MASE, providing a direct alternative
explanation for the divergence. Nor do the experiments establish resistance
to training contamination. Publishing only treated series would remove the
exact official--treatment supervision required by our paired loss and may raise
the engineering cost of direct effect-metric fitting, but this access-control
hypothesis requires a treated-only attack experiment. The observed checkpoint
trajectories instead motivate a controlled joint objective combining level and
effect losses and a strictly instance-disjoint third seed for model selection.

**Suggested figure caption -- trajectories.** Relative changes in
macro-stratum MASE and treatment-effect NRMSE during fine-tuning. Solid lines
denote cross-seed evaluation on seed B and dashed lines denote in-seed
evaluation on seed A. The standard objective ultimately improves MASE while
degrading effect fidelity; the paired objective produces the reverse movement.

**Suggested figure caption -- Pareto view.** Checkpoint trajectories in the
MASE--effect-NRMSE plane, expressed as percentage change from step 0. The
lower-left quadrant improves both criteria. Early standard-loss checkpoints
briefly approach this quadrant before later training trades effect fidelity for
lower MASE; paired-effect training moves strongly toward lower effect NRMSE at
the cost of MASE.
