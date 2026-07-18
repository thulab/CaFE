# Paper v5 E2 formal sample collection

Date: 2026-07-19

## Frozen experiment contract

- Horizon: `48`.
- Context views: `96`, `168`, `336`, and `504`.
- Master representation: one `L=504, H=48` sample; the shorter
  contexts are history suffixes and share the exact same 48-step future.
- Each context view is standardized from its own history before inference
  and scoring.
- Intensities: five within-dataset relative mechanism-strength levels.
- Replication: five fixed rounds and 32 paired groups per round.
- A paired group uses one attempt seed for all five intensities and is
  retained only if every intensity passes construction, dataset-local
  feature support, and dataset-local near-distance checks at all four
  contexts.
- Model inference is not part of this collection run. Later inference must
  retain every per-context prediction and metric. The current main summary
  may select the best context within each master sample.

The full statistical-unit and aggregation contract is defined in
`docs/superpowers/specs/2026-07-19-paper-v5-e2-h48-master-sample-protocol.md`.

## Formal calibration

Artifacts:

- `runtime/paper_exp/v5/01_nine_capability_suite`

The build used 20 dataset/task views. Dataset-local calibration produced 63
supported dataset-capability cells:

| Capability | Supported datasets |
| --- | ---: |
| trend | 10 |
| multi-seasonal | 10 |
| time-varying seasonality | 9 |
| regime switching | 9 |
| nonlinear persistence | 9 |
| predictable intermittency | 10 |
| common factor | 3 |
| covariate response | 2 |
| hierarchical coherence | 1 |

The remaining 117 cells are recorded as unsupported rather than imputed:
52 variable-structure mismatches, 43 missing required task views, 19 failed
independent train/holdout splits, two cells without real/generator tolerance
overlap, and one conditioning calibration failure.

Qualification tested all 63 supported cells with five intensities and eight
samples per intensity:

- Expected and accepted samples: `2,520 / 2,520`.
- Expected and accepted context views: `10,080 / 10,080`.
- Failed samples: `0`.
- `all_supported_cells_qualified`: `true`.

Calibration provenance:

| File | SHA-256 |
| --- | --- |
| `dataset_capability_support_matrix.json` | `8f15d0cd862c64553f9580a3b68d70ecf07ffe1f99a8855064abc5990b2e2779` |
| `qualification.json` | `29049e2ad830daa09abf8c51b45f28520c4f0a23798ad655e7cf47232b018832` |
| `manifest.json` | `49fd15160a58af9397d8bf4b1e692eceb06fb3e26fbf10136f78104930f37565` |

## Formal generation

Artifacts:

- `runtime/paper_exp/v5/E2_dynamic_stability`

Observed collection:

- Supported cells: `63`.
- Paired five-intensity groups:
  `63 × 5 rounds × 32 = 10,080`.
- Master samples:
  `10,080 × 5 intensities = 50,400`.
- Potential model views:
  `50,400 × 4 contexts = 201,600` per model.
- Complete cell shards: `63`; incomplete shards: `0`.
- Groups accepted on their first joint attempt: `9,460 / 10,080`
  (`93.85%`).
- Maximum joint attempts for any accepted group: `116`, below the frozen
  limit of `512`.
- Every persisted row has four qualified context views: `true`.
- Every paired group has all five intensities with a shared accepted attempt
  seed: `true`.

Generation provenance:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `generation_config.json` | 14,328 bytes | `097fde82bbd370467a679e0c3c0adf3dcbb2e0d0e914efd16a06665f5d60dcc5` |
| `sample_manifest.json` | 22,637 bytes | `392e65092ca8bed7cd09814a45971667e26cbeb860402ddb0e6b9986969f73f0` |
| `samples.jsonl` | 1,221,563,657 bytes | `23dae5c78e96a4ce8ba5f6169fa1ce879421f92ed0394d92095d80b23a4b698f` |

The output directory is about 2.3 GB because it intentionally retains both
the deterministic combined `samples.jsonl` and 63 resumable per-cell shards.
