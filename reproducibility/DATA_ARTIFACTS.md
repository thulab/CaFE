# External data artifact plan

This file records what should accompany the code submission outside Git. Sizes
below were measured on the experiment server on 2026-09-02. The final archive
should include a top-level manifest containing relative paths, byte sizes, and
SHA-256 digests.

## Recommended reviewer archive

| Item | Measured size | Why include it |
|---|---:|---|
| Official GIFT-Eval Arrow snapshot | 250 MB | single authentic source for GIFT-Eval replay |
| Frozen FEV Mini snapshot | 20 MB | exact Mini suite and source files |
| GIFT-Eval main experiments | 5.5 GB + 767 MB + 779 MB | contracts, predictions, stage contracts, and per-task analyses for all three horizons |
| FEV main experiment | 1.5 GB | contracts, predictions, stage contracts, and per-task analyses |
| Ten stability experiment trees | 7.9 GB | full audit trail for all seeds; compact summaries are already in Git |
| Fine-tuning CaFE contract trees | 1.5 GB each | regenerates both treatment corpora without archiving dense caches |
| Fine-tuning metric parts and curves | about 26 MB | independently reconstructs every checkpoint aggregate |
| Fine-tuning LoRA checkpoints | about 104 MB | permits checkpoint evaluation without rerunning training |
| Target-only ablation summaries | 51 MB + 63 MB as stored | paired summaries and coverage audit; paper-facing compact CSVs are already in Git |

The recommended complete archive is roughly 20 GB before compression, plus
small manifests. This is practical for an artifact-service or cloud-drive
release and is preferable to uploading dense fine-tuning caches.

## Minimal archive

If the submission platform has a stricter quota, the minimal useful archive is:

1. the 250 MB GIFT-Eval and 20 MB FEV source snapshots;
2. generation contracts and stage manifests for the four main experiments;
3. the two 1.5 GB fine-tuning contract trees;
4. fine-tuning metric parts, curves, and LoRA checkpoints;
5. all `experiment.json`, `stage_contracts/`, analysis manifests, and a global
   checksum manifest.

Predictions for the main experiments and the ten complete stability trees can
then be a separately downloadable extended archive. The compact analysis
snapshots already committed to Git are sufficient to reproduce the paper's
tables and figures without those large files.

## Do not archive redundantly

The following should not be uploaded unless convenient:

- the 1.8 GB fit and 1.8 GB evaluation Hugging Face treatment caches;
- the 3.6 GB paired-effect cache;
- the 169 MB direct-evaluation cache;
- the 478 MB Chronos-2 base checkpoint, because it is pinned to a public model
  revision and verified by SHA-256;
- Python virtual environments, Hugging Face caches, service logs, temporary
  merged checkpoints, or host-specific runtime paths.

## Sanitization before upload

Before publication, create a new immutable archive rather than exposing the
live `runtime/` directories. The packaging step must:

1. preserve relative experiment layouts and stage-contract hashes;
2. remove host names, IP addresses, usernames, absolute `/data/...` paths,
   scheduler logs, and unrelated models;
3. retain scientific identifiers, seeds, revisions, and model labels needed to
   interpret the results;
4. exclude unreleased model predictions from reviewer-facing artifacts;
5. generate `SHA256SUMS`, a JSON inventory, and extraction instructions;
6. verify the extracted archive with `reproducibility/reproduce.py verify` and
   the corresponding experiment manifests.

No external archive has been created or uploaded by the code changes in this
commit. The paths and sizes above are an actionable packaging plan for the next
step.
