# Complete reviewer artifact

The external artifact is one self-contained ZIP paired with the frozen CaFE
source revision recorded in its manifest. Sizes below were measured on the
experiment server on 2026-09-02.

The designated reviewer-data folder is
[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/77e1d26573a347e89b6b/).
The current release candidate is `cafe-reviewer-artifact-v1.zip` (22,844,720,139
bytes), with container SHA-256
`3485293f8ba7b06fec17e492b07ca9cbb80812c3edd5b3f0d90c1f4cd4c57cf0` and
manifest schema `cafe.reviewer_artifact.v1`. Its manifest binds the source
snapshot to CaFE commit
`55bd36e3aae9ce523d97faabde6e1243969d42da`. The folder URL is a distribution
location, while the filename, byte size, digest, and manifest identify the
bytes. Availability should be claimed only after an uploaded copy has been
downloaded and checked against this digest.

| Item | Measured size | Contents and purpose |
|---|---:|---|
| CaFE source snapshot | less than 10 MB | exact `research` revision, reproduction scripts, configs, compact analysis snapshots, and tests |
| Official GIFT-Eval Arrow snapshot | 250 MB | authentic source series used by every GIFT-Eval replay |
| Frozen FEV Mini snapshot | 20 MB | exact FEV 0.8.0 Mini suite and local source files |
| GIFT-Eval main experiments | 5.5 GB + 767 MB + 779 MB | contracts, public-model predictions, stage metadata, and per-task analyses for short, medium, and long horizons |
| FEV main experiment | 1.5 GB | contracts, public-model predictions, stage metadata, and per-task analyses |
| Ten stability experiment trees and summary | 7.9 GB | all seeds over the fixed task panel plus the cross-seed stability summary |
| Fine-tuning CaFE contract trees | 1.5 GB each | deterministic reconstruction of the two treatment corpora |
| Fine-tuning metric parts, curves, and LoRA checkpoints | about 130 MB | checkpoint-level re-evaluation without rerunning the two 40k-step training jobs |
| Target-only ablation summaries | 51 MB + 63 MB | paired auxiliary-input removal summaries and coverage audit |

The complete payload is approximately 20 GB before ZIP container overhead.
Parquet and Arrow files are stored without redundant recompression. The public
projection excludes the unreleased internal model, worker logs, host names,
private IP addresses, and absolute server paths. Public benchmark identifiers,
seeds, code revisions, model labels, replay contracts, and numerical outputs
are retained.

The ZIP has this top-level layout:

```text
cafe-reviewer-artifact-v1/
├── README.md
├── MANIFEST.json
├── SHA256SUMS
├── code/CaFE-research-<commit>.tar.gz
├── data/{gift-eval,fev-mini-v0.8.0}/
├── experiments/{main,stability,finetuning-contracts,ablation}/
├── stability-summary/
└── finetuning/{default,effect-nrmse}/{models,results}/
```

`MANIFEST.json` records the source revision, source class, archived relative
path, byte size, SHA-256 digest, and whether a file was projected for public
release. `SHA256SUMS` covers every payload file. After download, run
`sha256sum -c SHA256SUMS` from the extracted top-level directory, then follow
`reproducibility/README.md` in the included CaFE source snapshot.
