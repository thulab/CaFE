# Native model inference

CaFE now runs forecasting models inside its own GPU workers. The default path
is:

```text
source Arrow + compact treatment contracts
  → bounded replay and input adaptation in each worker
  → shape-homogeneous native model batch
  → source-sharded float32 prediction Parquet
```

It does not start the former REST application, serialize arrays through
MessagePack, use Uvicorn or ZMQ, query service health, or materialize a
model-specific task file. The service backend remains available through
`--backend service` only for parity and rollback.

## Adapter choices

| CaFE model | Native implementation | Reason |
|---|---|---|
| Chronos-2 | `chronos-forecasting` official package | Official batch and covariate API is sufficient |
| timesfm2.5 | `timesfm` official PyTorch package | Official point forecast plus the frozen CaFE XReg residual path |
| toto2.0 | `toto-2` official package | Official tensor API supports a true stacked batch |
| Timer-4.0 | staged model-only pipeline | Preserves the validated covariate and multivariate path |
| Timer-3.5 | staged model-only pipeline | Preserves the validated checkpoint-specific preprocessing |
| moirai2 | staged model-only pipeline | Preserves the exact univariate adaptation and output semantics |
| tirex2 | staged model-only pipeline | Preserves the validated local implementation and compiled kernels |

“Staged” means only the registry, model classes, and pipeline code are copied;
the REST server, routers, coordinator, persistence, and device-discovery code
are not used. The official upstreams are [Chronos](https://github.com/amazon-science/chronos-forecasting),
[TimesFM](https://github.com/google-research/timesfm),
[Toto](https://github.com/DataDog/toto),
[Timer](https://github.com/thuml/Large-Time-Series-Model),
[Moirai/uni2ts](https://github.com/SalesforceAIResearch/uni2ts), and
[TiRex](https://github.com/NX-AI/tirex-2).

## Provisioning

Use Python 3.12 on inference hosts because Toto 2.0 does not publish a Python
3.11 wheel. Install the project and stage a private, dereferenced checkpoint
copy:

```bash
uv sync --extra inference --extra test --extra fev
uv run python scripts/stage_native_runtime.py \
  --source ../timer-rest-service \
  --destination runtime
```

The resulting layout is:

```text
runtime/
├── model_runtime/              # model-only retained implementations
├── models/builtin/<model>/     # local checkpoint copies
└── model_runtime_manifest.json # file hashes and source revision
```

Treat this directory as immutable. Copy the same staged tree to the same
`<repo>/runtime` location on every worker host. The controller compares each
remote manifest SHA-256 with its local manifest before launching work.

## Parallel execution

One OS process owns one model replica on one CUDA device. Worker slots are the
Cartesian product of repeated `--worker-host` values, `--devices`, and the
model's measured `replicas_per_device`. Source shard `s` is assigned by
`s % worker_count`; each worker replays only its own shards beside the GPU and
returns hash-verified Parquet parts. No dense input array crosses machines.

For one four-GPU host:

```bash
uv run cafe run \
  --experiment-id gift-native-v14 \
  --dataset-id gift_ett1_h \
  --backend native \
  --worker-host timecho92 \
  --devices 0,1,2,3 \
  --distributed-repo-root /data/xmy/CaFE
```

Repeat `--worker-host` for multiple hosts. All selected hosts currently use the
same device list; run separate experiments if their visible GPU indexes differ.
When several datasets/tasks are selected, the top-level GIFT and FEV pipelines
schedule them model-major across stable host/GPU lanes. A direct invocation on
one dataset still splits that dataset over every configured GPU and replica.

## Verification evidence

On `timecho92` (4×RTX 5090, PyTorch 2.10/CUDA), all seven runtimes loaded and
returned finite `(target_dim, 48)` forecasts. A one-view native-versus-service
comparison using the same weights produced these maximum absolute differences:

| Model | max abs difference |
|---|---:|
| Timer-4.0 | 0 |
| Timer-3.5 | 0 |
| Chronos-2 | 0.000664 |
| timesfm2.5 | 0.000770 |
| tirex2 | 0.001024 |
| moirai2 | 0.000250 |
| toto2.0 | 0.000355 |

The small nonzero differences come from package/kernel execution paths rather
than shape or preprocessing differences. A 336-view Chronos end-to-end run on
two GPUs completed with 336 predictions, zero failures, three unique source
shards, and a successful analysis stage. Native took 7.12 seconds versus
13.87 seconds through the service (about 1.95× faster end to end); inference
after model loading was comparable, so the main gain is deletion of service
startup/loading and transport overhead.
