Workflow
========

The main pipeline moves from raw HDF5 inputs to processed artifacts and then to
trained checkpoints.

Pipeline stages
---------------

The default configured behavior is sequential preprocessing followed by
training.

1. Input validation and split resolution
----------------------------------------

Raw HDF5 inputs are treated as the canonical source of truth. Required keys,
shapes, and non-finite value checks are validated before later stages proceed.
If the configured split file is missing, it is generated automatically before
normalization.

2. Preprocessing
----------------

The preprocessing stage prepares dataset splits, normalization metadata,
padding-aware tensors, and processed shard outputs according to the active
configuration. Existing processed artifacts may be reused only if structure and
fingerprint validation both pass.

3. Training
-----------

The training pipeline uses explicit precision settings, deterministic seeding,
configured hardware setup, and a config-selected sequence model
implementation.

- Missing preprocessing artifacts are hard failures.
- Invalid sample indices are hard failures.
- All-padding batches are hard failures on first occurrence.
- If compile is enabled and compile fails, the run fails instead of falling
  back to eager execution.
- The baseline loss is masked mean-squared error over valid target elements.

4. Evaluation
-------------

Training produces checkpoints, logs, metadata, and immediate held-out test
metrics so the repository retains a simple post-fit summary of performance.

Optimizer, scheduler, and hardware
----------------------------------

The current supported training policy is intentionally narrow.

- Optimizer: AdamW
- Scheduler: warmup followed by cosine decay by default
- Model family: transformer by default, optional bidirectional LSTM baseline
- Allowed runtime backends: CPU, MPS, CUDA
- The checked-in default backend is CUDA so the default config aligns with the
  SLURM launcher in ``run.sh``.

Artifact policy
---------------

Required outputs from training are:

- model checkpoints
- training logs
- training metadata
- ``test_metrics.json`` from an immediate held-out test-loss pass after fitting

The training path does not produce export artifacts. Export and broader
scientific evaluation remain optional manual workflows outside the core
training loop.

Reproducibility and performance
-------------------------------

The current policy prefers performance first and reproducibility second.

- Reasonable reproducibility is required, but full strict determinism is not.
- Preprocessing throughput, training speed, and inference speed all matter.
- The manuscript-scale reference problem is on the order of millions of
  profiles with sequence lengths around 75 layers.
- Reported training time is on the order of hours on a high-end GPU, and
  inference is expected to reach millisecond-scale latency per profile with
  better amortization under batching.
