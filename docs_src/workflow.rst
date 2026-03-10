Workflow
========

The main pipeline moves from raw HDF5 inputs to processed artifacts and then to
trained checkpoints.

1. Data contract
----------------

Raw HDF5 inputs are treated as the canonical source of truth. Required keys,
shapes, and non-finite value checks are validated before later stages proceed.

2. Preprocessing
----------------

The preprocessing stage prepares dataset splits, normalization metadata,
padding-aware tensors, and processed shard outputs according to the active
configuration.

3. Training
-----------

The training pipeline uses explicit precision settings, deterministic seeding,
configured hardware setup, and a transformer-based model implementation.

4. Evaluation
-------------

Training produces checkpoints, logs, metadata, and immediate held-out test
metrics so the repository retains a simple post-fit summary of performance.
