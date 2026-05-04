# Problemulator: Neural Emulator for Exoplanet Radiative Transfer

A transformer-based emulator that learns the mapping from one-dimensional
atmospheric profiles to layerwise radiative fluxes, trained on outputs from
the [PICASO](https://natashabatalha.github.io/picaso/) radiative transfer code.
The emulator achieves millisecond-scale inference per profile versus seconds for
a full RT solve, enabling its use as a fast surrogate in atmospheric modeling
workflows.

This repository accompanies a manuscript on hot-Jupiter RT emulation. The
primary goal is predictive accuracy on held-out flux targets; throughput and
engineering polish are secondary.

**Documentation:** [Implementation spec](docs/spec.md)

---

## Repository layout

| Directory | Description |
|-----------|-------------|
| `src/` | Core pipeline: preprocessing, dataset, model definition, training loop, and the `main.py` entrypoint. |
| `config/` | JSONC runtime configs for the transformer baseline (`transformer.jsonc`) and LSTM baseline (`lstm.jsonc`). |
| `unit_tests/` | Automated tests covering config validation, dataset/model behavior, and training-loop semantics. |
| `testing/` | Post-training utilities: `export.py` (standalone `.pt2`), `errors.py` (test-set evaluation), `plot_example.py`, `training_progression.py`. |
| [`docs/`](docs/) | [`spec.md`](docs/spec.md) — authoritative implementation specification. |
| `data/` | Runtime data directory (gitignored): raw HDF5 inputs, processed NPY shards, and the dataset split file. |
| `models/` | Runtime models directory (gitignored): checkpoints, training logs, and `test_metrics.json`. |

---

## Prerequisites

- **Conda environment `nn`** — all commands must run inside this environment.
  Running outside `nn` is out of spec.
- **CUDA GPU** — the checked-in configs default to `cuda`; `mps` and `cpu` are
  also supported via `miscellaneous_settings.device_backend`.

---

## Quick start

### Local correctness check (no GPU required)

```bash
conda activate nn

# Run the full unit test suite.
python -m unittest discover -s unit_tests -v

# Static analysis.
ruff check src unit_tests
pyflakes src unit_tests
vulture src unit_tests
```

### Local training run

```bash
conda activate nn

# Train with the default transformer config.
python src/main.py

# Train with the LSTM baseline.
python src/main.py --config config/lstm.jsonc

# Preprocess only (set execution_mode = "normalize" in the config first).
python src/main.py --config config/transformer.jsonc
```

### Post-training evaluation

```bash
conda activate nn

# Export the best checkpoint to a standalone .pt2 with baked-in normalization.
python testing/export.py

# Evaluate the exported model on the held-out test split.
python testing/errors.py

# Plot an example prediction.
python testing/plot_example.py
```

### Full end-to-end run on SLURM

The SLURM launchers live in ``supercomputer_cmds/`` at the top level of the
local checkout alongside the data-generation pipeline. From the repo root:

```bash
# 1. Generate PICASO training shards (job array).
sbatch supercomputer_cmds/gen.sh

# 2. Merge shards, stage data, and train (run after gen.sh finishes).
sbatch supercomputer_cmds/train.sh

# Train a single config.
CONFIG_NAMES="transformer_v2" sbatch supercomputer_cmds/train.sh

# Force a full preprocessing cache rebuild.
REBUILD_PROCESSED=1 sbatch supercomputer_cmds/train.sh
```

---

## Documentation

- [`docs/spec.md`](docs/spec.md) — authoritative implementation specification;
  read this before making any non-trivial change to the source.

---

## Project variables (frozen)

The five dataset variables are fixed for the current manuscript:

| Variable | Role | Unit |
|----------|------|------|
| `pressure_bar` | Profile input | bar |
| `temperature_k` | Profile input | K |
| `orbital_distance_m` | Global input | m |
| `net_thermal_flux` | Layerwise target | erg cm⁻² s⁻¹ |
| `net_reflected_flux` | Layerwise target | erg cm⁻² s⁻¹ |

Do not add or remove variables without updating `data_specification` lists,
`normalization.key_methods`, `variable_units`, and the preprocessing cache.
