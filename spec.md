# Problemulator Scientific Development Spec

## 1. Purpose and Scope

This project is a scientific emulator for 1D radiative transfer profiles.

- Inputs:
  - Layerwise profile variables (for example: `temperature_k`, `pressure_bar`)
  - Per-profile global variables (for example: `orbital_distance_m`)
- Outputs:
  - Layerwise net flux targets (for example: `net_thermal_flux`, `net_reflected_flux`)
- Scope:
  - Scientific correctness, clean design, and high performance
  - Extensible to additional variables and related profile-to-profile regression tasks
- Non-goal:
  - Production hardening and backward-compatibility shims
  - Rollout or in-the-loop simulation workflows; this repository does not provide them currently

### 1.1 Scientific Objective (Current Project)

- Learn a mapping from 1D atmospheric state to radiative transfer outputs:
  - Inputs: layerwise pressure-temperature structure plus global context.
  - Outputs: layerwise net thermal and reflected (scattered starlight) fluxes.
- Emulation target:
  - Approximate outputs of a deterministic physical radiative transfer solver used to generate the training set.
- Intended use:
  - Fast surrogate for repeated radiative transfer evaluations in atmospheric modeling workflows.

## 2. Core Engineering Principles

- Fail fast:
  - On schema mismatch, missing artifacts, invalid config, invalid values, or unsupported runtime settings, raise immediately.
- No silent fallback behavior:
  - Do not skip missing variables/files.
  - Do not auto-generate missing required artifacts unless explicitly required by this spec.
  - Do not substitute default behavior when config is incomplete.
- No magic numbers in logic:
  - Every scientifically relevant threshold, clamp, precision, and behavior switch must be explicit in config.
  - Non-scientific constants may exist as module-level `UPPER_CASE` constants with a one-line rationale.
- Vectorized and efficient:
  - Prefer batch/chunk tensor and array operations.
  - Avoid Python loops over individual samples when a vectorized implementation exists.
- Prefer proven libraries over bespoke code:
  - Use built-in PyTorch modules and well-maintained utility frameworks when they make the code simpler or more robust.
  - Keep custom implementations only when they are clearly necessary for required behavior or performance.

### 2.1 Environment Exception (Current Setup)

- Runtime compatibility requirement for this setup:
  - The entrypoint sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing PyTorch.
- Rationale:
  - Prevents host-specific MKL/OpenMP duplicate-runtime aborts in the current environment.
- Scope:
  - This is an explicit environment constraint for this project setup, not a general silent-fallback policy.

## 3. Units and Physical Conventions

- Primary unit system target: CGS where possible.
- Current dataset variables are assumed to be correct as currently interpreted by their names and existing usage.
  - Example: `pressure_bar` is treated as bar because that is the current canonical field.
- Unit handling policy:
  - No implicit unit conversion.
  - Any conversion must be explicit, configurable, and auditable.
  - Variable-level unit metadata must be declared in config.
- Current baseline variable semantics from the project manuscript:
  - Pressure range baseline expressed in bar (`pressure_bar`).
  - Temperature baseline expressed in K (`temperature_k`).
  - Flux analysis commonly reported in cgs units (for example erg cm^-2 s^-1).
  - Orbital separation currently carried as configured and must be explicitly unit-tagged.
- Sign/validity constraints:
  - `temperature_k` > 0
  - `pressure_bar` > 0
  - `orbital_distance_m` > 0
  - Flux targets may be positive or negative.
- Domain knowledge constraints:
  - No additional hard physics-informed constraints are required at training time unless explicitly added later.

## 4. Data Contract (HDF5 and Processed Artifacts)

### 4.1 HDF5 Schema

- The existing project HDF5 schema is canonical.
- Required keys must exist exactly as configured.
- Missing files or keys: hard failure.
- Baseline canonical variables for current model:
  - Profile inputs: `pressure_bar`, `temperature_k`
  - Global inputs: `orbital_distance_m` (and future globals as explicitly configured)
  - Targets: `net_thermal_flux`, `net_reflected_flux`
- Required shape conventions:
  - Profile variables: identical shape `[N, L]` across all profile channels for a given file.
  - Global variables: matching leading dimension `N`.
  - No exceptions.
- Non-finite values:
  - Any NaN or Inf in required data is a hard failure.
  - This hard failure applies during every stage, including statistics calculation.

### 4.2 Sequence Length Policy

- `L > max_sequence_length`: hard failure.
- `L < max_sequence_length`: right-pad to `max_sequence_length`.
- Padding convention:
  - Numeric sentinel + boolean mask.
  - Mask semantics: `True == padding`, `False == valid`.
- Sentinel safety:
  - If sentinel appears naturally in raw (non-padding) data, hard failure.
  - Sentinel comparison uses absolute tolerance only:
    - `abs(x - padding_value) <= padding_comparison_epsilon` means sentinel.
    - Relative tolerance comparisons are disallowed.

### 4.3 Split Files

- Split file is a mandatory runtime artifact.
- If split file is missing, `main.py` auto-generates it from configured raw HDF5 files before preprocessing.
- Auto-generated splits use the compact canonical format (`file_stems`) and are deterministic from `miscellaneous_settings.random_seed`.
- Manual generation via `python src/generate_splits.py` remains supported.
- This is the only allowed auto-generation path for required artifacts.
- If split file is malformed or inconsistent with schema after generation/loading: hard failure.
- Only one canonical split format is allowed:
  - Compressed format using `file_stems` mapping.
- Legacy/alternate split formats are disallowed.
- Baseline split ratios for current scientific runs:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

## 5. Configuration Contract

- Configuration is the single source of truth for runtime behavior.
- Missing config sections/keys/types: hard failure.
- Implicit defaults in code are disallowed for runtime behavior.
- One default execution behavior is defined in config (not via CLI mode switches).
- CLI parser flags for operational mode selection are disallowed.
- Default CLI file/directory paths are resolved from the repository root (the directory containing `src/`, `unit_tests/`, and `config/`), not from process working directory.
- `data/` and `models/` are required project-root sibling directories and must never resolve inside `src/`.
- Hyperparameter and runtime policy values must be range-validated at config load time:
  - Invalid ranges are hard failures before preprocessing/training starts.

### 5.1 Runtime Control Keys (Required)

The following runtime behavior switches must be explicit in config:

- `miscellaneous_settings.rebuild_processed_data`:
  - If `true`, normalization removes existing processed artifacts before rebuilding.
  - If `false` and processed artifacts already exist, reuse is allowed only when strict validation passes:
    - Required processed files/directories and shard counts are valid.
    - A processed fingerprint (config hash + split-file hash + raw-file metadata) matches current inputs.
  - If `false` and reuse validation fails, hard failure with explicit instruction to rebuild.
- `miscellaneous_settings.dataset_loading_mode` in `{auto, ram, disk}`:
  - `ram`: force in-memory loading.
  - `disk`: force disk-cache loading.
  - `auto`: choose deterministically from configured memory policy.
- `miscellaneous_settings.dataset_ram_safety_fraction` in `(0, 1]`:
  - Used by `auto` mode when deciding RAM vs disk.
- `miscellaneous_settings.dataset_max_cached_shards`:
  - Positive integer LRU shard-cache cap.
- `miscellaneous_settings.dataset_large_shard_mmap_bytes`:
  - Positive integer threshold for mmap-vs-load behavior.
- `miscellaneous_settings.dataset_copy_mmap_slices`:
  - Explicit copy policy for mmap-backed sample slices (`true`/`false`).

### 5.2 Precision and Numeric Configuration

The following precision controls must be explicit and configurable:

- Input tensor dtype
- Statistics accumulation dtype
- Model parameter dtype
- Forward-pass compute dtype
- Loss compute dtype
- Optimizer state dtype
- AMP autocast dtype (if AMP enabled)

Behavior:

- Invalid precision combinations: hard failure.
- AMP:
  - Allowed only when appropriate for device/dtype.
  - CUDA-only unless explicitly extended.
  - Unsupported AMP requests must fail immediately.

## 6. Normalization Policy

- Every variable must have an explicit normalization method in config.
- No implicit normalization method defaults.
- Missing method/stats for required variable: hard failure.
- Current baseline normalization choices (must remain explicit in config):
  - `pressure_bar`: `log-min-max`
  - `temperature_k`: `standard` (z-score)
  - `orbital_distance_m`: `log-standard`
  - `net_thermal_flux`: `symlog`
  - `net_reflected_flux`: `symlog`
- Quantile behavior:
  - Approximate quantiles are allowed when chosen for speed/simplicity.
  - Approximation behavior must be explicit in config.
- Clamping:
  - Allowed only with config-defined clamp ranges.
  - Clamp configuration must be explicit.

## 7. Training and Runtime Behavior

- Pipeline behavior:
  - Sequential preprocessing + training is allowed as a single configured behavior.
- Missing required preprocessing artifacts:
  - Hard failure.
- Invalid indices:
  - Hard failure (no dropping/skipping).
- All-padding batch:
  - Hard failure on first occurrence.
- Statistics-pass fail-fast:
  - Any non-finite value or sentinel occurrence in raw statistics inputs must hard-fail immediately on first encounter.
- Compile behavior:
  - If compile is enabled in config and compile fails, hard failure (no eager fallback).
- Export behavior during training/tuning:
  - Disabled.
  - Training outputs checkpoints/logs/metadata only.
- Training objective baseline:
  - Loss is masked mean-squared error on valid (non-padding) target elements.
- Learning-rate policy baseline:
  - Cosine schedule with warmup.

### 7.1 Optimizer/Scheduler Standard

- Supported training policy for current project baseline:
  - Optimizer: AdamW
  - Scheduler: warmup followed by either cosine decay or validation plateau reduction
- Unsupported optimizer/scheduler choices: hard failure unless explicitly added.

### 7.2 Hardware Support

- Allowed runtime backends:
  - CPU
  - MPS
  - CUDA
- Hardware availability is not assumed; behavior must be explicit and deterministic per selected backend.
- The checked-in default backend is CUDA so the default config matches the SLURM launcher.

## 8. Artifact Policy

Required training outputs:

- Model checkpoint(s)
- Training logs
- Training metadata
- `test_metrics.json` from an immediate held-out test-loss pass after fitting

Not produced during training:

- Export artifacts

Post-training (manual workflow):

- Training mode may run a basic held-out test evaluation immediately after fitting by
  reloading the best checkpoint and writing normalized-space summary metrics.
- Export and full scientific evaluation are optional manual steps outside the core training loop.
- The exported model must accept physical-unit inputs and perform normalization/denormalization internally.
- No normalized-space PT2 export artifact is maintained.

## 9. Reproducibility and Performance

- Priority:
  - Performance first, reproducibility second.
- Reproducibility target:
  - Reasonable reproducibility without major slowdown.
  - Full strict determinism is not required.
- Performance expectations:
  - Preprocessing throughput, training speed, and inference speed all matter and should be optimized.
- Baseline performance references from the manuscript (for context, not hard gates):
  - Reported training scale: ~2,000,000 profiles.
  - Reported sequence length baseline: 75 layers.
  - Reported training runtime baseline: order of hours on high-end GPU (hardware-dependent).
  - Reported inference baseline: millisecond-scale per profile, sub-millisecond amortized with batching.

## 10. Static Analysis and Code Hygiene

Use these tools continuously during development:

- `ruff`
- `pyflakes`
- `vulture`

### 10.1 Execution Environment

- All development, preprocessing, training, and verification commands are run in the `nn` environment.
- Running commands outside `nn` is out of spec.

### 10.2 Standard Static-Analysis Checks

Run these checks from the project root in the `nn` environment:

- `ruff check src unit_tests`
- `pyflakes src unit_tests`
- `vulture src unit_tests`

Current policy:

- No strict merge gate is defined yet.
- Tool findings are used to keep code clean, detect dead code, and improve correctness.

### 10.3 Review Scope

- Do not spend time reading or auditing files in `data/` or `models/` unless explicitly asked.
- Default review and maintenance scope is source and automated test code (`src/`, `unit_tests/`, config/spec files).

## 11. Testing Policy

- Automated regression coverage lives in `unit_tests/`.
- Training itself is not blocked by fixed scientific metric thresholds at this stage.
- Continue using current loss-driven training objective unless explicitly revised.
- The configured training pipeline may run a post-fit test pass that records held-out masked
  MSE and the best training epoch in `test_metrics.json`.
- Post-training scientific evaluation should include:
  - Channel-wise absolute and signed percent-error analysis over the held-out test set.
  - Bias checks (signed-error central tendency near zero).
- Manual export/evaluation utilities may exist, but they are not part of the required repository contract.
- Any PT2 inference utility should load the exported program once and reuse `program.module()` for all batches.

## 12. Scientific Baseline Assumptions (Current Manuscript)

- Forward-model source and assumptions to emulate:
  - PICASO-generated flux labels from deterministic radiative transfer post-processing.
  - Current baseline assumes solar-composition hot-Jupiter-like training domain.
  - Current baseline includes thermal and scattered-starlight channels.
- Profile-generation assumptions currently used for training corpus:
  - Modified Line et al. (2013)-style parameterized pressure-temperature profiles.
  - Broad randomized parameter sampling to cover representative hot-Jupiter states.
  - Convective adjustment applied to a configured random subset.
- Current simplified irradiation geometry assumptions:
  - Baseline uses fixed normal-incidence geometry (`mu = 1`) for shortwave setup.
  - Future expansion to multiple incidence angles is expected via explicit additional globals.
- Current chemistry/radiative settings baseline:
  - Premixed correlated-k usage in source forward model.
  - Rayleigh and continuum processes as configured in data-generation workflow.

## 13. Model Architecture Baseline (Current)

- Architecture family:
  - Config-selected sequence model with transformer default and optional LSTM baseline.
- Current baseline structure:
  - Shared layerwise projection and FiLM conditioning with global features.
  - Transformer path: sinusoidal positional encoding + multi-head self-attention encoder stack.
  - LSTM path: bidirectional recurrent stack projected back to the shared latent width.
  - Regression head to layerwise target channels with no final activation clamp.
- Hyperparameters are config-defined and must not be hardcoded in source.
- Current checked-in config baseline (`config/config.jsonc`, as of March 7, 2026):
  - `device_backend = cuda`
  - `model_type = transformer`
  - `d_model = 128`
  - `transformer.nhead = 4`
  - `transformer.num_layers = 3`
  - `transformer.dim_feedforward = 512`
  - `dropout = 0.05`
  - `epochs = 300`
  - `optimizer = AdamW`
  - `scheduler_type = plateau`
  - `learning_rate = 1e-4`
  - `min_lr = 1e-6`
  - `weight_decay = 1e-5`
  - `gradient_clip_val = 2.0`

## 14. Implementation Checklist

Any refactor toward this spec must verify all of the following:

- No implicit defaults for required runtime behavior.
- No parser-flag mode switching for normal operation.
- No legacy split format support.
- Hard-fail handling for missing/invalid data and configuration.
- Config-driven precision end-to-end.
- No export in training path.
- Training path behavior is explicit: fitting may be followed by a basic held-out test-loss pass.
- All-padding batches fail immediately.
- Compile failure cannot silently downgrade behavior.
- Performance-sensitive paths are vectorized and profiled.
