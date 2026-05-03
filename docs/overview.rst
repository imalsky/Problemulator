Overview
========

Problemulator is designed as a scientific machine learning pipeline for
profile-to-profile regression in radiative transfer workflows.

Purpose and scope
-----------------

The repository is scoped around a single scientific task and a strict runtime
contract.

- Inputs are layerwise profile variables such as ``temperature_k`` and
  ``pressure_bar``, plus per-profile globals such as ``orbital_distance_m``.
- Outputs are layerwise radiative flux targets such as ``net_thermal_flux``
  and ``net_reflected_flux``.
- The current scope prioritizes scientific correctness, clean design, high
  performance, and extension to closely related profile-regression tasks.
- Production hardening, backward-compatibility shims, and in-the-loop rollout
  workflows are explicitly out of scope.

Scientific objective
--------------------

The current baseline task is to emulate deterministic radiative transfer
outputs from one-dimensional atmospheric profiles.

- The model learns a mapping from pressure-temperature structure plus global
  context to layerwise thermal and reflected fluxes.
- The emulation target is a deterministic physical radiative transfer solver
  used to generate the training corpus.
- The intended use is fast surrogate evaluation in atmospheric modeling
  workflows that would otherwise require repeated radiative transfer solves.

Repository priorities
---------------------

The repository is organized around a few explicit engineering rules:

- Fail fast on schema mismatch, missing artifacts, invalid configuration,
  invalid values, or unsupported runtime settings.
- Do not silently skip missing variables or files, and do not substitute
  fallback behavior when required configuration is incomplete.
- Keep scientifically relevant thresholds, clamping behavior, and precision
  policy explicit in configuration rather than hidden in source.
- Prefer vectorized implementations and proven library components where they
  simplify the code without weakening correctness.

Environment note
----------------

The current setup includes one explicit runtime exception in the entrypoint:
``KMP_DUPLICATE_LIB_OK=TRUE`` is set before importing PyTorch. This is a
documented environment constraint for the checked-in runtime setup, not a
general fallback policy.
