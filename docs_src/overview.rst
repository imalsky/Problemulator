Overview
========

Problemulator is designed as a scientific machine learning pipeline for
profile-to-profile regression in radiative transfer workflows.

Current modeling objective
--------------------------

The current baseline task is to emulate deterministic radiative transfer
outputs from one-dimensional atmospheric profiles.

- Inputs include profile variables such as ``pressure_bar`` and
  ``temperature_k``.
- Global context can also be included, such as ``orbital_distance_m``.
- Outputs are layerwise flux targets including ``net_thermal_flux`` and
  ``net_reflected_flux``.

Repository priorities
---------------------

The repository is organized around a few explicit engineering rules:

- Scientific correctness over convenience behavior.
- Fail-fast validation when schema or runtime settings are invalid.
- Explicit configuration for normalization, precision, and training policy.
- Reproducible preprocessing and training behavior driven by checked-in config.
