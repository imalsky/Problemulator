Data Contract
=============

The repository treats data schema and processed artifacts as part of the core
runtime contract. Missing files, missing keys, invalid values, or unexpected
shapes are hard failures.

Units and physical conventions
------------------------------

The baseline unit system is CGS where practical, with variable semantics
anchored to the current dataset naming and checked-in configuration.

.. list-table:: Canonical baseline variables
   :header-rows: 1

   * - Variable
     - Role
     - Unit
     - Constraint
   * - ``pressure_bar``
     - Profile input
     - ``bar``
     - Strictly positive
   * - ``temperature_k``
     - Profile input
     - ``K``
     - Strictly positive
   * - ``orbital_distance_m``
     - Global input
     - ``m``
     - Strictly positive
   * - ``net_thermal_flux``
     - Layerwise target
     - ``erg cm^-2 s^-1``
     - Signed values allowed
   * - ``net_reflected_flux``
     - Layerwise target
     - ``erg cm^-2 s^-1``
     - Signed values allowed

Unit handling rules are explicit:

- No implicit unit conversion is allowed.
- Any conversion must be explicit, configurable, and auditable.
- Variable-level unit metadata must exist in configuration.
- No additional hard physics-informed training constraints are applied unless
  they are explicitly added later.

HDF5 schema
-----------

The current HDF5 schema is the canonical input format.

- Required keys must exist exactly as configured.
- Missing files or keys are hard failures.
- Profile variables for a given file must share a common shape ``[N, L]``.
- Global variables must share the same leading sample dimension ``N``.
- NaN or Inf values in required data are hard failures during every stage,
  including statistics collection.

Sequence length and padding
---------------------------

Sequence handling is explicit and deterministic.

- If ``L > max_sequence_length``, preprocessing fails immediately.
- If ``L < max_sequence_length``, sequences are right-padded up to the maximum.
- Padding uses a numeric sentinel together with a boolean mask.
- Mask semantics are ``True == padding`` and ``False == valid``.
- If the sentinel value appears naturally in non-padding data, preprocessing
  fails immediately.
- Sentinel checks use absolute tolerance only through
  ``padding_comparison_epsilon``.

Split files
-----------

The split file is a required runtime artifact.

- If the configured split file is missing, ``src/main.py`` auto-generates it
  before preprocessing using the configured raw HDF5 files.
- Auto-generated splits are deterministic from
  ``miscellaneous_settings.random_seed``.
- Manual generation through ``python src/generate_splits.py`` remains supported.
- Only the compact canonical ``file_stems`` format is allowed.
- Malformed split files or alternate legacy formats are hard failures.

The checked-in baseline split fractions are:

- Train: 70%
- Validation: 15%
- Test: 15%
