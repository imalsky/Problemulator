Problemulator documentation
===========================

Problemulator is a scientific emulator for one-dimensional radiative transfer
profiles. The project learns a mapping from atmospheric state inputs to
layerwise radiative flux outputs, with the repository centered on explicit
configuration, fail-fast validation, and reproducible training behavior.

This documentation summarizes the checked-in repository contracts from
``spec.md`` and the active defaults in ``config/transformer.jsonc``. The spec file
remains the more exhaustive source for implementation requirements.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   data_contract
   configuration
   workflow
   scientific_baseline
   repository
   development
   getting_started
