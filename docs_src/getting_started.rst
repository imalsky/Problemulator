Getting Started
===============

The default configured entrypoint for the repository is ``src/main.py`` and the
default execution mode is training.

Default command
---------------

.. code-block:: bash

   python src/main.py

Default configuration
---------------------

By default, the pipeline reads project settings from
``config/transformer.jsonc``. The repository also expects project-root directories
such as ``data/``, ``models/``, and ``unit_tests/`` to remain alongside
``src/`` and ``config/``.

Command-line path arguments
---------------------------

The main entrypoint accepts path overrides for repository-local resources:

.. code-block:: text

   --config
   --data-dir
   --models-dir

These arguments override file locations, not the operational pipeline mode.

Expected startup behavior
-------------------------

When ``python src/main.py`` starts with the checked-in defaults, it:

- resolves paths from the repository root
- validates configured raw HDF5 files
- ensures the split file exists, auto-generating it if necessary
- reuses processed artifacts only when validation and fingerprint checks pass
- runs the configured preprocessing and training path

Launcher note
-------------

The checked-in ``run.sh`` file is the SLURM-oriented launcher and is configured
around the default CUDA backend and the ``nn`` environment.
