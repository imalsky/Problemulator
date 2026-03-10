Repository
==========

The project keeps the core runtime behavior in a small set of top-level
directories.

``src/``
--------

Core pipeline code for preprocessing, dataset handling, hardware setup, model
definition, training, and the main configured execution path.

Key entrypoints include ``src/main.py`` for the configured pipeline and
``src/generate_splits.py`` for explicit split generation.

``config/``
-----------

Checked-in runtime configuration for data paths, normalization policy,
precision settings, model hyperparameters, and training controls.

``unit_tests/``
---------------

Focused tests covering configuration behavior, dataset/model expectations, and
training-loop semantics.

``data/`` and ``models/``
-------------------------

These directories are part of the default project-root layout for raw data,
processed artifacts, and trained outputs. They must remain sibling directories
of ``src/`` rather than resolving inside source code directories.

``docs_src/`` and ``docs/``
---------------------------

``docs_src/`` contains the Sphinx source files for the documentation site.
``docs/`` contains the generated HTML output that GitHub Pages serves.

Review boundaries
-----------------

The default maintenance and review scope is source, tests, config, and spec
files. ``data/`` and ``models/`` are not default audit targets unless a task
explicitly requires them.
