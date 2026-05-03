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

``docs/`` (project root)
------------------------

The top-level ``docs/`` directory contains all project documentation:

- ``docs/spec.md`` — the authoritative implementation spec; non-optional reading
  before any non-trivial change to ``Problemulator/``.
- ``docs/*.rst`` — Sphinx source files for this documentation site.

Build the HTML output locally with::

   sphinx-build docs/ docs/_build/html

The ``docs/_build/`` directory is not tracked in version control.

Review boundaries
-----------------

The default maintenance and review scope is source, tests, config, and spec
files. ``data/`` and ``models/`` are not default audit targets unless a task
explicitly requires them.
