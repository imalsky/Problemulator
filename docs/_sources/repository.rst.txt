Repository
==========

The project keeps the core runtime behavior in a small set of top-level
directories.

``src/``
--------

Core pipeline code for preprocessing, dataset handling, hardware setup, model
definition, training, and the main configured execution path.

``config/``
-----------

Checked-in runtime configuration for data paths, normalization policy,
precision settings, model hyperparameters, and training controls.

``unit_tests/``
---------------

Focused tests covering configuration behavior, dataset/model expectations, and
training-loop semantics.
