Configuration
=============

The configuration file is the single source of truth for runtime behavior.
Required keys, valid types, and allowed ranges are enforced before the main
pipeline proceeds.

Configuration contract
----------------------

The repository follows a few strict configuration rules:

- Missing configuration sections, keys, or incompatible types are hard
  failures.
- Runtime behavior is defined in config rather than through operational CLI
  mode switches.
- Default data and model paths resolve from the repository root, not from the
  process working directory.
- ``data/`` and ``models/`` must remain sibling directories of ``src/`` and
  must not resolve inside ``src/``.

Current runtime defaults
------------------------

The checked-in ``config/transformer.jsonc`` currently uses these baseline runtime
settings. A comparable bidirectional LSTM baseline is also provided in
``config/lstm.jsonc``.

.. list-table:: Runtime and training defaults
   :header-rows: 1

   * - Setting
     - Value
   * - ``miscellaneous_settings.device_backend``
     - ``cuda``
   * - ``miscellaneous_settings.execution_mode``
     - ``train``
   * - ``miscellaneous_settings.rebuild_processed_data``
     - ``false``
   * - ``miscellaneous_settings.dataset_loading_mode``
     - ``auto``
   * - ``miscellaneous_settings.random_seed``
     - ``42``
   * - ``training_hyperparameters.epochs``
     - ``300``
   * - ``training_hyperparameters.batch_size``
     - ``1024``
   * - ``training_hyperparameters.optimizer``
     - ``adamw``
   * - ``training_hyperparameters.scheduler_type``
     - ``cosine``
   * - ``training_hyperparameters.learning_rate``
     - ``1e-4``
   * - ``model_hyperparameters.model_type``
     - ``transformer``
   * - ``model_hyperparameters.dropout``
     - ``0.0``

Architecture-specific keys
--------------------------

The checked-in config uses one shared model section plus architecture-specific
subsections.

- Shared model keys include ``model_type``, ``d_model``, ``dropout``,
  ``film_clamp``, ``max_sequence_length``, ``output_head_divisor``, and
  ``output_head_dropout_factor``.
- Transformer runs use ``model_hyperparameters.transformer`` with
  ``nhead``, ``num_layers``, ``dim_feedforward``, and
  ``attention_dropout``.
- LSTM runs use ``model_hyperparameters.lstm`` with ``num_layers`` and
  ``bidirectional``.
- Only the active architecture subsection is required at validation time.

Runtime control keys
--------------------

Several runtime switches are explicitly required by the spec:

- ``rebuild_processed_data`` controls whether processed artifacts are removed
  and regenerated before normalization.
- ``dataset_loading_mode`` chooses among ``auto``, ``ram``, and ``disk``.
- ``dataset_ram_safety_fraction`` controls the ``auto`` RAM-vs-disk decision.
- ``dataset_max_cached_shards`` sets the LRU shard cache cap.
- ``dataset_large_shard_mmap_bytes`` controls mmap-vs-load behavior.
- ``dataset_copy_mmap_slices`` makes the mmap slice copy policy explicit.

Precision policy
----------------

Precision behavior is also configuration-driven end to end.

- Input, statistics, model, forward, loss, optimizer-state, and AMP dtypes are
  all explicit configuration fields.
- Invalid precision combinations are hard failures.
- AMP is only allowed when the selected backend and dtype support it.
- Unsupported AMP requests fail immediately rather than downgrading behavior.

Normalization policy
--------------------

Every required variable must declare an explicit normalization method.

.. list-table:: Checked-in normalization methods
   :header-rows: 1

   * - Variable
     - Method
   * - ``pressure_bar``
     - ``log-min-max``
   * - ``temperature_k``
     - ``standard``
   * - ``orbital_distance_m``
     - ``log-standard``
   * - ``net_thermal_flux``
     - ``symlog``
   * - ``net_reflected_flux``
     - ``symlog``

Additional normalization rules from the spec:

- Missing normalization methods or required stats are hard failures.
- Approximate quantiles are allowed only when that behavior is explicit.
- Clamping is allowed only through explicit config-defined ranges.
