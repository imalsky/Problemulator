Scientific Baseline
===================

This page summarizes the current manuscript-aligned scientific assumptions and
the checked-in model baseline.

Forward-model assumptions
-------------------------

The training labels are based on deterministic PICASO radiative transfer
post-processing under the current hot-Jupiter-like training domain.

- The baseline target includes both thermal and scattered-starlight channels.
- The current domain assumes solar-composition, hot-Jupiter-like atmospheres.
- The docs here summarize the current baseline rather than a general-purpose
  forward-model interface.

Profile-generation assumptions
------------------------------

The manuscript baseline assumes:

- modified Line et al. (2013)-style parameterized pressure-temperature profiles
- broad randomized parameter sampling across representative hot-Jupiter states
- convective adjustment applied to a configured random subset

The simplified shortwave geometry currently assumes fixed normal incidence with
``mu = 1``. Future expansion to multiple incidence angles is expected through
additional explicit global features rather than hidden mode switches.

Chemistry and radiative settings
--------------------------------

The current baseline uses premixed correlated-k settings in the source forward
model, together with Rayleigh and continuum processes as configured in the
data-generation workflow.

Architecture baseline
---------------------

The checked-in model family is an encoder-only transformer for sequence
regression.

- layerwise token embedding
- sinusoidal positional encoding
- multi-head self-attention encoder stack
- FiLM conditioning with global features
- regression head that outputs layerwise target channels without a final output
  clamp

Model and training defaults
---------------------------

The current checked-in configuration uses the following baseline values:

.. list-table:: Checked-in model baseline
   :header-rows: 1

   * - Setting
     - Value
   * - ``model_hyperparameters.d_model``
     - ``128``
   * - ``model_hyperparameters.nhead``
     - ``4``
   * - ``model_hyperparameters.num_encoder_layers``
     - ``3``
   * - ``model_hyperparameters.dim_feedforward``
     - ``512``
   * - ``model_hyperparameters.dropout``
     - ``0.0``
   * - ``model_hyperparameters.attention_dropout``
     - ``0.0``
   * - ``model_hyperparameters.max_sequence_length``
     - ``100``
   * - ``training_hyperparameters.learning_rate``
     - ``1e-4``
   * - ``training_hyperparameters.weight_decay``
     - ``1e-5``
   * - ``training_hyperparameters.gradient_clip_val``
     - ``2.0``
