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

The checked-in model family supports two sequence-regression backbones with a
shared FiLM-conditioned regression head.

- default baseline: encoder-only transformer
- optional comparison baseline: bidirectional LSTM
- shared latent width and FiLM conditioning with global features
- regression head that outputs layerwise target channels without a final output
  clamp

Model and training defaults
---------------------------

The current checked-in configuration uses the following baseline values:

.. list-table:: Checked-in model baseline
   :header-rows: 1

   * - Setting
     - Value
   * - ``model_hyperparameters.model_type``
     - ``transformer``
   * - ``model_hyperparameters.d_model``
     - ``256``
   * - ``model_hyperparameters.transformer.nhead``
     - ``4``
   * - ``model_hyperparameters.transformer.num_layers``
     - ``4``
   * - ``model_hyperparameters.transformer.dim_feedforward``
     - ``1024``
   * - ``model_hyperparameters.dropout``
     - ``0.0``
   * - ``model_hyperparameters.transformer.attention_dropout``
     - ``0.0``
   * - ``model_hyperparameters.max_sequence_length``
     - ``75``
   * - ``training_hyperparameters.learning_rate``
     - ``1e-4``
   * - ``training_hyperparameters.scheduler_type``
     - ``cosine``
   * - ``training_hyperparameters.weight_decay``
     - ``1e-5``
   * - ``training_hyperparameters.gradient_clip_val``
     - ``2.0``
