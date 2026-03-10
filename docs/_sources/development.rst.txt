Development
===========

The repository spec defines a narrow development environment and a small set of
standard checks.

Execution environment
---------------------

All development, preprocessing, training, and verification commands are
expected to run in the ``nn`` environment. Running outside that environment is
documented as out of spec.

Static analysis
---------------

The standard static-analysis commands are run from the project root:

.. code-block:: bash

   ruff check src unit_tests
   pyflakes src unit_tests
   vulture src unit_tests

These checks are used to keep the codebase clean and to surface correctness and
dead-code issues, even though no strict merge gate is defined yet.

Testing policy
--------------

Automated regression coverage lives in ``unit_tests/``.

- Training is not blocked by fixed scientific metric thresholds at this stage.
- The baseline objective remains loss-driven masked MSE unless explicitly
  revised.
- A post-fit test pass may record held-out masked MSE and the best epoch in
  ``test_metrics.json``.
- Broader scientific evaluation remains a manual workflow outside the required
  repository contract.

Implementation checklist
------------------------

Any refactor toward the current spec should preserve the following behavior:

- no implicit defaults for required runtime behavior
- no legacy split format support
- hard-fail handling for missing or invalid data and configuration
- config-driven precision policy end to end
- no export path during training
- immediate failure on all-padding batches
- no silent compile fallback when compile is explicitly enabled
- vectorized, performance-sensitive preprocessing and training paths
