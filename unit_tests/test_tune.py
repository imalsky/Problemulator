#!/usr/bin/env python3
"""Unit tests for Optuna tuner control flow."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tune  # noqa: E402
from train import NonFiniteTensorError  # noqa: E402
from utils import load_config  # noqa: E402


class _FakeTrial:
    """Minimal trial object for deterministic model-family selection tests."""

    def __init__(self, number: int) -> None:
        self.number = number


class _ModelNonFiniteTrainer:
    """Trainer stub that simulates an unstable sampled architecture."""

    model = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def train(self) -> float:
        raise NonFiniteTensorError(
            label="model predictions",
            mode="training",
            batch_idx=2,
            bad_count=38400,
        )


class _DataNonFiniteTrainer:
    """Trainer stub that simulates invalid processed data reaching training."""

    model = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def train(self) -> float:
        raise NonFiniteTensorError(
            label="targets",
            mode="training",
            batch_idx=1,
            bad_count=1,
        )


class TuneControlFlowTests(unittest.TestCase):
    """Cover tuner model-family and failure-classification behavior."""

    def test_choose_model_type_alternates_only_for_explicit_both(self) -> None:
        self.assertEqual(tune._choose_model_type(_FakeTrial(0), "both"), "transformer")
        self.assertEqual(tune._choose_model_type(_FakeTrial(1), "both"), "lstm")
        self.assertEqual(tune._choose_model_type(_FakeTrial(7), "transformer"), "transformer")

        with self.assertRaisesRegex(ValueError, "resolved before sampling"):
            tune._choose_model_type(_FakeTrial(0), "config")

    def test_choose_model_type_sequential_split(self) -> None:
        # First half (0..n_trials_lstm-1) -> lstm, rest -> transformer.
        self.assertEqual(
            tune._choose_model_type(_FakeTrial(0), "sequential", n_trials_lstm=32),
            "lstm",
        )
        self.assertEqual(
            tune._choose_model_type(_FakeTrial(31), "sequential", n_trials_lstm=32),
            "lstm",
        )
        self.assertEqual(
            tune._choose_model_type(_FakeTrial(32), "sequential", n_trials_lstm=32),
            "transformer",
        )
        self.assertEqual(
            tune._choose_model_type(_FakeTrial(63), "sequential", n_trials_lstm=32),
            "transformer",
        )

    def test_choose_model_type_sequential_requires_n_trials_lstm(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_trials_lstm"):
            tune._choose_model_type(_FakeTrial(0), "sequential")

    def test_build_study_sequential_uses_nop_pruner(self) -> None:
        args = argparse.Namespace(
            sampler_seed=42,
            model_family="sequential",
            study_name="unit_test_sequential",
            storage=None,
            prune_warmup_epochs=10,
        )
        study = tune._build_study(args)
        self.assertIsInstance(study.pruner, optuna.pruners.NopPruner)

    def test_objective_prunes_model_side_nonfinite_trials(self) -> None:
        original_trainer = tune.TunerCallbackTrainer
        tune.TunerCallbackTrainer = _ModelNonFiniteTrainer
        try:
            objective, study = self._build_objective()
            trial = study.ask()
            with self.assertRaises(optuna.TrialPruned):
                objective(trial)
        finally:
            tune.TunerCallbackTrainer = original_trainer

    def test_objective_does_not_prune_data_nonfinite_failures(self) -> None:
        original_trainer = tune.TunerCallbackTrainer
        tune.TunerCallbackTrainer = _DataNonFiniteTrainer
        try:
            objective, study = self._build_objective()
            trial = study.ask()
            with self.assertRaisesRegex(NonFiniteTensorError, "targets"):
                objective(trial)
        finally:
            tune.TunerCallbackTrainer = original_trainer

    def _build_objective(self) -> tuple[Any, optuna.Study]:
        base_config = load_config(PROJECT_ROOT / "config" / "transformer_v2.jsonc")
        args = argparse.Namespace(
            epochs=2,
            data_fraction=0.1,
            patience=1,
            warmup_epochs=1,
            num_workers=0,
            sampler_seed=42,
            model_family="lstm",
        )
        study = optuna.create_study(direction="minimize")
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        models_root = Path(tmpdir.name)

        objective = tune._build_objective(
            base_config=base_config,
            device=torch.device("cpu"),
            processed_dir=PROJECT_ROOT / "data" / "processed",
            splits={"train": [], "val": [], "test": []},
            collate_fn=lambda batch: batch,
            models_root=models_root,
            study_name="unit_test",
            args=args,
        )
        return objective, study


if __name__ == "__main__":
    unittest.main()
