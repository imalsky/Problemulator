#!/usr/bin/env python3
"""Unit tests for training-loop logging and validation-control behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import ModelTrainer, WarmupScheduler


class _NoOpAfterScheduler:
    """Minimal downstream scheduler used for warmup-only tests."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _MetricScheduler:
    """Metric-accepting scheduler stub for trainer flow control tests."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.metrics: list[float] = []

    def step(self, metric: float | None = None) -> None:
        if metric is None:
            raise AssertionError("MetricScheduler expected a validation metric.")
        self.metrics.append(metric)


class TrainingLoggingTests(unittest.TestCase):
    """Cover epoch-level LR and improvement logging semantics."""

    def test_train_logs_pre_step_warmup_lr_and_blank_first_improvement(self) -> None:
        """Epoch logs should reflect the warmup LR just used, with no bogus first-epoch delta."""
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.cfg = {
            "training_hyperparameters": {
                "epochs": 1,
                "early_stopping_patience": 5,
                "min_delta": 1e-6,
            }
        }
        param = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([param], lr=0.1)
        trainer.scheduler = WarmupScheduler(
            trainer.optimizer,
            warmup_epochs=2,
            warmup_start_factor=0.1,
            after_scheduler=_NoOpAfterScheduler(),
            after_scheduler_requires_metric=False,
        )
        trainer.train_loader = [object()]
        trainer.val_loader = [object()]
        trainer.best_val_loss = float("inf")
        trainer.best_early_stopping_val = float("inf")
        trainer.best_epoch = -1
        trainer.early_stopping_min_delta = 1e-6

        losses = iter((1.25, 0.5))
        trainer._run_epoch = lambda loader, is_train: next(losses)
        trainer._save_best_model = lambda: None

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.log_path = Path(tmpdir) / "training_log.csv"
            trainer.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")

            best_val = ModelTrainer.train(trainer)

            rows = trainer.log_path.read_text().splitlines()

        self.assertAlmostEqual(best_val, 0.5)
        self.assertEqual(trainer.best_epoch, 1)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.055)
        self.assertEqual(len(rows), 2)

        epoch_fields = rows[1].split(",")
        self.assertEqual(epoch_fields[0], "1")
        self.assertEqual(epoch_fields[3], "1.000000e-02")
        self.assertEqual(epoch_fields[5], "")

    def test_plateau_scheduler_drops_lr_after_ten_non_improving_epochs(self) -> None:
        """Plateau scheduling should halve the LR after patience is exceeded."""
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.cfg = {
            "training_hyperparameters": {
                "epochs": 13,
                "early_stopping_patience": 30,
                "min_delta": 1e-6,
            }
        }
        param = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([param], lr=0.1)
        trainer.scheduler = WarmupScheduler(
            trainer.optimizer,
            warmup_epochs=0,
            warmup_start_factor=1.0,
            after_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                threshold=1e-5,
                threshold_mode="abs",
                min_lr=1e-6,
            ),
            after_scheduler_requires_metric=True,
        )
        trainer.train_loader = [object()]
        trainer.val_loader = [object()]
        trainer.best_val_loss = float("inf")
        trainer.best_early_stopping_val = float("inf")
        trainer.best_epoch = -1
        trainer.early_stopping_min_delta = 1e-6

        loss_values: list[float] = []
        for _epoch in range(13):
            loss_values.extend((1.0, 1.0))
        losses = iter(loss_values)
        trainer._run_epoch = lambda loader, is_train: next(losses)
        trainer._save_best_model = lambda: None

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.log_path = Path(tmpdir) / "training_log.csv"
            trainer.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")

            ModelTrainer.train(trainer)

            rows = trainer.log_path.read_text().splitlines()

        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.05)
        epoch_13_fields = rows[13].split(",")
        self.assertEqual(epoch_13_fields[0], "13")
        self.assertEqual(epoch_13_fields[3], "5.000000e-02")

    def test_early_stopping_triggers_after_thirty_non_improving_epochs(self) -> None:
        """The configured patience should stop training after thirty bad validation epochs."""
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.cfg = {
            "training_hyperparameters": {
                "epochs": 40,
                "early_stopping_patience": 30,
                "min_delta": 1e-6,
            }
        }
        param = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([param], lr=0.1)
        trainer.scheduler = _MetricScheduler(trainer.optimizer)
        trainer.train_loader = [object()]
        trainer.val_loader = [object()]
        trainer.best_val_loss = float("inf")
        trainer.best_early_stopping_val = float("inf")
        trainer.best_epoch = -1
        trainer.early_stopping_min_delta = 1e-6

        validation_losses = [1.0] * 31
        loss_values: list[float] = []
        for val_loss in validation_losses:
            loss_values.extend((1.0, val_loss))
        losses = iter(loss_values)
        trainer._run_epoch = lambda loader, is_train: next(losses)
        saved_epochs: list[int] = []
        trainer._save_best_model = lambda: saved_epochs.append(trainer.best_epoch)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.log_path = Path(tmpdir) / "training_log.csv"
            trainer.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")

            best_val = ModelTrainer.train(trainer)

            rows = trainer.log_path.read_text().splitlines()

        self.assertEqual(len(rows), 32)
        self.assertEqual(trainer.best_epoch, 1)
        self.assertEqual(saved_epochs[-1], 1)
        self.assertAlmostEqual(best_val, 1.0)
        self.assertEqual(len(trainer.scheduler.metrics), 31)

    def test_strict_best_checkpoint_updates_can_outrun_early_stop_delta(self) -> None:
        """Strict checkpoint updates should remain independent from the early-stop threshold."""
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.cfg = {
            "training_hyperparameters": {
                "epochs": 10,
                "early_stopping_patience": 2,
                "min_delta": 1e-6,
            }
        }
        param = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([param], lr=0.1)
        trainer.scheduler = _MetricScheduler(trainer.optimizer)
        trainer.train_loader = [object()]
        trainer.val_loader = [object()]
        trainer.best_val_loss = float("inf")
        trainer.best_early_stopping_val = float("inf")
        trainer.best_epoch = -1
        trainer.early_stopping_min_delta = 1e-6

        validation_losses = [1.0, 0.9999995, 0.9999990]
        loss_values: list[float] = []
        for val_loss in validation_losses:
            loss_values.extend((1.0, val_loss))
        losses = iter(loss_values)
        trainer._run_epoch = lambda loader, is_train: next(losses)
        saved_epochs: list[int] = []
        trainer._save_best_model = lambda: saved_epochs.append(trainer.best_epoch)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.log_path = Path(tmpdir) / "training_log.csv"
            trainer.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")

            best_val = ModelTrainer.train(trainer)

        self.assertEqual(trainer.best_epoch, 3)
        self.assertEqual(saved_epochs[-1], 3)
        self.assertAlmostEqual(best_val, validation_losses[-1])


if __name__ == "__main__":
    unittest.main()
