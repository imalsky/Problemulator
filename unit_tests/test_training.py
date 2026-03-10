#!/usr/bin/env python3
"""Unit tests for training-loop logging behavior."""

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

from train import ModelTrainer


class _StepScheduler:
    """Minimal scheduler stub that mutates the optimizer LR on step."""

    def __init__(self, optimizer: torch.optim.Optimizer, next_lr: float) -> None:
        self.optimizer = optimizer
        self.next_lr = next_lr
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        self.optimizer.param_groups[0]["lr"] = self.next_lr


class TrainingLoggingTests(unittest.TestCase):
    """Cover epoch-level LR and improvement logging semantics."""

    def test_train_logs_pre_step_lr_and_blank_first_improvement(self) -> None:
        """Epoch logs should reflect the LR just used, with no bogus first-epoch delta."""
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer.cfg = {
            "training_hyperparameters": {
                "epochs": 1,
                "early_stopping_patience": 5,
                "min_delta": 1e-9,
            }
        }
        param = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([param], lr=0.1)
        trainer.scheduler = _StepScheduler(trainer.optimizer, next_lr=0.01)
        trainer.train_loader = [object()]
        trainer.val_loader = [object()]
        trainer.best_val_loss = float("inf")
        trainer.best_epoch = -1

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
        self.assertEqual(trainer.scheduler.steps, 1)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(len(rows), 2)

        epoch_fields = rows[1].split(",")
        self.assertEqual(epoch_fields[0], "1")
        self.assertEqual(epoch_fields[3], "1.000000e-01")
        self.assertEqual(epoch_fields[5], "")


if __name__ == "__main__":
    unittest.main()
