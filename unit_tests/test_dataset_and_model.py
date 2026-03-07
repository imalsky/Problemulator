#!/usr/bin/env python3
"""Unit tests for padding masks and the simplified model contracts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import pad_collate
from model import FiLMLayer, PredictionModel


class DatasetAndModelTests(unittest.TestCase):
    """Cover the padding-mask contract and the dead-code reductions."""

    def test_pad_collate_builds_expected_masks_and_shapes(self) -> None:
        """Sequence and target masks should line up exactly on padded timesteps."""
        padding_value = -9999.0
        padding_epsilon = 1e-6
        batch = [
            (
                {
                    "sequence": np.asarray(
                        [[1.0, 10.0], [padding_value, padding_value]],
                        dtype=np.float32,
                    ),
                    "global_features": np.asarray([0.5], dtype=np.float32),
                },
                np.asarray([[100.0], [padding_value]], dtype=np.float32),
            ),
            (
                {
                    "sequence": np.asarray(
                        [[2.0, 20.0], [3.0, 30.0]],
                        dtype=np.float32,
                    ),
                    "global_features": np.asarray([1.5], dtype=np.float32),
                },
                np.asarray([[200.0], [300.0]], dtype=np.float32),
            ),
        ]

        inputs, masks, targets, target_masks = pad_collate(
            batch,
            padding_value=padding_value,
            padding_epsilon=padding_epsilon,
            tensor_dtype=torch.float32,
        )

        self.assertEqual(tuple(inputs["sequence"].shape), (2, 2, 2))
        self.assertEqual(tuple(inputs["global_features"].shape), (2, 1))
        self.assertEqual(tuple(masks["sequence"].shape), (2, 2))
        self.assertEqual(tuple(targets.shape), (2, 2, 1))
        self.assertEqual(tuple(target_masks.shape), (2, 2))
        self.assertTrue(torch.equal(masks["sequence"], target_masks))
        self.assertTrue(bool(masks["sequence"][0, 1]))
        self.assertFalse(bool(masks["sequence"][1, 1]))

    def test_pad_collate_rejects_all_padding_batch(self) -> None:
        """The fail-fast policy should reject batches with no valid targets."""
        padding_value = -9999.0
        batch = [
            (
                {
                    "sequence": np.full((2, 2), padding_value, dtype=np.float32),
                },
                np.full((2, 1), padding_value, dtype=np.float32),
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "All-padding batch"):
            pad_collate(batch, padding_value=padding_value, padding_epsilon=1e-6)

    def test_film_layer_requires_positive_clamp(self) -> None:
        """The unreachable no-clamp branch was removed; clamp must stay positive."""
        with self.assertRaisesRegex(ValueError, "clamp_gamma must be > 0"):
            FiLMLayer(context_dim=2, feature_dim=4, clamp_gamma=0.0)

    def test_prediction_model_requires_global_features_when_configured(self) -> None:
        """Configured global conditioning must be provided at inference time."""
        model = PredictionModel(
            input_dim=2,
            global_input_dim=1,
            output_dim=1,
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=8,
            dropout=0.0,
            attention_dropout=0.0,
            max_sequence_length=8,
            film_clamp=2.0,
            output_head_divisor=2,
            output_head_dropout_factor=0.5,
        )
        sequence = torch.randn(2, 4, 2)
        sequence_mask = torch.zeros(2, 4, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "expects global_features"):
            model(sequence, global_features=None, sequence_mask=sequence_mask)

    def test_prediction_model_returns_expected_shape(self) -> None:
        """The simplified model path should still produce sequence-shaped outputs."""
        model = PredictionModel(
            input_dim=2,
            global_input_dim=1,
            output_dim=3,
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=8,
            dropout=0.0,
            attention_dropout=0.0,
            max_sequence_length=8,
            film_clamp=2.0,
            output_head_divisor=2,
            output_head_dropout_factor=0.5,
        )
        sequence = torch.randn(2, 4, 2)
        global_features = torch.randn(2, 1)
        sequence_mask = torch.zeros(2, 4, dtype=torch.bool)

        output = model(sequence, global_features=global_features, sequence_mask=sequence_mask)

        self.assertEqual(tuple(output.shape), (2, 4, 3))


if __name__ == "__main__":
    unittest.main()
