#!/usr/bin/env python3
"""Unit tests for the normalized-MSE plus fractional-L1 loss path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loss import AdaptiveSignedLogLoss  # noqa: E402
from utils import load_config, validate_config  # noqa: E402


def _symlog_stats(threshold: float = 2.0, scale_factor: float = 1.0) -> Dict[str, Any]:
    return {
        "method": "symlog",
        "threshold": threshold,
        "scale_factor": scale_factor,
    }


def _default_loss_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "type": "signed_log_adaptive",
        "lambda_z": 0.5,
        "lambda_phys": 0.5,
        "fractional_epsilon": 4.0,
        "p_norm": 1,
        # Legacy fields may remain in old configs but are ignored by the loss.
        "weight_mode": "range",
        "weight_power": 0.5,
        "w_min": 0.5,
        "w_max": 2.0,
    }
    cfg.update(overrides)
    return cfg


class AdaptiveSignedLogLossForwardTests(unittest.TestCase):
    """Forward-pass shape, value, and stability checks."""

    def _make_two_channel_loss(self, **kwargs: Any) -> AdaptiveSignedLogLoss:
        cfg = _default_loss_cfg(**kwargs)
        target_methods = ["symlog", "symlog"]
        target_stats = [_symlog_stats(), _symlog_stats()]
        return AdaptiveSignedLogLoss(
            loss_cfg=cfg,
            target_methods=target_methods,
            target_stats=target_stats,
        )

    def test_returns_unreduced_BTC_shape(self) -> None:
        loss = self._make_two_channel_loss()
        pred = torch.zeros(2, 5, 2)
        targ = torch.zeros(2, 5, 2)
        out = loss(pred, targ)
        self.assertEqual(tuple(out.shape), (2, 5, 2))

    def test_lambda_phys_zero_is_exact_normalized_mse_fast_path(self) -> None:
        loss = AdaptiveSignedLogLoss(
            loss_cfg=_default_loss_cfg(lambda_z=1.25, lambda_phys=0.0),
            target_methods=["unsupported-method"],
            target_stats=[{}],
        )
        pred = torch.tensor([[[1.0], [3.0]]])
        targ = torch.tensor([[[0.5], [1.0]]])

        out = loss(pred, targ)
        expected = 1.25 * (pred - targ).pow(2)

        torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)

    def test_zero_residual_gives_zero_loss(self) -> None:
        loss = self._make_two_channel_loss()
        pred = torch.randn(3, 4, 2)
        targ = pred.clone()
        out = loss(pred, targ)
        self.assertAlmostEqual(float(out.abs().sum().item()), 0.0, places=6)

    def test_fractional_l1_uses_denormalized_physical_values(self) -> None:
        loss = self._make_two_channel_loss(lambda_z=0.0, lambda_phys=1.0)
        # With symlog threshold=2 and scale_factor=1, z values in [-1, 1]
        # denormalize linearly: phys = 2 * z.
        pred = torch.tensor([[[0.5, 1.0]]])
        targ = torch.tensor([[[0.0, 0.5]]])

        out = loss(pred, targ)
        expected = torch.tensor([[[0.5, 1.0 / (5.0**0.5)]]])

        torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-6)

    def test_finite_for_zero_crossing_targets(self) -> None:
        loss = self._make_two_channel_loss()
        targ = torch.zeros(2, 3, 2)
        pred = torch.tensor(
            [
                [[1e-3, -1e-3], [10.0, -10.0], [0.0, 0.0]],
                [[-1e-3, 1e-3], [-10.0, 10.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        out = loss(pred, targ)
        self.assertTrue(torch.isfinite(out).all())

    def test_padding_mask_excludes_padded_positions(self) -> None:
        # Compute the masked reduction the way ModelTrainer._run_epoch does
        # and confirm it matches a hand reduction over only non-padding
        # positions.
        loss = self._make_two_channel_loss()
        pred = torch.randn(1, 4, 2)
        targ = torch.randn(1, 4, 2)
        target_masks = torch.tensor([[False, False, True, True]])  # last 2 padded

        unreduced = loss(pred, targ)
        valid_steps = ~target_masks
        valid_step_mask = valid_steps.unsqueeze(-1)
        masked_sum = unreduced.masked_fill(~valid_step_mask, 0).sum()
        num_valid = valid_steps.sum(dtype=torch.int64) * unreduced.shape[-1]
        masked_mean = masked_sum / num_valid.clamp_min(1)

        # Reference: only the first two timesteps contribute.
        unreduced_only_valid = loss(pred[:, :2], targ[:, :2])
        ref_mean = unreduced_only_valid.mean()

        torch.testing.assert_close(masked_mean, ref_mean, rtol=0.0, atol=1e-6)


class AdaptiveSignedLogLossValidationTests(unittest.TestCase):
    """Constructor-time validation of the loss config."""

    _STATS = [_symlog_stats()]
    _METHODS = ["symlog"]

    def _construct(self, cfg: Dict[str, Any]) -> AdaptiveSignedLogLoss:
        return AdaptiveSignedLogLoss(
            loss_cfg=cfg, target_methods=self._METHODS, target_stats=self._STATS
        )

    def test_missing_required_key_raises(self) -> None:
        cfg = _default_loss_cfg()
        del cfg["lambda_z"]
        with self.assertRaisesRegex(KeyError, "lambda_z"):
            self._construct(cfg)

    def test_both_lambdas_zero_raises(self) -> None:
        cfg = _default_loss_cfg(lambda_z=0.0, lambda_phys=0.0)
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            self._construct(cfg)

    def test_missing_fractional_epsilon_raises(self) -> None:
        cfg = _default_loss_cfg()
        del cfg["fractional_epsilon"]
        with self.assertRaisesRegex(KeyError, "fractional_epsilon"):
            self._construct(cfg)

    def test_invalid_fractional_epsilon_raises(self) -> None:
        cfg = _default_loss_cfg(fractional_epsilon=0.0)
        with self.assertRaisesRegex(ValueError, "fractional_epsilon"):
            self._construct(cfg)

    def test_invalid_p_norm_raises(self) -> None:
        cfg = _default_loss_cfg(p_norm=2)
        with self.assertRaisesRegex(ValueError, "p_norm"):
            self._construct(cfg)


class LossDispatchValidatorTests(unittest.TestCase):
    """validate_config dispatch on loss.type."""

    @staticmethod
    def _base_cfg() -> Dict[str, Any]:
        return load_config(PROJECT_ROOT / "config" / "transformer_main_v3.jsonc")

    def test_default_signed_log_adaptive_config_validates(self) -> None:
        cfg = self._base_cfg()
        # Should not raise.
        validate_config(cfg)
        self.assertEqual(cfg["loss"]["type"], "signed_log_adaptive")

    def test_missing_loss_type_raises(self) -> None:
        cfg = self._base_cfg()
        del cfg["loss"]["type"]
        with self.assertRaisesRegex(ValueError, "loss.type"):
            validate_config(cfg)

    def test_unknown_loss_type_raises(self) -> None:
        cfg = self._base_cfg()
        cfg["loss"]["type"] = "bogus"
        with self.assertRaisesRegex(ValueError, "Unknown loss.type"):
            validate_config(cfg)

    def test_legacy_hybrid_fractional_now_rejected(self) -> None:
        cfg = self._base_cfg()
        cfg["loss"] = {
            "type": "hybrid_fractional",
            "lambda_mse": 0.5,
            "lambda_frac": 0.5,
            "fractional_epsilon": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "Unknown loss.type"):
            validate_config(cfg)

    def test_missing_fractional_epsilon_rejected(self) -> None:
        cfg = self._base_cfg()
        del cfg["loss"]["fractional_epsilon"]
        with self.assertRaisesRegex(ValueError, "fractional_epsilon"):
            validate_config(cfg)

    def test_nonpositive_fractional_epsilon_rejected(self) -> None:
        cfg = self._base_cfg()
        cfg["loss"]["fractional_epsilon"] = 0.0
        with self.assertRaisesRegex(ValueError, "fractional_epsilon"):
            validate_config(cfg)

    def test_p_norm_two_rejected(self) -> None:
        cfg = self._base_cfg()
        cfg["loss"]["p_norm"] = 2
        with self.assertRaisesRegex(ValueError, "p_norm"):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
