#!/usr/bin/env python3
"""Unit tests for the signed-log adaptive loss and the loss.type dispatch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loss import AdaptiveSignedLogLoss, _signed_log10  # noqa: E402
from normalizer import DataNormalizer  # noqa: E402
from utils import load_config, validate_config  # noqa: E402


def _signed_log_stats(mean: float, std: float) -> Dict[str, Any]:
    return {"method": "signed-log", "mean": mean, "std": std}


def _default_loss_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "type": "signed_log_adaptive",
        "lambda_z": 0.5,
        "lambda_phys": 0.5,
        "weight_mode": "range",
        "weight_power": 0.5,
        "w_min": 0.5,
        "w_max": 2.0,
        "p_norm": 1,
    }
    cfg.update(overrides)
    return cfg


class SignedLog10Tests(unittest.TestCase):
    """Mathematical sanity checks on the signed-log10 helper."""

    def test_zero_safety_value_and_finite_gradient(self) -> None:
        x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        y = _signed_log10(x)
        # Reference values: signed_log10(-1)=-log10(2), signed_log10(0)=0,
        # signed_log10(1)=log10(2).
        self.assertAlmostEqual(float(y[0].item()), -float(torch.log10(torch.tensor(2.0))), places=5)
        self.assertAlmostEqual(float(y[1].item()), 0.0, places=7)
        self.assertAlmostEqual(float(y[2].item()), float(torch.log10(torch.tensor(2.0))), places=5)

        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sign_preservation_and_monotonicity(self) -> None:
        x = torch.tensor([-100.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 100.0])
        y = _signed_log10(x)
        # Signs match.
        self.assertTrue(torch.all(torch.sign(x) == torch.sign(y)))
        # Strictly monotone in x.
        self.assertTrue(torch.all(y[1:] - y[:-1] > 0))


class SignedLogRoundTripTests(unittest.TestCase):
    """The fast-path identity is exact for signed-log normalized channels."""

    def test_signed_log10_of_denormalized_equals_z_times_std_plus_mean(self) -> None:
        stats = _signed_log_stats(mean=0.5, std=1.2)
        z = torch.linspace(-3.0, 3.0, steps=11)

        denormed = DataNormalizer.denormalize_tensor(z, "signed-log", stats)
        recovered = _signed_log10(denormed)
        expected = z * stats["std"] + stats["mean"]

        torch.testing.assert_close(recovered, expected, rtol=0.0, atol=1e-5)


class AdaptiveSignedLogLossWeightTests(unittest.TestCase):
    """Per-channel weight construction."""

    def _make_loss(
        self,
        stds: List[float],
        weight_mode: str = "range",
        weight_power: float = 0.5,
        w_min: float = 0.5,
        w_max: float = 2.0,
    ) -> AdaptiveSignedLogLoss:
        target_methods = ["signed-log"] * len(stds)
        target_stats = [_signed_log_stats(0.0, s) for s in stds]
        cfg = _default_loss_cfg(
            weight_mode=weight_mode,
            weight_power=weight_power,
            w_min=w_min,
            w_max=w_max,
        )
        return AdaptiveSignedLogLoss(
            loss_cfg=cfg,
            target_methods=target_methods,
            target_stats=target_stats,
        )

    def test_uniform_weights_are_ones(self) -> None:
        loss = self._make_loss([0.1, 1.0, 10.0], weight_mode="uniform")
        torch.testing.assert_close(
            loss.channel_weights, torch.ones(3, dtype=torch.float32), rtol=0.0, atol=0.0
        )

    def test_range_weights_mean_one_and_dispersion_bounded(self) -> None:
        # The clamp constrains dispersion before the final renormalization, so
        # max/min ratio must respect w_max/w_min. The renormalization restores
        # mean=1 (so the phys-term magnitude is independent of the bounds).
        w_min, w_max = 0.5, 2.0
        loss = self._make_loss([0.1, 1.0, 10.0], w_min=w_min, w_max=w_max)
        weights = loss.channel_weights
        self.assertEqual(weights.numel(), 3)
        self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=5)
        ratio = float(weights.max().item() / weights.min().item())
        self.assertLessEqual(ratio, w_max / w_min + 1e-6)

    def test_range_weights_within_bounds_when_unclamped(self) -> None:
        # When the raw weights stay inside [w_min, w_max] after mean
        # normalization, the post-clamp renormalization is the identity, so
        # the final weights satisfy the literal bounds.
        loss = self._make_loss([1.0, 4.0])  # ratio sqrt(4) = 2 within [0.5, 2.0]
        weights = loss.channel_weights
        self.assertTrue(torch.all(weights >= 0.5 - 1e-6))
        self.assertTrue(torch.all(weights <= 2.0 + 1e-6))
        self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=5)

    def test_range_weights_two_channel_ratio_matches_sqrt_of_std_ratio(self) -> None:
        # Two channels with std=1 and std=4 under weight_power=0.5: raw w =
        # [1, 2], mean-normalized to [2/3, 4/3], both within [0.5, 2.0] so
        # clamping is a no-op and the renormalization preserves them. The
        # ratio between the two weights should equal sqrt(4) = 2.
        loss = self._make_loss([1.0, 4.0])
        weights = loss.channel_weights
        ratio = float(weights[1].item() / weights[0].item())
        self.assertAlmostEqual(ratio, 2.0, places=5)


class AdaptiveSignedLogLossForwardTests(unittest.TestCase):
    """Forward-pass shape, value, and stability checks."""

    def _make_two_channel_loss(self, **kwargs: Any) -> AdaptiveSignedLogLoss:
        cfg = _default_loss_cfg(**kwargs)
        target_methods = ["signed-log", "signed-log"]
        target_stats = [
            _signed_log_stats(0.0, 1.0),
            _signed_log_stats(0.0, 1.0),
        ]
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

    def test_zero_residual_gives_zero_loss(self) -> None:
        loss = self._make_two_channel_loss()
        pred = torch.randn(3, 4, 2)
        targ = pred.clone()
        out = loss(pred, targ)
        self.assertAlmostEqual(float(out.abs().sum().item()), 0.0, places=6)

    def test_finite_for_zero_crossing_targets(self) -> None:
        # Targets near zero, predictions far off in signed-log space.
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

    def test_p_norm_one_vs_two_differ_predictably(self) -> None:
        cfg_kwargs = {"weight_mode": "uniform"}  # eliminate weighting noise
        loss_l1 = self._make_two_channel_loss(p_norm=1, **cfg_kwargs)
        loss_l2 = self._make_two_channel_loss(p_norm=2, **cfg_kwargs)
        pred = torch.tensor([[[1.0, 0.0]]])
        targ = torch.tensor([[[0.0, 0.0]]])
        v_l1 = float(loss_l1(pred, targ).sum().item())
        v_l2 = float(loss_l2(pred, targ).sum().item())
        # MSE(z) contribution is identical in both. The phys-term is |delta|
        # vs delta^2, where |delta| = std=1 here, so they happen to agree;
        # use a non-unit value to make them differ.
        pred2 = torch.tensor([[[2.0, 0.0]]])
        v_l1b = float(loss_l1(pred2, targ).sum().item())
        v_l2b = float(loss_l2(pred2, targ).sum().item())
        # delta_phys = (pred_z - targ_z) * std = 2.0 for channel 0.
        # phys term: |2| = 2 vs 2^2 = 4, so L2 > L1.
        self.assertLess(v_l1b, v_l2b)
        # Sanity for the unit case:
        self.assertAlmostEqual(v_l1, v_l2, places=5)

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

    _STATS = [_signed_log_stats(0.0, 1.0)]
    _METHODS = ["signed-log"]

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

    def test_unknown_weight_mode_raises(self) -> None:
        cfg = _default_loss_cfg(weight_mode="bogus")
        with self.assertRaisesRegex(ValueError, "weight_mode"):
            self._construct(cfg)

    def test_invalid_p_norm_raises(self) -> None:
        cfg = _default_loss_cfg(p_norm=3)
        with self.assertRaisesRegex(ValueError, "p_norm"):
            self._construct(cfg)

    def test_w_min_greater_than_w_max_raises(self) -> None:
        cfg = _default_loss_cfg(w_min=2.0, w_max=1.0)
        with self.assertRaisesRegex(ValueError, "w_min"):
            self._construct(cfg)


class LossDispatchValidatorTests(unittest.TestCase):
    """validate_config dispatch on loss.type."""

    @staticmethod
    def _base_cfg() -> Dict[str, Any]:
        return load_config(PROJECT_ROOT / "config" / "transformer_v2.jsonc")

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


if __name__ == "__main__":
    unittest.main()
