#!/usr/bin/env python3
"""Per-channel-weighted signed-log loss for the atmospheric-profile emulator.

Total per-element loss returned by :class:`AdaptiveSignedLogLoss.forward`:

    L[b, t, c] = lambda_z   * (pred_z   - true_z)^2
               + lambda_phys * w_c
                             * |signed_log10(pred_phys_c) - signed_log10(true_phys_c)|^p_norm

with ``signed_log10(x) := sign(x) * log10(|x| + 1)``. Both targets and
predictions are passed in normalized space; the physical-space term is
recovered per-channel via the normalization metadata stored at preprocessing
time.

For ``signed-log`` normalized channels the inverse identity
``signed_log10(denormalize(z)) == z * std + mean`` is exact, so the slow
``denormalize -> signed_log10`` round-trip is skipped (see ``_apply_signed_log10``).

The loss is intentionally returned UNREDUCED with shape ``[B, T, C]`` so the
trainer can apply the same padding mask reduction used by the legacy hybrid
loss path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from normalizer import DataNormalizer

_ALLOWED_WEIGHT_MODES = {"uniform", "range"}
_ALLOWED_P_NORMS = {1, 2}
_SIGNED_LOG_METHOD = "signed-log"


def _signed_log10(x: Tensor) -> Tensor:
    """Sign-preserving base-10 logarithm: ``sign(x) * log10(|x| + 1)``.

    Well-defined for every real ``x`` (including 0, where it returns 0). The
    derivative has a removable kink at zero — autograd uses the subgradient
    convention, which is finite.
    """
    return torch.sign(x) * torch.log10(torch.abs(x) + 1.0)


class AdaptiveSignedLogLoss(nn.Module):
    """Per-channel weighted signed-log + normalized-MSE loss.

    Args:
        loss_cfg: ``cfg["loss"]`` block. Required keys: ``lambda_z``,
            ``lambda_phys``, ``weight_mode`` (``"uniform"`` | ``"range"``),
            ``weight_power``, ``w_min``, ``w_max``, ``p_norm`` (1 or 2).
        target_methods: Per-channel normalization method strings (length C).
        target_stats: Per-channel normalization stats dicts (length C).
    """

    def __init__(
        self,
        loss_cfg: Dict[str, Any],
        target_methods: List[str],
        target_stats: List[Dict[str, Any]],
    ) -> None:
        super().__init__()

        if len(target_methods) != len(target_stats):
            raise ValueError(
                "target_methods and target_stats must have equal length "
                f"(got {len(target_methods)} vs {len(target_stats)})."
            )
        if not target_methods:
            raise ValueError("AdaptiveSignedLogLoss requires at least one target channel.")

        required = (
            "lambda_z",
            "lambda_phys",
            "weight_mode",
            "weight_power",
            "w_min",
            "w_max",
            "p_norm",
        )
        missing = [k for k in required if k not in loss_cfg]
        if missing:
            raise KeyError(
                f"AdaptiveSignedLogLoss missing required loss config keys: {missing}"
            )

        lambda_z = float(loss_cfg["lambda_z"])
        lambda_phys = float(loss_cfg["lambda_phys"])
        if lambda_z < 0.0 or lambda_phys < 0.0:
            raise ValueError("lambda_z and lambda_phys must be non-negative.")
        if lambda_z == 0.0 and lambda_phys == 0.0:
            raise ValueError("lambda_z and lambda_phys cannot both be zero.")

        weight_mode = str(loss_cfg["weight_mode"]).lower()
        if weight_mode not in _ALLOWED_WEIGHT_MODES:
            raise ValueError(
                f"loss.weight_mode must be one of {sorted(_ALLOWED_WEIGHT_MODES)}; "
                f"got {weight_mode!r}."
            )

        weight_power = float(loss_cfg["weight_power"])
        if weight_power < 0.0:
            raise ValueError("loss.weight_power must be >= 0.")

        w_min = float(loss_cfg["w_min"])
        w_max = float(loss_cfg["w_max"])
        if not (0.0 < w_min <= w_max):
            raise ValueError(
                f"loss weight bounds must satisfy 0 < w_min <= w_max; got {w_min}, {w_max}."
            )

        p_norm = int(loss_cfg["p_norm"])
        if p_norm not in _ALLOWED_P_NORMS:
            raise ValueError(f"loss.p_norm must be in {sorted(_ALLOWED_P_NORMS)}; got {p_norm}.")

        self.lambda_z = lambda_z
        self.lambda_phys = lambda_phys
        self.weight_mode = weight_mode
        self.weight_power = weight_power
        self.w_min = w_min
        self.w_max = w_max
        self.p_norm = p_norm

        self._target_methods: List[str] = list(target_methods)
        self._target_stats: List[Dict[str, Any]] = [dict(s) for s in target_stats]

        num_channels = len(self._target_methods)
        is_fast: List[bool] = []
        fast_means: List[float] = []
        fast_stds: List[float] = []
        for method, stats in zip(self._target_methods, self._target_stats):
            if str(method).lower() == _SIGNED_LOG_METHOD:
                if "mean" not in stats or "std" not in stats:
                    raise KeyError(
                        f"signed-log channel stats must contain 'mean' and 'std'; got keys={sorted(stats.keys())}."
                    )
                is_fast.append(True)
                fast_means.append(float(stats["mean"]))
                fast_stds.append(float(stats["std"]))
            else:
                is_fast.append(False)
                fast_means.append(0.0)
                fast_stds.append(1.0)

        self.register_buffer(
            "_is_signed_log_fast",
            torch.tensor(is_fast, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_fast_mean",
            torch.tensor(fast_means, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fast_std",
            torch.tensor(fast_stds, dtype=torch.float32),
            persistent=False,
        )

        weights = self._compute_channel_weights(num_channels)
        self.register_buffer(
            "channel_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=False,
        )

    def _compute_channel_weights(self, num_channels: int) -> List[float]:
        """Build the per-channel weight vector once at construction time."""
        if self.weight_mode == "uniform":
            return [1.0] * num_channels

        # weight_mode == "range": use std (signed-log spread) as the per-channel
        # range proxy. Channels without a 'std' field (i.e. non-signed-log)
        # fall back to 1.0, which is the neutral choice.
        raw: List[float] = []
        for method, stats in zip(self._target_methods, self._target_stats):
            if str(method).lower() == _SIGNED_LOG_METHOD and "std" in stats:
                raw.append(float(stats["std"]))
            else:
                raw.append(1.0)

        powered = [r ** self.weight_power for r in raw]
        mean_pow = sum(powered) / float(num_channels)
        if mean_pow <= 0.0:
            raise ValueError(
                "Per-channel weight normalization failed: mean(weight_power) is non-positive."
            )
        normalized = [p / mean_pow for p in powered]
        clamped = [min(max(n, self.w_min), self.w_max) for n in normalized]
        # Renormalize after clamping so the mean weight is exactly 1.0 (so the
        # phys-term magnitude does not depend on the choice of bounds). The
        # clamp constrains the *dispersion* of the per-channel weights via
        # max/min <= w_max/w_min; after the final renormalization individual
        # weights may sit slightly outside [w_min, w_max] when clamping kicked
        # in for some channels.
        mean_clamped = sum(clamped) / float(num_channels)
        if mean_clamped <= 0.0:
            raise ValueError(
                "Per-channel weight normalization failed: mean of clamped weights is non-positive."
            )
        return [c / mean_clamped for c in clamped]

    def _apply_signed_log10(self, z_channel: Tensor, channel_idx: int) -> Tensor:
        """Compute ``signed_log10(physical(z_channel))`` for one target channel.

        For ``signed-log`` channels the algebraic identity
        ``signed_log10(denorm(z)) == z * std + mean`` is exact, so the
        denormalize + re-log round-trip is skipped.
        """
        if bool(self._is_signed_log_fast[channel_idx].item()):
            mean = self._fast_mean[channel_idx].to(dtype=z_channel.dtype)
            std = self._fast_std[channel_idx].to(dtype=z_channel.dtype)
            return z_channel * std + mean

        method = self._target_methods[channel_idx]
        stats = self._target_stats[channel_idx]
        phys = DataNormalizer.denormalize_tensor(z_channel, method, stats)
        return _signed_log10(phys)

    def forward(self, predictions_z: Tensor, targets_z: Tensor) -> Tensor:
        """Compute the unreduced loss tensor of shape ``[B, T, C]``.

        Args:
            predictions_z: Model output in normalized space, shape ``[B, T, C]``.
            targets_z: Ground-truth targets in normalized space, shape ``[B, T, C]``.

        Returns:
            Element-wise loss with the same shape as the inputs. The trainer
            applies the padding mask reduction.
        """
        if predictions_z.shape != targets_z.shape:
            raise ValueError(
                f"predictions and targets must have identical shape; "
                f"got {tuple(predictions_z.shape)} vs {tuple(targets_z.shape)}."
            )
        if predictions_z.dim() < 1 or predictions_z.shape[-1] != self.channel_weights.numel():
            raise ValueError(
                f"Last dim of predictions ({predictions_z.shape[-1] if predictions_z.dim() else 'N/A'}) "
                f"must equal number of target channels ({self.channel_weights.numel()})."
            )

        mse_z = (predictions_z - targets_z).pow(2)

        diffs: List[Tensor] = []
        for c in range(self.channel_weights.numel()):
            pred_sl = self._apply_signed_log10(predictions_z[..., c], c)
            targ_sl = self._apply_signed_log10(targets_z[..., c], c)
            delta = pred_sl - targ_sl
            if self.p_norm == 1:
                diffs.append(delta.abs())
            else:
                diffs.append(delta.pow(2))
        physical = torch.stack(diffs, dim=-1)

        weights = self.channel_weights.to(dtype=physical.dtype)
        weighted_physical = physical * weights

        return self.lambda_z * mse_z + self.lambda_phys * weighted_physical


__all__ = ["AdaptiveSignedLogLoss", "_signed_log10"]
