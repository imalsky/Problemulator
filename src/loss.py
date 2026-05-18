#!/usr/bin/env python3
"""Normalized-MSE plus physical fractional-L1 loss for the emulator.

Total per-element loss returned by :class:`AdaptiveSignedLogLoss.forward`:

    L[b, t, c] = lambda_z   * (pred_z   - true_z)^2
               + lambda_phys * |pred_phys_c - true_phys_c|
                             / sqrt(true_phys_c^2 + fractional_epsilon)

Both targets and predictions are passed in normalized space; the physical-space
term is recovered per channel via the normalization metadata stored at
preprocessing time.

The loss is intentionally returned UNREDUCED with shape ``[B, T, C]`` so the
trainer can apply the same padding mask reduction used by the training path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from normalizer import DataNormalizer


class AdaptiveSignedLogLoss(nn.Module):
    """Normalized-MSE plus physical-space fractional-L1 loss.

    Args:
        loss_cfg: ``cfg["loss"]`` block. Required keys: ``lambda_z``,
            ``lambda_phys``, ``fractional_epsilon``, and ``p_norm``. The only
            supported ``p_norm`` is 1; it is retained as a compatibility guard.
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
            "fractional_epsilon",
            "p_norm",
        )
        missing = [k for k in required if k not in loss_cfg]
        if missing:
            raise KeyError(
                f"AdaptiveSignedLogLoss missing required loss config keys: {missing}"
            )

        if isinstance(loss_cfg["lambda_z"], bool) or isinstance(
            loss_cfg["lambda_phys"], bool
        ):
            raise ValueError("lambda_z and lambda_phys must be numeric.")
        lambda_z = float(loss_cfg["lambda_z"])
        lambda_phys = float(loss_cfg["lambda_phys"])
        if not math.isfinite(lambda_z) or not math.isfinite(lambda_phys):
            raise ValueError("lambda_z and lambda_phys must be finite.")
        if lambda_z < 0.0 or lambda_phys < 0.0:
            raise ValueError("lambda_z and lambda_phys must be non-negative.")
        if lambda_z == 0.0 and lambda_phys == 0.0:
            raise ValueError("lambda_z and lambda_phys cannot both be zero.")

        if isinstance(loss_cfg["fractional_epsilon"], bool):
            raise ValueError("loss.fractional_epsilon must be numeric.")
        fractional_epsilon = float(loss_cfg["fractional_epsilon"])
        if not math.isfinite(fractional_epsilon) or fractional_epsilon <= 0.0:
            raise ValueError("loss.fractional_epsilon must be finite and > 0.")

        p_norm = loss_cfg["p_norm"]
        if isinstance(p_norm, bool) or not isinstance(p_norm, int) or p_norm != 1:
            raise ValueError("loss.p_norm must be the integer 1 for fractional L1 loss.")

        self.lambda_z = lambda_z
        self.lambda_phys = lambda_phys
        self.fractional_epsilon = fractional_epsilon
        self.p_norm = p_norm

        self._target_methods: List[str] = list(target_methods)
        self._target_stats: List[Dict[str, Any]] = [dict(s) for s in target_stats]
        self.num_channels = len(self._target_methods)

    def _denormalize_per_channel(self, values_z: Tensor) -> Tensor:
        """Denormalize a ``[..., C]`` tensor using per-target metadata."""
        if values_z.shape[-1] != self.num_channels:
            raise ValueError(
                f"Last dim of tensor ({values_z.shape[-1]}) does not match "
                f"number of target channels ({self.num_channels})."
            )

        chunks: List[Tensor] = []
        for channel_idx, (method, stats) in enumerate(
            zip(self._target_methods, self._target_stats)
        ):
            chunks.append(
                DataNormalizer.denormalize_tensor(
                    values_z[..., channel_idx],
                    method,
                    stats,
                )
            )
        return torch.stack(chunks, dim=-1)

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
        if predictions_z.dim() < 1 or predictions_z.shape[-1] != self.num_channels:
            raise ValueError(
                f"Last dim of predictions ({predictions_z.shape[-1] if predictions_z.dim() else 'N/A'}) "
                f"must equal number of target channels ({self.num_channels})."
            )

        mse_z = (predictions_z - targets_z).pow(2)
        if self.lambda_phys == 0.0:
            return self.lambda_z * mse_z

        pred_phys = self._denormalize_per_channel(predictions_z)
        targ_phys = self._denormalize_per_channel(targets_z)
        eps = torch.as_tensor(
            self.fractional_epsilon,
            dtype=targ_phys.dtype,
            device=targ_phys.device,
        )
        fractional_l1 = (pred_phys - targ_phys).abs() / torch.sqrt(
            targ_phys.pow(2) + eps
        )

        return self.lambda_z * mse_z + self.lambda_phys * fractional_l1


__all__ = ["AdaptiveSignedLogLoss"]
