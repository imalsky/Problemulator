#!/usr/bin/env python3
# ruff: noqa: E402
"""
Export a standalone model with built-in normalization, masking, and dynamic sequence length.

The exported ``.pt2`` file is fully standalone: all normalization statistics
and configuration (methods, means, stds, etc.) are baked into the computation
graph as constants, so the file operates entirely in physical space without
needing any external metadata files (normalization_metadata.json, config, etc.).

The exported model accepts inputs in physical units and handles all preprocessing
internally:
    - Infers padding masks from configured padding values
    - Normalizes inputs using stored statistics
    - Runs the transformer
    - Denormalizes outputs back to physical units
    - Restores padding values in output

Input signature (all physical units):
    sequence_phys         [batch, seq_len, input_dim]   Physical-unit input profiles
    global_features_phys  [batch, global_dim]           Physical-unit global scalars

Output:
    predictions           [batch, seq_len, target_dim]  Physical-unit predictions

Output: models/trained_model/stand_alone_model.pt2 (CPU, torch.export program)
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch import Tensor
from torch.export import export as texport, save as tsave, Dim

# Path setup
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE
while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in (PROJECT_ROOT, SRC_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from model import create_prediction_model
from utils import load_config

MODEL_DIR = PROJECT_ROOT / "models" / "trained_model"
NORMALIZATION_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "normalization_metadata.json"
STANDALONE_MODEL_FILENAME = "stand_alone_model.pt2"
DEVICE_EXPORT = torch.device("cpu")
EXPORT_DTYPE = torch.float32


def load_normalization_metadata(path: Path = NORMALIZATION_METADATA_PATH) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


class StandaloneModel(nn.Module):
    """
    Wrapper that handles normalization, masking, and denormalization internally.

    Accepts inputs in physical units: forward(sequence_phys, global_features_phys)
    Returns outputs in physical units with padding values restored.
    """

    def __init__(
            self,
            base_model: nn.Module,
            config: Dict[str, Any],
            norm_metadata: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.base = base_model

        data_spec = config["data_specification"]
        self.input_variables = list(data_spec["input_variables"])
        self.target_variables = list(data_spec["target_variables"])
        self.global_variables = list(data_spec["global_variables"])
        self.padding_value = float(data_spec["padding_value"])
        self.max_seq_len = int(config["model_hyperparameters"]["max_sequence_length"])
        normalization_cfg = config["normalization"]
        self.padding_epsilon = float(normalization_cfg["padding_comparison_epsilon"])
        self.normalized_value_clamp = float(normalization_cfg["normalized_value_clamp"])

        self.norm_methods = norm_metadata["normalization_methods"]
        self.per_key_stats = norm_metadata["per_key_stats"]

    @staticmethod
    def _normalize_tensor_export_safe(
        x: Tensor,
        method: str,
        stats: Dict[str, Any],
        *,
        normalized_value_clamp: float,
    ) -> Tensor:
        if method == "bool" or not stats:
            return x

        if method == "standard":
            y = (x - stats["mean"]) / stats["std"]
        elif method == "log-standard":
            x_safe = torch.clamp_min(x, torch.finfo(x.dtype).eps)
            y = (torch.log10(x_safe) - stats["log_mean"]) / stats["log_std"]
        elif method == "signed-log":
            y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
            y = (y - stats["mean"]) / stats["std"]
        elif method == "log-min-max":
            x_safe = torch.clamp_min(x, torch.finfo(x.dtype).eps)
            log_x = torch.log10(x_safe)
            denom = stats["max"] - stats["min"]
            denom_safe = denom if denom > 0 else torch.finfo(x.dtype).eps
            y = (log_x - stats["min"]) / denom_safe
            y = torch.clamp(y, stats["clamp_min"], stats["clamp_max"])
        elif method == "max-out":
            y = x / stats["max_val"]
        elif method == "iqr":
            y = (x - stats["median"]) / stats["iqr"]
        elif method == "scaled_signed_offset_log":
            y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
            y = y / stats["m"]
        elif method == "symlog":
            thr = stats["threshold"]
            sf = stats["scale_factor"]
            abs_x = torch.abs(x)
            linear_mask = abs_x <= thr
            y = torch.zeros_like(x)
            y[linear_mask] = x[linear_mask] / thr
            y[~linear_mask] = torch.sign(x[~linear_mask]) * (
                torch.log10(abs_x[~linear_mask] / thr) + 1.0
            )
            y = y / sf
        else:
            raise ValueError(f"Unsupported normalization method '{method}'.")

        if method in ("standard", "log-standard", "signed-log", "iqr"):
            y = torch.clamp(y, -normalized_value_clamp, normalized_value_clamp)
        return y

    @staticmethod
    def _denormalize_tensor_export_safe(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        if method == "bool":
            return x
        if not stats:
            raise ValueError(f"No stats for denormalization with method '{method}'")

        if method == "standard":
            return x * stats["std"] + stats["mean"]
        if method == "log-standard":
            return 10 ** (x * stats["log_std"] + stats["log_mean"])
        if method == "signed-log":
            unscaled_log = x * stats["std"] + stats["mean"]
            return torch.sign(unscaled_log) * (10 ** torch.abs(unscaled_log) - 1.0)
        if method == "log-min-max":
            unscaled = (
                torch.clamp(x, stats["clamp_min"], stats["clamp_max"])
                * (stats["max"] - stats["min"])
                + stats["min"]
            )
            return 10 ** unscaled
        if method == "max-out":
            return x * stats["max_val"]
        if method == "iqr":
            return x * stats["iqr"] + stats["median"]
        if method == "scaled_signed_offset_log":
            ytmp = x * stats["m"]
            return torch.sign(ytmp) * (10 ** torch.abs(ytmp) - 1.0)
        if method == "symlog":
            unscaled = x * stats["scale_factor"]
            abs_unscaled = torch.abs(unscaled)
            linear_mask = abs_unscaled <= 1.0
            thr = stats["threshold"]
            y = torch.zeros_like(x)
            y[linear_mask] = unscaled[linear_mask] * thr
            y[~linear_mask] = (
                torch.sign(unscaled[~linear_mask])
                * thr
                * (10 ** (abs_unscaled[~linear_mask] - 1.0))
            )
            return y
        raise ValueError(f"Unsupported denormalization method '{method}'.")

    def _normalize_sequence(self, sequence_phys: Tensor, sequence_mask: Tensor) -> Tensor:
        seq = sequence_phys
        output = seq.clone()
        sequence_mask = sequence_mask.bool()
        padding_fill = torch.full_like(seq[..., 0], self.padding_value)
        safe_fill = torch.ones_like(seq[..., 0])

        for j, var_name in enumerate(self.input_variables):
            method = self.norm_methods[var_name]
            stats = self.per_key_stats[var_name]
            column = seq[..., j]

            if method != "bool" and stats:
                # Replace padded positions with a safe positive value before normalization,
                # then restore padding sentinel after normalization.
                safe_column = torch.where(sequence_mask, safe_fill, column)
                normalized = self._normalize_tensor_export_safe(
                    safe_column,
                    method,
                    stats,
                    normalized_value_clamp=self.normalized_value_clamp,
                )
                output[..., j] = torch.where(sequence_mask, padding_fill, normalized)
            else:
                output[..., j] = torch.where(sequence_mask, padding_fill, column)

        return output

    def _normalize_globals(self, globals_phys: Optional[Tensor]) -> Optional[Tensor]:
        if globals_phys is None or not self.global_variables:
            return None

        g = globals_phys
        output = torch.empty_like(g)

        for j, var_name in enumerate(self.global_variables):
            method = self.norm_methods[var_name]
            stats = self.per_key_stats[var_name]
            column = g[..., j]

            if method != "bool" and stats:
                column_norm = self._normalize_tensor_export_safe(
                    column,
                    method,
                    stats,
                    normalized_value_clamp=self.normalized_value_clamp,
                )
            else:
                column_norm = column

            output[..., j] = column_norm

        return output

    def _denormalize_targets(self, outputs_norm: Tensor) -> Tensor:
        out = outputs_norm
        physical = torch.empty_like(out)

        for j, var_name in enumerate(self.target_variables):
            method = self.norm_methods[var_name]
            stats = self.per_key_stats[var_name]
            column = out[..., j]

            if method != "bool" and stats:
                column_phys = self._denormalize_tensor_export_safe(column, method, stats)
            else:
                column_phys = column

            physical[..., j] = column_phys

        return physical

    def _build_mask_from_padding(self, sequence_phys: Tensor) -> Tensor:
        """Infer padding mask where all features equal padding_value."""
        difference = (sequence_phys - self.padding_value).abs()
        return difference.amax(dim=-1) <= self.padding_epsilon

    def _check_sequence_length(self, sequence_phys: Tensor) -> None:
        if sequence_phys.ndim != 3:
            raise ValueError(f"sequence_phys must be [B, L, F_in], got {tuple(sequence_phys.shape)}")

        length = sequence_phys.shape[1]
        if length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {length} exceeds max_sequence_length={self.max_seq_len}. "
                "Pad or truncate upstream."
            )

    @torch.inference_mode()
    def forward(
            self,
            sequence_phys: Tensor,
            global_features_phys: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run physical-space inference end to end.

        Args:
            sequence_phys: Physical-unit input tensor with shape
                ``[batch, seq_len, input_dim]``
            global_features_phys: Optional physical-unit global features with shape
                ``[batch, global_dim]``

        Returns:
            Physical-unit predictions with shape ``[batch, seq_len, target_dim]``.
            Padding positions are restored to ``self.padding_value`` in the returned tensor.
        """
        self._check_sequence_length(sequence_phys)

        sequence_mask = self._build_mask_from_padding(sequence_phys)
        sequence_norm = self._normalize_sequence(sequence_phys, sequence_mask)
        globals_norm = self._normalize_globals(global_features_phys)

        outputs_norm = self.base(sequence_norm, globals_norm, sequence_mask)
        outputs_phys = self._denormalize_targets(outputs_norm)

        outputs_phys = outputs_phys.clone()
        outputs_phys[sequence_mask] = self.padding_value

        return outputs_phys


def build_base_model(
        cfg_path: Path,
        ckpt_path: Path,
        device: torch.device,
        dtype: torch.dtype,
) -> Tuple[nn.Module, Dict[str, Any]]:
    config = load_config(cfg_path)

    model = create_prediction_model(
        config=config,
        device=torch.device("cpu"),
        compile_model=False,
    ).eval()
    torch.set_grad_enabled(False)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
        print("Removed '_orig_mod.' prefix from state dict keys")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype).eval()

    print(
        f"Loaded base model from {ckpt_path.name}: "
        f"epoch={checkpoint['epoch']}, "
        f"val_loss={checkpoint['val_loss']:.6e}"
    )
    return model, config


def build_stand_alone_model(
        device: torch.device,
        dtype: torch.dtype,
) -> Tuple[StandaloneModel, Dict[str, Any]]:
    cfg_path = MODEL_DIR / "train_config.json"
    ckpt_path = MODEL_DIR / "best_model.pt"

    base_model, config = build_base_model(cfg_path, ckpt_path, device=device, dtype=dtype)
    norm_metadata = load_normalization_metadata(NORMALIZATION_METADATA_PATH)

    stand_alone = StandaloneModel(
        base_model=base_model,
        config=config,
        norm_metadata=norm_metadata,
    ).to(device=device, dtype=dtype).eval()

    return stand_alone, config


def export_stand_alone_model(
        stand_alone_model: StandaloneModel,
        config: Dict[str, Any],
        out_path: Path,
) -> Path:
    """Export model as PT2 with dynamic batch size and sequence length."""
    data_spec = config["data_specification"]
    max_seq_len = int(config["model_hyperparameters"]["max_sequence_length"])
    input_dim = int(len(data_spec["input_variables"]))
    global_dim = int(len(data_spec["global_variables"]))
    output_dim = int(len(data_spec["target_variables"]))

    print(
        f"Exporting: max_seq_len={max_seq_len}, "
        f"input_dim={input_dim}, global_dim={global_dim}, output_dim={output_dim}"
    )

    kwargs = {
        "sequence_phys": torch.ones(2, max_seq_len, input_dim, dtype=EXPORT_DTYPE, device=DEVICE_EXPORT),
    }
    if global_dim > 0:
        kwargs["global_features_phys"] = torch.ones(2, global_dim, dtype=EXPORT_DTYPE, device=DEVICE_EXPORT)

    batch_dim = Dim("batch", min=1, max=8192)
    length_dim = Dim("length", min=1, max=max_seq_len)

    dynamic_shapes = {"sequence_phys": {0: batch_dim, 1: length_dim}}
    if global_dim > 0:
        dynamic_shapes["global_features_phys"] = {0: batch_dim}

    program = texport(
        stand_alone_model.to(device=DEVICE_EXPORT, dtype=EXPORT_DTYPE),
        args=(),
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    tsave(program, str(out_path))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved standalone model to {out_path} ({size_mb:.2f} MB)")
    return out_path


def main() -> None:
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"MODEL_DIR = {MODEL_DIR}")

    stand_alone_model, config = build_stand_alone_model(
        device=DEVICE_EXPORT,
        dtype=EXPORT_DTYPE,
    )

    out_path = MODEL_DIR / STANDALONE_MODEL_FILENAME
    export_stand_alone_model(stand_alone_model, config, out_path)


if __name__ == "__main__":
    main()
