#!/usr/bin/env python3
# ruff: noqa: E402
"""Plot true vs predicted target profiles for one test example."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import Tensor
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import create_collate_fn, create_dataset
from normalizer import DataNormalizer
from utils import load_config

MODELS_ROOT = PROJECT_ROOT / "models"
PROC_ROOT = PROJECT_ROOT / "data" / "processed"
PROC_TEST = PROC_ROOT / "test"
STYLE_PATH = THIS_FILE.with_name("science.mplstyle")
DEFAULT_EXAMPLE_INDEX = 0
DEFAULT_MODEL_DIR_NAMES = ("trained_model", "newtrained_model", "old_trained_model")
DEFAULT_PADDING_COMPARISON_EPSILON = 1e-6
NUM_PROFILES_TO_PLOT = 4
THERMAL_TARGET_NAME = "net_thermal_flux"
THERMAL_FLUX_CGS_UNIT = r"erg cm$^{-2}$ s$^{-1}$"
PROFILE_LINE_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
TRUE_LINE_ALPHA = 0.35
PRED_LINE_ALPHA = 1.0
PROFILE_LINE_WIDTH = 3.0
PRED_DASH_PATTERN = (0, (10, 6))
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16
LEGACY_DATASET_MISC_DEFAULTS = {
    "dataset_loading_mode": "auto",
    "dataset_max_cached_shards": 200,
    "dataset_large_shard_mmap_bytes": 52_428_800,
    "dataset_ram_safety_fraction": 0.8,
    "dataset_copy_mmap_slices": True,
}

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _resolve_default_model_dir() -> Path:
    for model_dir_name in DEFAULT_MODEL_DIR_NAMES:
        candidate = MODELS_ROOT / model_dir_name
        if candidate.is_dir():
            return candidate

    available_model_dirs = sorted(
        path for path in MODELS_ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if len(available_model_dirs) == 1:
        return available_model_dirs[0]

    raise FileNotFoundError(
        f"Could not determine a default model directory in {MODELS_ROOT}. "
        "Pass --model-dir explicitly."
    )


def _resolve_model_dir(model_dir_arg: Optional[str]) -> Path:
    if model_dir_arg is None:
        return _resolve_default_model_dir()

    model_dir = Path(model_dir_arg).expanduser()
    if not model_dir.is_absolute():
        model_dir = MODELS_ROOT / model_dir
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    return model_dir


def _resolve_norm_meta_path(model_dir: Path) -> Path:
    model_local_norm_meta = model_dir / "normalization_metadata.json"
    if model_local_norm_meta.is_file():
        return model_local_norm_meta
    return PROC_ROOT / "normalization_metadata.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true vs predicted thermal profiles for several test examples."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model directory name under models/ or an absolute path.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_EXAMPLE_INDEX,
        help=(
            "0-based starting index in processed test split. "
            f"Up to {NUM_PROFILES_TO_PLOT} consecutive profiles are plotted."
        ),
    )
    return parser.parse_args()


def _denormalize_channel(values: Tensor, var_name: str, norm_meta: Dict[str, Any]) -> Tensor:
    method = norm_meta["normalization_methods"][var_name]
    stats = norm_meta["per_key_stats"][var_name]
    if method != "bool" and stats:
        return DataNormalizer.denormalize_tensor(values, method, stats)
    return values


def _denormalize_batch_to_physical(
    config: Dict[str, Any],
    norm_meta: Dict[str, Any],
    batch_inputs_norm: Dict[str, Tensor],
    batch_masks: Dict[str, Tensor],
    batch_targets_norm: Tensor,
    target_masks: Tensor,
) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
    data_spec = config["data_specification"]
    input_vars = data_spec["input_variables"]
    global_vars = data_spec["global_variables"]
    target_vars = data_spec["target_variables"]
    padding_value = float(data_spec["padding_value"])

    sequence_norm = batch_inputs_norm["sequence"].to(dtype=DTYPE)
    sequence_mask = batch_masks["sequence"].bool()
    sequence_phys = torch.empty_like(sequence_norm, dtype=DTYPE)
    for j, var_name in enumerate(input_vars):
        sequence_phys[..., j] = _denormalize_channel(
            sequence_norm[..., j], var_name, norm_meta
        ).to(dtype=DTYPE)
    sequence_phys[sequence_mask] = padding_value

    globals_phys: Optional[Tensor] = None
    if global_vars and "global_features" in batch_inputs_norm:
        globals_norm = batch_inputs_norm["global_features"].to(dtype=DTYPE)
        globals_phys = torch.empty_like(globals_norm, dtype=DTYPE)
        for j, var_name in enumerate(global_vars):
            globals_phys[..., j] = _denormalize_channel(
                globals_norm[..., j], var_name, norm_meta
            ).to(dtype=DTYPE)

    targets_norm = batch_targets_norm.to(dtype=DTYPE)
    target_mask_bool = target_masks.bool()
    targets_phys = torch.empty_like(targets_norm, dtype=DTYPE)
    for j, var_name in enumerate(target_vars):
        targets_phys[..., j] = _denormalize_channel(
            targets_norm[..., j], var_name, norm_meta
        ).to(dtype=DTYPE)
    targets_phys[target_mask_bool] = padding_value

    return sequence_phys, globals_phys, targets_phys, target_mask_bool


def _load_plot_config(config_path: Path) -> Dict[str, Any]:
    """Load the saved model config, tolerating legacy snapshots for plotting."""
    try:
        config = load_config(config_path)
    except RuntimeError:
        with config_path.open("r", encoding="utf-8-sig") as f:
            config = json.load(f)

    required_sections = (
        "miscellaneous_settings",
        "data_specification",
        "normalization",
    )
    missing_sections = [
        section_name
        for section_name in required_sections
        if not isinstance(config.get(section_name), dict)
    ]
    if missing_sections:
        raise RuntimeError(
            f"Legacy plotting config is missing required sections: {missing_sections}"
        )

    misc_cfg = config["miscellaneous_settings"]
    for key, value in LEGACY_DATASET_MISC_DEFAULTS.items():
        misc_cfg.setdefault(key, value)

    normalization_cfg = config["normalization"]
    normalization_cfg.setdefault(
        "padding_comparison_epsilon",
        DEFAULT_PADDING_COMPARISON_EPSILON,
    )

    data_spec = config["data_specification"]
    data_spec.setdefault("variable_units", {})
    data_spec.setdefault("global_variables", [])

    return config


def main() -> None:
    if not STYLE_PATH.is_file():
        raise FileNotFoundError(f"Missing matplotlib style file: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))

    args = _parse_args()
    model_dir = _resolve_model_dir(args.model_dir)
    standalone_pt2_path = model_dir / "stand_alone_model.pt2"
    norm_meta_path = _resolve_norm_meta_path(model_dir)
    if args.index < 0:
        raise ValueError("--index must be >= 0.")

    if not standalone_pt2_path.is_file():
        raise FileNotFoundError(
            f"Missing standalone export: {standalone_pt2_path}. "
            "Generate it with: python testing/export.py"
        )

    config = _load_plot_config(model_dir / "train_config.json")
    padding_value = float(config["data_specification"]["padding_value"])
    padding_epsilon = float(config["normalization"]["padding_comparison_epsilon"])
    with norm_meta_path.open("r", encoding="utf-8") as f:
        norm_meta = json.load(f)

    with (PROC_TEST / "metadata.json").open("r", encoding="utf-8") as f:
        total_samples = int(json.load(f)["total_samples"])
    if args.index >= total_samples:
        raise IndexError(
            f"Requested index {args.index} is out of range for test split size {total_samples}."
        )
    selected_indices = list(
        range(args.index, min(args.index + NUM_PROFILES_TO_PLOT, total_samples))
    )

    dataset = create_dataset(PROC_TEST, config, indices=selected_indices)
    loader = DataLoader(
        dataset,
        batch_size=len(selected_indices),
        shuffle=False,
        collate_fn=create_collate_fn(padding_value, padding_epsilon),
    )
    batch_inputs_norm, batch_masks, batch_targets_norm, target_masks = next(iter(loader))

    sequence_phys, globals_phys, targets_phys, target_mask_bool = _denormalize_batch_to_physical(
        config=config,
        norm_meta=norm_meta,
        batch_inputs_norm=batch_inputs_norm,
        batch_masks=batch_masks,
        batch_targets_norm=batch_targets_norm,
        target_masks=target_masks,
    )

    program = torch.export.load(str(standalone_pt2_path))
    model = program.module()
    with torch.inference_mode():
        predictions_phys = model(
            sequence_phys=sequence_phys.to(device=DEVICE, dtype=DTYPE),
            global_features_phys=(
                globals_phys.to(device=DEVICE, dtype=DTYPE)
                if globals_phys is not None
                else None
            ),
        )

    input_vars = config["data_specification"]["input_variables"]
    target_vars = config["data_specification"]["target_variables"]
    if THERMAL_TARGET_NAME not in target_vars:
        raise RuntimeError(
            f"Required thermal target '{THERMAL_TARGET_NAME}' not found in config."
        )
    thermal_idx = target_vars.index(THERMAL_TARGET_NAME)

    coord_label: str
    use_pressure_axis = False
    if "pressure_bar" in input_vars:
        pressure_idx = input_vars.index("pressure_bar")
        coord_label = "Pressure (bar)"
        use_pressure_axis = True
    else:
        coord_label = "Layer Index"

    fig, ax = plt.subplots(figsize=(6, 6))

    print(f"Profile indices: {selected_indices}")
    plotted_profiles = 0
    for profile_offset, sample_index in enumerate(selected_indices):
        valid = (~target_mask_bool[profile_offset]).cpu().numpy()
        if not np.any(valid):
            print(f"Skipping index {sample_index}: no valid (non-padding) target steps.")
            continue

        pred_np = predictions_phys[profile_offset].cpu().numpy()[valid]
        true_np = targets_phys[profile_offset].cpu().numpy()[valid]
        finite_rows = np.isfinite(pred_np).all(axis=1) & np.isfinite(true_np).all(axis=1)
        if not bool(np.any(finite_rows)):
            print(f"Skipping index {sample_index}: no finite rows available for plotting.")
            continue

        if use_pressure_axis:
            pressure_values = sequence_phys[profile_offset, :, pressure_idx].cpu().numpy()[valid]
            pressure_values = pressure_values[finite_rows]
            finite_pos = np.isfinite(pressure_values) & (pressure_values > 0)
            if not bool(np.all(finite_pos)):
                print(
                    f"Skipping index {sample_index}: non-positive or non-finite pressure values."
                )
                continue
            coord_values = pressure_values
        else:
            coord_values = np.arange(int(valid.sum()), dtype=np.int64)[finite_rows]

        pred_np = pred_np[finite_rows]
        true_np = true_np[finite_rows]

        if use_pressure_axis:
            order = np.argsort(coord_values)
            coord_values = coord_values[order]
            pred_np = pred_np[order]
            true_np = true_np[order]

        y_true = true_np[:, thermal_idx]
        y_pred = pred_np[:, thermal_idx]
        positive_mask = (y_true > 0.0) & (y_pred > 0.0)
        if not bool(np.any(positive_mask)):
            print(f"Skipping index {sample_index}: no positive thermal flux values.")
            continue

        y_true_plot = y_true[positive_mask]
        y_pred_plot = y_pred[positive_mask]
        coord_var = coord_values[positive_mask]
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        print(
            f"Index {sample_index}: MAE={mae:.3e} {THERMAL_FLUX_CGS_UNIT}  "
            f"RMSE={rmse:.3e} {THERMAL_FLUX_CGS_UNIT}"
        )

        color = PROFILE_LINE_COLORS[profile_offset % len(PROFILE_LINE_COLORS)]
        if use_pressure_axis:
            ax.plot(
                y_true_plot,
                coord_var,
                color=color,
                linewidth=PROFILE_LINE_WIDTH,
                alpha=TRUE_LINE_ALPHA,
                zorder=1,
            )
            ax.plot(
                y_pred_plot,
                coord_var,
                color=color,
                linestyle=PRED_DASH_PATTERN,
                linewidth=PROFILE_LINE_WIDTH,
                alpha=PRED_LINE_ALPHA,
                zorder=3,
            )
        else:
            ax.plot(
                coord_var,
                y_true_plot,
                color=color,
                linewidth=PROFILE_LINE_WIDTH,
                alpha=TRUE_LINE_ALPHA,
                zorder=1,
            )
            ax.plot(
                coord_var,
                y_pred_plot,
                color=color,
                linestyle=PRED_DASH_PATTERN,
                linewidth=PROFILE_LINE_WIDTH,
                alpha=PRED_LINE_ALPHA,
                zorder=3,
            )
        plotted_profiles += 1

    if plotted_profiles == 0:
        raise RuntimeError("No profiles were eligible for plotting.")

    if use_pressure_axis:
        ax.set_ylabel(coord_label)
        ax.set_xlabel(f"Thermal Flux ({THERMAL_FLUX_CGS_UNIT})")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1e4, 1e10)
        ax.set_ylim(1e2, 1e-5)
    else:
        ax.set_ylabel(f"Thermal Flux ({THERMAL_FLUX_CGS_UNIT})")
        ax.set_xlabel(coord_label)
        ax.set_yscale("log")
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_FONT_SIZE)
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=PROFILE_LINE_WIDTH, linestyle="-"),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=PROFILE_LINE_WIDTH,
            linestyle=PRED_DASH_PATTERN,
        ),
    ]
    ax.legend(legend_handles, ["True", "Surrogate Model"], loc="best")
    ax.set_box_aspect(1)

    fig.tight_layout()
    out_path = (
        model_dir
        / "plots"
        / (
            f"example_thermal_prediction_vs_true_idx_{selected_indices[0]:06d}"
            f"_to_{selected_indices[-1]:06d}.png"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
