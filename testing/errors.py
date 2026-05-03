#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate the standalone physical-space PT2 model on the test split."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
CPU_THREADS = max(1, (os.cpu_count() or 8) // 2)
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)
os.environ["ACCELERATE_MATMUL_MULTITHREADING"] = "1"

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import create_collate_fn, create_dataset
from normalizer import DataNormalizer
from utils import load_config

MODEL_DIR = PROJECT_ROOT / "models" / "trained_model"
PROC_ROOT = PROJECT_ROOT / "data" / "processed"
PROC_TEST = PROC_ROOT / "test"
NORM_META_PATH = PROC_ROOT / "normalization_metadata.json"
STANDALONE_PT2_PATH = MODEL_DIR / "stand_alone_model.pt2"

TEST_FRACTION = 1.0
RANDOM_SEED = 42
BATCH_SIZE = 1024
NUM_WORKERS = 0
PIN_MEMORY = False
PCT_DENOM_FLOOR = 1.0

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _denormalize_channel(
    values: Tensor,
    var_name: str,
    norm_meta: Dict[str, Any],
) -> Tensor:
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


def _combine_stats(
    count_a: int,
    mean_a: float,
    m2_a: float,
    count_b: int,
    mean_b: float,
    m2_b: float,
) -> Tuple[int, float, float]:
    if count_a == 0:
        return count_b, mean_b, m2_b
    if count_b == 0:
        return count_a, mean_a, m2_a
    delta = mean_b - mean_a
    count = count_a + count_b
    mean = mean_a + delta * (count_b / count)
    m2 = m2_a + m2_b + delta * delta * (count_a * count_b / count)
    return count, mean, m2


def main() -> None:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, CPU_THREADS // 2))

    if not STANDALONE_PT2_PATH.is_file():
        raise FileNotFoundError(
            f"Missing standalone export: {STANDALONE_PT2_PATH}. "
            "Generate it with: python testing/export.py"
        )

    config = load_config(MODEL_DIR / "train_config.json")
    padding_value = float(config["data_specification"]["padding_value"])
    padding_epsilon = float(config["normalization"]["padding_comparison_epsilon"])
    with NORM_META_PATH.open("r", encoding="utf-8") as f:
        norm_meta = json.load(f)

    standalone_program = torch.export.load(str(STANDALONE_PT2_PATH))
    model = standalone_program.module()

    with (PROC_TEST / "metadata.json").open("r", encoding="utf-8") as f:
        total_samples = int(json.load(f)["total_samples"])
    n_samples = max(1, int(total_samples * TEST_FRACTION))
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.sort(rng.choice(total_samples, n_samples, replace=False)).tolist()
    print(
        f"Processing {n_samples}/{total_samples} samples ({TEST_FRACTION * 100:.1f}%), "
        f"batch_size={BATCH_SIZE}"
    )

    dataset = create_dataset(PROC_TEST, config, indices)
    collate_fn = create_collate_fn(padding_value, padding_epsilon)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=False,
    )

    target_vars = config["data_specification"]["target_variables"]
    variable_units = config["data_specification"]["variable_units"]
    flux_vars = [n for n in ("net_thermal_flux", "net_reflected_flux") if n in target_vars]
    if not flux_vars:
        raise RuntimeError("No required flux targets found in config.")
    flux_idx = {n: target_vars.index(n) for n in flux_vars}

    pct_denom_floor = torch.tensor(PCT_DENOM_FLOOR, dtype=DTYPE, device=DEVICE)
    acc = {
        name: {
            "count": 0,
            "mae_sum": 0.0,
            "sse_sum": 0.0,
            "t_count": 0,
            "t_mean": 0.0,
            "t_M2": 0.0,
            "abs_pct_err_chunks": [],
            "signed_pct_err_chunks": [],
        }
        for name in flux_vars
    }

    for (
        batch_inputs_norm,
        batch_masks,
        batch_targets_norm,
        target_masks,
    ) in tqdm(loader, total=len(loader), desc="Batches"):
        sequence_phys, globals_phys, targets_phys, target_mask_bool = _denormalize_batch_to_physical(
            config=config,
            norm_meta=norm_meta,
            batch_inputs_norm=batch_inputs_norm,
            batch_masks=batch_masks,
            batch_targets_norm=batch_targets_norm,
            target_masks=target_masks,
        )

        with torch.inference_mode():
            predictions_phys = model(
                sequence_phys=sequence_phys.to(device=DEVICE, dtype=DTYPE),
                global_features_phys=(
                    globals_phys.to(device=DEVICE, dtype=DTYPE)
                    if globals_phys is not None
                    else None
                ),
            )

        b, seq_len, s = predictions_phys.shape
        valid = (~target_mask_bool).reshape(-1)
        if not bool(torch.any(valid)):
            continue

        preds_all = predictions_phys.reshape(b * seq_len, s)[valid]
        targs_all = targets_phys.reshape(b * seq_len, s)[valid]

        for name in flux_vars:
            i = flux_idx[name]
            p = preds_all[:, i]
            t = targs_all[:, i]

            diff = p - t
            abs_diff = diff.abs()
            n = int(p.numel())

            acc[name]["mae_sum"] += float(abs_diff.sum().item())
            acc[name]["sse_sum"] += float((diff * diff).sum().item())
            acc[name]["count"] += n

            b_mean = float(t.mean().item())
            b_m2 = float(((t - b_mean) ** 2).sum().item())
            (
                acc[name]["t_count"],
                acc[name]["t_mean"],
                acc[name]["t_M2"],
            ) = _combine_stats(
                acc[name]["t_count"],
                acc[name]["t_mean"],
                acc[name]["t_M2"],
                n,
                b_mean,
                b_m2,
            )

            denom = torch.maximum(t.abs(), pct_denom_floor)
            abs_pct = 100.0 * abs_diff / denom
            signed_pct = 100.0 * diff / denom
            acc[name]["abs_pct_err_chunks"].append(abs_pct.cpu().numpy())
            acc[name]["signed_pct_err_chunks"].append(signed_pct.cpu().numpy())

    print("\n" + "=" * 60)
    print("TEST SET ERROR STATISTICS (physical-space standalone model)")
    print(f"Error floor: {PCT_DENOM_FLOOR}")
    print("=" * 60)

    for name in flux_vars:
        c = int(acc[name]["count"])
        if c == 0:
            print(f"\n{name.replace('_', ' ').title()}: no valid points")
            continue

        mae = acc[name]["mae_sum"] / c
        rmse = math.sqrt(acc[name]["sse_sum"] / c)

        t_count = int(acc[name]["t_count"])
        t_m2 = float(acc[name]["t_M2"])
        if t_count <= 1 or t_m2 == 0.0:
            r2 = float("nan")
        else:
            r2 = 1.0 - (acc[name]["sse_sum"] / t_m2)

        abs_pct_all = (
            np.concatenate(acc[name]["abs_pct_err_chunks"])
            if acc[name]["abs_pct_err_chunks"]
            else np.array([])
        )
        signed_pct_all = (
            np.concatenate(acc[name]["signed_pct_err_chunks"])
            if acc[name]["signed_pct_err_chunks"]
            else np.array([])
        )

        abs_mean_pct = float(np.mean(abs_pct_all)) if abs_pct_all.size else float("nan")
        abs_median_pct = float(np.median(abs_pct_all)) if abs_pct_all.size else float("nan")

        signed_mean_pct = float(np.mean(signed_pct_all)) if signed_pct_all.size else float("nan")
        signed_median_pct = float(np.median(signed_pct_all)) if signed_pct_all.size else float("nan")
        if signed_pct_all.size:
            signed_std_pct = float(np.std(signed_pct_all))
            signed_q25 = float(np.percentile(signed_pct_all, 25))
            signed_q75 = float(np.percentile(signed_pct_all, 75))
            signed_q05 = float(np.percentile(signed_pct_all, 5))
            signed_q95 = float(np.percentile(signed_pct_all, 95))
        else:
            signed_std_pct = signed_q25 = signed_q75 = signed_q05 = signed_q95 = float("nan")

        print(f"\n{name.replace('_', ' ').title()}:")
        unit = variable_units.get(name, "native units")
        print(f"  MAE:                      {mae:.3e} {unit}")
        print(f"  RMSE:                     {rmse:.3e} {unit}")
        print(f"  R-squared:                {r2:.6f}")

        print("\n  Absolute Percent Errors:")
        print(f"    Mean:                   {abs_mean_pct:.2f}%")
        print(f"    Median:                 {abs_median_pct:.2f}%")

        print("\n  Signed Percent Errors:")
        print(f"    Mean:                   {signed_mean_pct:+.2f}%")
        print(f"    Median:                 {signed_median_pct:+.2f}%")
        print(f"    Std Dev:                {signed_std_pct:.2f}%")
        print(f"    5th percentile:         {signed_q05:+.2f}%")
        print(f"    25th percentile (Q1):   {signed_q25:+.2f}%")
        print(f"    75th percentile (Q3):   {signed_q75:+.2f}%")
        print(f"    95th percentile:        {signed_q95:+.2f}%")

        if abs(signed_mean_pct) < 0.5:
            bias = "No significant bias"
        elif signed_mean_pct > 0:
            bias = "Overprediction bias"
        else:
            bias = "Underprediction bias"
        print(f"    Bias assessment:        {bias}")


if __name__ == "__main__":
    main()
