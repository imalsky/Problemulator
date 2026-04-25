#!/usr/bin/env python3
"""
preprocess.py - Preprocess raw HDF5 data into normalized NPY shards.

This module:
- Computes normalization statistics from the training split
- Normalizes raw HDF5 data into fixed-length NPY shards for train/val/test
- Groups channels by normalization method so preprocessing can stay NumPy-native
- Preserves the padding convention (padding values remain padding_value)
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from normalizer import DataNormalizer
from utils import (
    ensure_dirs,
    get_precision_config,
    save_json,
)

logger = logging.getLogger(__name__)

TORCH_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}

METHOD_STAT_FIELDS = {
    "standard": ("mean", "std"),
    "log-standard": ("log_mean", "log_std"),
    "signed-log": ("mean", "std"),
    "log-min-max": ("min", "max", "clamp_min", "clamp_max"),
    "max-out": ("max_val",),
    "iqr": ("median", "iqr"),
    "scaled_signed_offset_log": ("m",),
    "symlog": ("threshold", "scale_factor"),
}


def _build_channel_normalization_groups(
    variables: List[str],
    norm_methods: Dict[str, str],
    norm_stats: Dict[str, Dict[str, Any]],
    *,
    dtype: np.dtype,
) -> List[Dict[str, Any]]:
    """Group channel indices by normalization method with broadcast-ready stats."""
    grouped_specs: Dict[str, Dict[str, Any]] = {}

    for channel_idx, var in enumerate(variables):
        method = str(norm_methods[var])
        stats = norm_stats[var]
        if method == "bool" or not stats:
            continue

        if method not in METHOD_STAT_FIELDS:
            raise ValueError(f"Unsupported normalization method '{method}' for '{var}'.")

        spec = grouped_specs.setdefault(
            method,
            {
                "method": method,
                "indices": [],
                "stats": {field: [] for field in METHOD_STAT_FIELDS[method]},
            },
        )
        spec["indices"].append(channel_idx)
        for field in METHOD_STAT_FIELDS[method]:
            spec["stats"][field].append(stats[field])

    output_specs: List[Dict[str, Any]] = []
    for method in grouped_specs:
        spec = grouped_specs[method]
        output_specs.append(
            {
                "method": method,
                "indices": np.asarray(spec["indices"], dtype=np.int64),
                "stats": {
                    field: np.asarray(values, dtype=dtype)
                    for field, values in spec["stats"].items()
                },
            }
        )

    return output_specs


def _right_pad_sequence_batch(
    array: np.ndarray, *, max_seq_len: int, padding_value: float
) -> np.ndarray:
    """Right-pad a [batch, seq_len, channels] array to ``max_seq_len``."""
    cur_len = int(array.shape[1])
    if cur_len >= max_seq_len:
        return array

    padded = np.full(
        (array.shape[0], max_seq_len, array.shape[2]),
        padding_value,
        dtype=array.dtype,
    )
    padded[:, :cur_len, :] = array
    return padded


def _normalize_valid_sequence_batch_inplace(
    batch: np.ndarray,
    valid_mask: np.ndarray,
    channel_specs: List[Dict[str, Any]],
    *,
    normalized_value_clamp: float,
) -> None:
    """Normalize valid timesteps in-place using grouped channel specs."""
    if not channel_specs or not bool(valid_mask.any()):
        return

    valid_rows = batch[valid_mask]
    for spec in channel_specs:
        cols = spec["indices"]
        valid_rows[:, cols] = DataNormalizer.normalize_array(
            valid_rows[:, cols],
            spec["method"],
            spec["stats"],
            normalized_value_clamp=normalized_value_clamp,
        )
    batch[valid_mask] = valid_rows


def _normalize_global_batch_inplace(
    batch: np.ndarray,
    channel_specs: List[Dict[str, Any]],
    *,
    normalized_value_clamp: float,
) -> None:
    """Normalize per-profile global features in-place with grouped specs."""
    if not channel_specs or batch.size == 0:
        return

    for spec in channel_specs:
        cols = spec["indices"]
        batch[:, cols] = DataNormalizer.normalize_array(
            batch[:, cols],
            spec["method"],
            spec["stats"],
            normalized_value_clamp=normalized_value_clamp,
        )


def _assert_finite_processed_array(
    array: Optional[np.ndarray],
    *,
    array_name: str,
    split_key: str,
    file_stem: str,
    chunk_start: int,
    chunk_stop: int,
) -> None:
    """Hard-fail when preprocessing produces non-finite values."""
    if array is None:
        return

    finite_mask = np.isfinite(array)
    if bool(finite_mask.all()):
        return

    bad_count = int((~finite_mask).sum())
    raise ValueError(
        f"Non-finite values detected in processed {array_name} for split '{split_key}' "
        f"from file '{file_stem}' chunk [{chunk_start}, {chunk_stop}) "
        f"({bad_count} values)."
    )


def _load_and_restore_chunk(
    hf_file: h5py.File,
    variables: List[str],
    indices: np.ndarray,
    *,
    expected_ndim: int,
    max_block_bytes: int,
    max_span_multiplier: int,
    pad_comparison_epsilon: float,
    max_seq_len: Optional[int] = None,
    padding_value: Optional[float] = None,
    strictly_positive_vars: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load a chunk of data for specified variables and indices.

    Uses sort/unsort technique for efficient HDF5 reading while preserving order.
    Uses guarded contiguous block reads to avoid slow HDF5 fancy indexing.

    Args:
        hf_file: Open HDF5 file handle
        variables: List of variable names to load
        indices: Array of indices to load
        expected_ndim: Required dataset rank for all variables in this call
        max_block_bytes: Maximum contiguous read size for guarded block reads
        max_span_multiplier: Maximum read-span multiplier for guarded block reads
        pad_comparison_epsilon: Tolerance for padding sentinel detection
        max_seq_len: Optional maximum sequence length (2D arrays only)
        padding_value: Optional padding sentinel to reject in raw, non-padding data

    Returns:
        Dictionary of variable_name -> data array
    """
    if indices.size == 0:
        return {}

    # Sort indices for more efficient HDF5 access
    sorter = np.argsort(indices)
    sorted_indices = indices[sorter]

    # Inverse permutation to restore original order
    inverse_sorter = np.empty_like(sorter)
    inverse_sorter[sorter] = np.arange(sorter.size)

    data_chunk: Dict[str, np.ndarray] = {}

    for var in variables:
        if var not in hf_file:
            raise KeyError(f"Critical error: Variable '{var}' not found in HDF5 file.")

        ds = hf_file[var]
        if ds.ndim != expected_ndim:
            raise ValueError(
                f"Variable '{var}' expected rank {expected_ndim}, got rank {ds.ndim}."
            )

        # Guarded contiguous block read: fast when indices are reasonably dense
        start = int(sorted_indices[0])
        end = int(sorted_indices[-1]) + 1
        span = end - start

        row_elems = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
        est_bytes = span * row_elems * ds.dtype.itemsize

        if (
            span <= max_span_multiplier * int(sorted_indices.size)
            and est_bytes <= max_block_bytes
        ):
            block = ds[start:end]
            data_sorted = block[sorted_indices - start]
        else:
            data_sorted = ds[sorted_indices]

        # Restore original order
        data_orig_order = data_sorted[inverse_sorter]

        # Validate finiteness and sentinel safety on raw values.
        if not np.isfinite(data_orig_order).all():
            raise ValueError(f"Non-finite values detected in variable '{var}'.")
        if (
            padding_value is not None
            and np.any(np.abs(data_orig_order - padding_value) <= pad_comparison_epsilon)
        ):
            raise ValueError(
                f"Padding sentinel {padding_value} appears in raw variable '{var}'."
            )
        if strictly_positive_vars and var in strictly_positive_vars:
            if np.any(data_orig_order <= 0):
                raise ValueError(
                    f"Variable '{var}' must be strictly positive but contains values <= 0."
                )

        # Sequence-length policy: overflow is an error.
        if max_seq_len is not None and expected_ndim == 2:
            if data_orig_order.shape[1] > max_seq_len:
                raise ValueError(
                    f"Sequence length overflow for '{var}': "
                    f"{data_orig_order.shape[1]} > max_sequence_length={max_seq_len}."
                )

        data_chunk[var] = data_orig_order

    return data_chunk


def _save_shard(
    seq_data: np.ndarray,
    tgt_data: np.ndarray,
    glb_data: Optional[np.ndarray],
    seq_dir: Path,
    tgt_dir: Path,
    glb_dir: Optional[Path],
    shard_idx: int,
) -> None:
    """Save a shard of data to NPY files."""
    np.save(seq_dir / f"shard_{shard_idx:06d}.npy", seq_data)
    np.save(tgt_dir / f"shard_{shard_idx:06d}.npy", tgt_data)
    if glb_data is not None and glb_dir is not None:
        np.save(glb_dir / f"shard_{shard_idx:06d}.npy", glb_data)


def _save_preprocessing_summary(
    output_dir: Path,
    config: Dict[str, Any],
    splits: Dict[str, List[Tuple[str, int]]],
    norm_metadata: Dict[str, Any],
) -> None:
    """Save a human-readable summary of preprocessing results."""
    summary_path = output_dir / "preprocessing_summary.txt"
    data_spec = config["data_specification"]
    model_hp = config["model_hyperparameters"]
    misc = config["miscellaneous_settings"]
    methods = norm_metadata["normalization_methods"]
    stats = norm_metadata["per_key_stats"]

    with open(summary_path, "w") as f:
        f.write("=== Preprocessing Summary ===\n\n")
        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
        f.write("Data Hash: disabled\n\n")

        f.write("=== Configuration ===\n")
        f.write(f"Processed Data Directory: {str(output_dir)}\n")
        f.write(f"Max Sequence Length: {model_hp['max_sequence_length']}\n")
        f.write(f"Shard Size: {misc['shard_size']}\n")
        f.write(f"Padding Value: {data_spec['padding_value']}\n\n")

        f.write("=== Data Splits ===\n")
        for split_name in ("train", "validation", "test"):
            f.write(f"{split_name}: {len(splits[split_name])} profiles\n")

        f.write("\n=== Normalization ===\n")
        for var, method in methods.items():
            f.write(f"{var}: {method}\n")

        f.write("\n=== Normalization Statistics (per_key_stats) ===\n")
        for var, stat_dict in stats.items():
            f.write(f"{var}:\n")
            for k, v in stat_dict.items():
                f.write(f"  {k}: {v}\n")


def preprocess_data(
    *,
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    splits: Dict[str, List[Tuple[str, int]]],
    processed_dir: Path,
) -> bool:
    """
    Preprocess raw HDF5 data into normalized NPY shards.

    Args:
        config: Configuration dictionary
        raw_hdf5_paths: List of raw HDF5 files
        splits: Dict with keys {"train","validation","test"} containing (file_stem, idx) tuples
        processed_dir: Output directory for processed shards

    Returns:
        True if preprocessing succeeded
    """
    preprocessing_start_time = time.time()

    # Validate splits
    required_split_keys = {"train", "validation", "test"}
    if not required_split_keys.issubset(splits.keys()):
        raise ValueError(f"Splits dict must contain keys: {sorted(required_split_keys)}")

    data_spec = config["data_specification"]
    input_vars = list(data_spec["input_variables"])
    target_vars = list(data_spec["target_variables"])
    global_vars = list(data_spec["global_variables"])
    strictly_positive_vars = set(data_spec["strictly_positive_variables"])
    max_seq_len = int(config["model_hyperparameters"]["max_sequence_length"])

    misc = config["miscellaneous_settings"]
    shard_size = int(misc["shard_size"])
    hdf5_read_chunk_size = int(misc["hdf5_read_chunk_size"])

    if hdf5_read_chunk_size >= shard_size:
        raise ValueError(
            f"hdf5_read_chunk_size ({hdf5_read_chunk_size}) must be less than "
            f"shard_size ({shard_size}) to prevent shard buffer overflow."
        )

    padding_value = float(data_spec["padding_value"])
    normalization_cfg = config["normalization"]
    normalized_value_clamp = float(normalization_cfg["normalized_value_clamp"])
    pad_comparison_epsilon = float(normalization_cfg["padding_comparison_epsilon"])
    stats_max_block_bytes = int(normalization_cfg["stats_max_block_bytes"])
    stats_max_span_multiplier = int(normalization_cfg["stats_max_span_multiplier"])
    precision = get_precision_config(config)
    processed_torch_dtype = precision["input_dtype"]
    if processed_torch_dtype not in TORCH_TO_NUMPY_DTYPE:
        raise ValueError(
            "Processed shard dtype must map to a NumPy dtype. "
            f"Unsupported precision.input_dtype={precision['input_dtype_name']!r}. "
            "Use float16/float32/float64."
        )
    processed_numpy_dtype = TORCH_TO_NUMPY_DTYPE[processed_torch_dtype]

    # Ensure output dirs exist
    if not ensure_dirs(processed_dir):
        raise RuntimeError(f"Failed to create processed data directory: {processed_dir}")

    # Map file stems to paths
    file_map = {p.stem: p for p in raw_hdf5_paths if p.is_file()}
    if not file_map:
        raise RuntimeError("No valid raw HDF5 paths provided.")

    # Validate split references and index bounds up front.
    required_vars = list(dict.fromkeys(input_vars + target_vars + global_vars))
    profile_vars = list(dict.fromkeys(input_vars + target_vars))
    file_num_rows: Dict[str, int] = {}
    for stem, path in file_map.items():
        with h5py.File(path, "r", swmr=True, libver="latest") as hf:
            missing_vars = [v for v in required_vars if v not in hf]
            if missing_vars:
                raise ValueError(f"Missing required variables in {path.name}: {missing_vars}")

            profile_shapes = {var: tuple(hf[var].shape) for var in profile_vars}
            if any(len(shape) != 2 for shape in profile_shapes.values()):
                bad = {
                    var: shape for var, shape in profile_shapes.items() if len(shape) != 2
                }
                raise ValueError(
                    "Profile variables must have shape [N, L]. "
                    f"Invalid shapes in {path.name}: {bad}"
                )
            unique_profile_shapes = sorted(set(profile_shapes.values()))
            if len(unique_profile_shapes) != 1:
                raise ValueError(
                    "All profile variables must share identical [N, L] shape. "
                    f"Found in {path.name}: {profile_shapes}"
                )
            n_profiles, seq_len = unique_profile_shapes[0]
            if seq_len <= 0:
                raise ValueError(f"Invalid sequence length L={seq_len} in {path.name}.")
            if seq_len > max_seq_len:
                raise ValueError(
                    f"Sequence length overflow in {path.name}: {seq_len} > "
                    f"max_sequence_length={max_seq_len}."
                )

            for var in global_vars:
                glb_shape = tuple(hf[var].shape)
                if len(glb_shape) != 1:
                    raise ValueError(
                        f"Global variable '{var}' must have shape [N], got {glb_shape} in {path.name}."
                    )
                if glb_shape[0] != n_profiles:
                    raise ValueError(
                        f"Global variable '{var}' leading dimension mismatch in {path.name}: "
                        f"{glb_shape[0]} != {n_profiles}."
                    )

            file_num_rows[stem] = int(n_profiles)

    for split_key in ("train", "validation", "test"):
        for stem, idx in splits[split_key]:
            if stem not in file_map:
                raise KeyError(f"Unknown file stem '{stem}' in split '{split_key}'.")
            if not isinstance(idx, int) or idx < 0 or idx >= file_num_rows[stem]:
                raise ValueError(
                    f"Out-of-range index in split '{split_key}': ({stem}, {idx}), "
                    f"valid range is [0, {file_num_rows[stem]})."
                )

    # Compute normalization stats from training split
    normalizer = DataNormalizer(config_data=config)

    logger.info("Calculating normalization statistics from training split...")
    stats_start_time = time.perf_counter()
    norm_metadata = normalizer.calculate_stats(raw_hdf5_paths, splits["train"])
    stats_elapsed_s = time.perf_counter() - stats_start_time

    # Save normalization metadata
    if not save_json(norm_metadata, processed_dir / "normalization_metadata.json"):
        raise RuntimeError("Failed to save normalization_metadata.json.")

    norm_methods: Dict[str, str] = norm_metadata["normalization_methods"]
    norm_stats: Dict[str, Dict[str, Any]] = norm_metadata["per_key_stats"]
    input_norm_specs = _build_channel_normalization_groups(
        input_vars,
        norm_methods,
        norm_stats,
        dtype=np.dtype(processed_numpy_dtype),
    )
    target_norm_specs = _build_channel_normalization_groups(
        target_vars,
        norm_methods,
        norm_stats,
        dtype=np.dtype(processed_numpy_dtype),
    )
    global_norm_specs = _build_channel_normalization_groups(
        global_vars,
        norm_methods,
        norm_stats,
        dtype=np.dtype(processed_numpy_dtype),
    )
    timing_totals = {
        "read_s": 0.0,
        "normalize_s": 0.0,
        "write_s": 0.0,
    }

    # Create split directories (validation -> 'val' on disk)
    split_dir_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for split_key, split_dirname in split_dir_map.items():
        split_dir = processed_dir / split_dirname
        if not ensure_dirs(split_dir, split_dir / "sequence_inputs", split_dir / "targets"):
            raise RuntimeError(f"Failed to create split directories for '{split_key}'.")
        if global_vars:
            if not ensure_dirs(split_dir / "globals"):
                raise RuntimeError(f"Failed to create globals directory for '{split_key}'.")

    # Process each split and write shards
    for split_key, split_dirname in split_dir_map.items():
        split_indices = splits[split_key]
        split_dir = processed_dir / split_dirname

        seq_dir = split_dir / "sequence_inputs"
        tgt_dir = split_dir / "targets"
        glb_dir = split_dir / "globals" if global_vars else None

        total_samples = len(split_indices)
        num_shards = (total_samples + shard_size - 1) // shard_size if total_samples > 0 else 0

        split_metadata = {
            "split": split_dirname,
            "total_samples": total_samples,
            "shard_size": shard_size,
            "num_shards": num_shards,
            "sequence_length": max_seq_len,
            "has_globals": bool(global_vars),
        }
        if not save_json(split_metadata, split_dir / "metadata.json"):
            raise RuntimeError(f"Failed to save split metadata for '{split_key}'.")

        # Group indices by file stem to reduce file open churn
        grouped_indices: Dict[str, List[int]] = {}
        for stem, idx in split_indices:
            grouped_indices.setdefault(stem, []).append(int(idx))
        for stem in grouped_indices:
            grouped_indices[stem].sort()

        # Allocate shard buffers; initialize sequence/target with padding_value
        n_in = len(input_vars)
        n_tgt = len(target_vars)
        n_glb = len(global_vars)

        seq_shard = np.full(
            (shard_size, max_seq_len, n_in),
            padding_value,
            dtype=processed_numpy_dtype,
        )
        tgt_shard = np.full(
            (shard_size, max_seq_len, n_tgt),
            padding_value,
            dtype=processed_numpy_dtype,
        )
        glb_shard = (
            np.zeros((shard_size, n_glb), dtype=processed_numpy_dtype)
            if global_vars else None
        )

        write_pos = 0
        shard_idx = 0

        logger.info(f"Processing split '{split_key}' -> '{split_dirname}' ({total_samples} profiles)...")

        # Precompute static variable lists outside the inner loop.
        seq_vars = list(dict.fromkeys(input_vars + target_vars))

        with tqdm(total=total_samples, desc=f"Processing {split_dirname}") as pbar:
            for file_stem, indices in grouped_indices.items():
                if file_stem not in file_map:
                    raise KeyError(f"Unknown file stem '{file_stem}' encountered during preprocessing.")

                with h5py.File(file_map[file_stem], "r", swmr=True, libver="latest") as hf_raw:
                    idx_arr = np.asarray(indices, dtype=np.int64)

                    for i in range(0, len(idx_arr), hdf5_read_chunk_size):
                        chunk_idx = idx_arr[i:i + hdf5_read_chunk_size]
                        if chunk_idx.size == 0:
                            continue

                        read_start = time.perf_counter()
                        # Load sequence-like variables (inputs + targets) in one pass
                        seq_data = _load_and_restore_chunk(
                            hf_raw,
                            seq_vars,
                            chunk_idx,
                            expected_ndim=2,
                            max_block_bytes=stats_max_block_bytes,
                            max_span_multiplier=stats_max_span_multiplier,
                            pad_comparison_epsilon=pad_comparison_epsilon,
                            max_seq_len=max_seq_len,
                            padding_value=padding_value,
                            strictly_positive_vars=strictly_positive_vars,
                        )

                        # Load globals separately (typically 1D)
                        if global_vars:
                            global_data = _load_and_restore_chunk(
                                hf_raw,
                                global_vars,
                                chunk_idx,
                                expected_ndim=1,
                                max_block_bytes=stats_max_block_bytes,
                                max_span_multiplier=stats_max_span_multiplier,
                                pad_comparison_epsilon=pad_comparison_epsilon,
                                max_seq_len=None,
                                padding_value=padding_value,
                                strictly_positive_vars=strictly_positive_vars,
                            )
                        else:
                            global_data = {}
                        timing_totals["read_s"] += time.perf_counter() - read_start

                        normalize_start = time.perf_counter()
                        # Stack into arrays
                        seq_in_np = np.stack(
                            [seq_data[var] for var in input_vars],
                            axis=-1,
                        ).astype(processed_numpy_dtype, copy=False)
                        tgt_np = np.stack(
                            [seq_data[var] for var in target_vars],
                            axis=-1,
                        ).astype(processed_numpy_dtype, copy=False)

                        glb_np = (
                            np.stack([global_data[var] for var in global_vars], axis=-1).astype(
                                processed_numpy_dtype,
                                copy=False,
                            )
                            if global_vars
                            else None
                        )

                        seq_in_np = _right_pad_sequence_batch(
                            seq_in_np,
                            max_seq_len=max_seq_len,
                            padding_value=padding_value,
                        )
                        tgt_np = _right_pad_sequence_batch(
                            tgt_np,
                            max_seq_len=max_seq_len,
                            padding_value=padding_value,
                        )

                        # Compute padding mask once for all sequence channels.
                        # Per spec §4.2, padding is per-timestep (all features padded together),
                        # so one channel suffices for the mask.
                        pad_mask = np.isfinite(seq_in_np[:, :, 0]) & (
                            np.abs(seq_in_np[:, :, 0] - padding_value) <= pad_comparison_epsilon
                        )
                        valid_mask = ~pad_mask
                        _normalize_valid_sequence_batch_inplace(
                            seq_in_np,
                            valid_mask,
                            input_norm_specs,
                            normalized_value_clamp=normalized_value_clamp,
                        )
                        _normalize_valid_sequence_batch_inplace(
                            tgt_np,
                            valid_mask,
                            target_norm_specs,
                            normalized_value_clamp=normalized_value_clamp,
                        )

                        if glb_np is not None:
                            _normalize_global_batch_inplace(
                                glb_np,
                                global_norm_specs,
                                normalized_value_clamp=normalized_value_clamp,
                            )
                        _assert_finite_processed_array(
                            seq_in_np,
                            array_name="sequence inputs",
                            split_key=split_key,
                            file_stem=file_stem,
                            chunk_start=int(chunk_idx[0]),
                            chunk_stop=int(chunk_idx[-1]) + 1,
                        )
                        _assert_finite_processed_array(
                            tgt_np,
                            array_name="targets",
                            split_key=split_key,
                            file_stem=file_stem,
                            chunk_start=int(chunk_idx[0]),
                            chunk_stop=int(chunk_idx[-1]) + 1,
                        )
                        _assert_finite_processed_array(
                            glb_np,
                            array_name="global features",
                            split_key=split_key,
                            file_stem=file_stem,
                            chunk_start=int(chunk_idx[0]),
                            chunk_stop=int(chunk_idx[-1]) + 1,
                        )
                        timing_totals["normalize_s"] += time.perf_counter() - normalize_start

                        # Write into shard buffers
                        write_start = time.perf_counter()
                        batch_n = int(seq_in_np.shape[0])
                        start_pos = write_pos
                        end_pos = write_pos + batch_n

                        if end_pos <= shard_size:
                            seq_shard[start_pos:end_pos] = seq_in_np
                            tgt_shard[start_pos:end_pos] = tgt_np
                            if glb_shard is not None and glb_np is not None:
                                glb_shard[start_pos:end_pos] = glb_np
                            write_pos = end_pos
                            pbar.update(batch_n)
                        else:
                            # Fill remainder of current shard
                            first_n = shard_size - start_pos
                            if first_n > 0:
                                seq_shard[start_pos:shard_size] = seq_in_np[:first_n]
                                tgt_shard[start_pos:shard_size] = tgt_np[:first_n]
                                if glb_shard is not None and glb_np is not None:
                                    glb_shard[start_pos:shard_size] = glb_np[:first_n]

                            # Save full shard
                            _save_shard(seq_shard, tgt_shard, glb_shard, seq_dir, tgt_dir, glb_dir, shard_idx)
                            shard_idx += 1

                            # Reset buffers
                            seq_shard.fill(padding_value)
                            tgt_shard.fill(padding_value)
                            if glb_shard is not None:
                                glb_shard.fill(0.0)

                            # Write remaining portion into new shard
                            remaining = batch_n - first_n
                            if remaining > 0:
                                seq_shard[0:remaining] = seq_in_np[first_n:]
                                tgt_shard[0:remaining] = tgt_np[first_n:]
                                if glb_shard is not None and glb_np is not None:
                                    glb_shard[0:remaining] = glb_np[first_n:]

                            write_pos = remaining
                            pbar.update(batch_n)
                        timing_totals["write_s"] += time.perf_counter() - write_start

        # Save final partial shard if any
        if write_pos > 0:
            _save_shard(
                seq_shard[:write_pos].copy(),
                tgt_shard[:write_pos].copy(),
                glb_shard[:write_pos].copy() if glb_shard is not None else None,
                seq_dir,
                tgt_dir,
                glb_dir,
                shard_idx,
            )

    # Save summary and hash
    _save_preprocessing_summary(processed_dir, config, splits, norm_metadata)

    total_time = time.time() - preprocessing_start_time
    logger.info(f"Preprocessing completed successfully in {total_time:.2f}s.")
    logger.info(
        "Preprocessing timings: stats=%.2fs read=%.2fs normalize=%.2fs write=%.2fs",
        stats_elapsed_s,
        timing_totals["read_s"],
        timing_totals["normalize_s"],
        timing_totals["write_s"],
    )

    return True


__all__ = ["preprocess_data"]
