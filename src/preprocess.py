#!/usr/bin/env python3
"""
preprocess.py - Preprocess raw HDF5 data into normalized NPY shards.

This module:
- Computes normalization statistics from the training split
- Normalizes raw HDF5 data into fixed-length NPY shards for train/val/test
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
    PADDING_VALUE,
    compute_data_hash_with_stats,
    ensure_dirs,
    save_json,
)

logger = logging.getLogger(__name__)

_TRUNCATION_WARNED: set[str] = set()  # warn once per variable for truncation

# Processing constants
HDF5_READ_CHUNK_SIZE = 8192
_MAX_BLOCK_BYTES = 512 * 1024 * 1024  # 512 MiB
_MAX_SPAN_MULTIPLIER = 16             # span <= multiplier * n_indices


def _load_and_restore_chunk(
    hf_file: h5py.File,
    variables: List[str],
    indices: np.ndarray,
    max_seq_len: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a chunk of data for specified variables and indices.

    Uses sort/unsort technique for efficient HDF5 reading while preserving order.
    Uses guarded contiguous block reads to avoid slow HDF5 fancy indexing.

    Args:
        hf_file: Open HDF5 file handle
        variables: List of variable names to load
        indices: Array of indices to load
        max_seq_len: Optional maximum sequence length for truncation (2D arrays only)

    Returns:
        Dictionary of variable_name -> data array, or None if error
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
            logger.error(f"Critical error: Variable '{var}' not found in HDF5 file.")
            return None

        ds = hf_file[var]

        # Guarded contiguous block read: fast when indices are reasonably dense
        start = int(sorted_indices[0])
        end = int(sorted_indices[-1]) + 1
        span = end - start

        row_elems = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
        est_bytes = span * row_elems * ds.dtype.itemsize

        if span <= _MAX_SPAN_MULTIPLIER * int(sorted_indices.size) and est_bytes <= _MAX_BLOCK_BYTES:
            block = ds[start:end]
            data_sorted = block[sorted_indices - start]
        else:
            data_sorted = ds[sorted_indices]

        # Restore original order
        data_orig_order = data_sorted[inverse_sorter]

        # Optional truncation (sequence variables)
        if max_seq_len is not None and data_orig_order.ndim == 2:
            if data_orig_order.shape[1] > max_seq_len:
                if var not in _TRUNCATION_WARNED:
                    logger.warning(
                        f"Truncating sequence from {data_orig_order.shape[1]} to {max_seq_len} "
                        f"for variable '{var}'"
                    )
                    _TRUNCATION_WARNED.add(var)
                data_orig_order = data_orig_order[:, :max_seq_len]

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
    data_hash: str,
) -> None:
    """Save a human-readable summary of preprocessing results."""
    summary_path = output_dir / "preprocessing_summary.txt"

    # Best-effort: summary should never crash preprocessing
    try:
        data_spec = config.get("data_specification", {})
        model_hp = config.get("model_hyperparameters", {})
        misc = config.get("miscellaneous_settings", {})

        with open(summary_path, "w") as f:
            f.write("=== Preprocessing Summary ===\n\n")
            f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Data Hash: {data_hash}\n\n")

            f.write("=== Configuration ===\n")
            f.write(f"Processed Data Directory: {str(output_dir)}\n")
            f.write(f"Max Sequence Length: {model_hp.get('max_sequence_length')}\n")
            f.write(f"Shard Size: {misc.get('shard_size')}\n")
            f.write(f"Padding Value: {data_spec.get('padding_value', PADDING_VALUE)}\n\n")

            f.write("=== Data Splits ===\n")
            for split_name in ("train", "validation", "test"):
                if split_name in splits:
                    f.write(f"{split_name}: {len(splits[split_name])} profiles\n")

            f.write("\n=== Normalization ===\n")
            methods = norm_metadata.get("normalization_methods", {})
            for var, method in methods.items():
                f.write(f"{var}: {method}\n")

            f.write("\n=== Normalization Statistics (per_key_stats) ===\n")
            stats = norm_metadata.get("per_key_stats", {})
            for var, stat_dict in stats.items():
                f.write(f"{var}:\n")
                for k, v in stat_dict.items():
                    f.write(f"  {k}: {v}\n")
    except Exception as e:
        logger.debug(f"Failed to write preprocessing summary: {e}")


def _normalize_channel_preserve_padding(
    x: torch.Tensor,
    *,
    method: str,
    stats: Dict[str, Any],
    padding_value: float,
) -> torch.Tensor:
    """
    Normalize a tensor while preserving padding_value exactly at padding positions.
    """
    if method in ("none", "bool") or not stats:
        return x

    # Padding mask computed on the unnormalized values
    pad_mask = torch.isfinite(x) & (torch.abs(x - padding_value) <= 1e-6)

    y = DataNormalizer.normalize_tensor(x, method, stats)

    if pad_mask.any():
        # avoid in-place modification side effects if y is a view
        y = y.clone()
        y[pad_mask] = padding_value

    return y


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
        True if preprocessing succeeded, False otherwise
    """
    preprocessing_start_time = time.time()

    # Validate splits
    required_split_keys = {"train", "validation", "test"}
    if not required_split_keys.issubset(splits.keys()):
        logger.error(f"Splits dict must contain keys: {sorted(required_split_keys)}")
        return False

    data_spec = config.get("data_specification", {})
    if not isinstance(data_spec, dict):
        logger.error("Config missing 'data_specification' section.")
        return False

    input_vars = list(data_spec.get("input_variables", []))
    target_vars = list(data_spec.get("target_variables", []))
    global_vars = list(data_spec.get("global_variables", []))

    if not input_vars or not target_vars:
        logger.error("Config must specify non-empty input_variables and target_variables.")
        return False

    model_hp = config.get("model_hyperparameters", {})
    if "max_sequence_length" not in model_hp:
        logger.error("Config missing model_hyperparameters.max_sequence_length.")
        return False
    max_seq_len = int(model_hp["max_sequence_length"])

    misc = config.get("miscellaneous_settings", {})
    shard_size = int(misc.get("shard_size", 8192))
    hdf5_read_chunk_size = int(misc.get("hdf5_read_chunk_size", HDF5_READ_CHUNK_SIZE))

    padding_value = float(data_spec.get("padding_value", PADDING_VALUE))

    # Ensure output dirs exist
    ensure_dirs(processed_dir)

    # Compute hash and short-circuit if unchanged
    try:
        current_hash = compute_data_hash_with_stats(config, raw_hdf5_paths)
    except Exception as e:
        logger.exception(f"Failed to compute data hash: {e}")
        return False

    hash_path = processed_dir / ".preprocess_hash"
    if hash_path.exists():
        old_hash = hash_path.read_text().strip()
        if old_hash == current_hash:
            logger.info("Preprocessing already up to date (hash match). Skipping.")
            return True

    # Map file stems to paths
    file_map = {p.stem: p for p in raw_hdf5_paths if p.is_file()}
    if not file_map:
        logger.error("No valid raw HDF5 paths provided.")
        return False

    # Compute normalization stats from training split
    normalizer = DataNormalizer(config_data=config)

    logger.info("Calculating normalization statistics from training split...")
    try:
        norm_metadata = normalizer.calculate_stats(raw_hdf5_paths, splits["train"])
    except Exception as e:
        logger.exception(f"Failed to calculate normalization stats: {e}")
        return False

    # Save normalization metadata
    if not save_json(norm_metadata, processed_dir / "normalization_metadata.json"):
        logger.warning("Failed to save normalization_metadata.json")

    norm_methods: Dict[str, str] = norm_metadata.get("normalization_methods", {})
    norm_stats: Dict[str, Dict[str, Any]] = norm_metadata.get("per_key_stats", {})

    # Create split directories (validation -> 'val' on disk)
    split_dir_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    for split_key, split_dirname in split_dir_map.items():
        split_dir = processed_dir / split_dirname
        ensure_dirs(split_dir)
        ensure_dirs(split_dir / "sequence_inputs")
        ensure_dirs(split_dir / "targets")
        if global_vars:
            ensure_dirs(split_dir / "globals")

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
        save_json(split_metadata, split_dir / "metadata.json")

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

        seq_shard = np.full((shard_size, max_seq_len, n_in), padding_value, dtype=np.float32)
        tgt_shard = np.full((shard_size, max_seq_len, n_tgt), padding_value, dtype=np.float32)
        glb_shard = np.zeros((shard_size, n_glb), dtype=np.float32) if global_vars else None

        write_pos = 0
        shard_idx = 0

        logger.info(f"Processing split '{split_key}' -> '{split_dirname}' ({total_samples} profiles)...")

        with tqdm(total=total_samples, desc=f"Processing {split_dirname}") as pbar:
            for file_stem, indices in grouped_indices.items():
                if file_stem not in file_map:
                    logger.warning(f"Skipping unknown file stem '{file_stem}'.")
                    pbar.update(len(indices))
                    continue

                with h5py.File(file_map[file_stem], "r", swmr=True, libver="latest") as hf_raw:
                    idx_arr = np.asarray(indices, dtype=np.int64)

                    for i in range(0, len(idx_arr), hdf5_read_chunk_size):
                        chunk_idx = idx_arr[i:i + hdf5_read_chunk_size]
                        if chunk_idx.size == 0:
                            continue

                        # Load sequence-like variables (inputs + targets) in one pass
                        seq_vars = list(dict.fromkeys(input_vars + target_vars))
                        seq_data = _load_and_restore_chunk(hf_raw, seq_vars, chunk_idx, max_seq_len=max_seq_len)
                        if seq_data is None:
                            return False

                        # Load globals separately (typically 1D)
                        if global_vars:
                            global_data = _load_and_restore_chunk(hf_raw, global_vars, chunk_idx, max_seq_len=None)
                            if global_data is None:
                                return False
                        else:
                            global_data = {}

                        # Stack into arrays
                        seq_in_np = np.stack([seq_data[var] for var in input_vars], axis=-1)
                        tgt_np = np.stack([seq_data[var] for var in target_vars], axis=-1)

                        glb_np = (
                            np.stack([global_data[var] for var in global_vars], axis=-1)
                            if global_vars
                            else None
                        )

                        seq_in = torch.from_numpy(seq_in_np).float()
                        tgt = torch.from_numpy(tgt_np).float()
                        glb = torch.from_numpy(glb_np).float() if glb_np is not None else None

                        # Ensure fixed length by right-padding with padding_value (not zeros)
                        cur_len = int(seq_in.shape[1])
                        if cur_len < max_seq_len:
                            pad_width = max_seq_len - cur_len
                            seq_in = torch.nn.functional.pad(seq_in, (0, 0, 0, pad_width), value=padding_value)
                            tgt = torch.nn.functional.pad(tgt, (0, 0, 0, pad_width), value=padding_value)

                        # Normalize per-channel, preserving padding values
                        for j, var in enumerate(input_vars):
                            method = str(norm_methods.get(var, "none"))
                            stats = norm_stats.get(var, {})
                            seq_in[:, :, j] = _normalize_channel_preserve_padding(
                                seq_in[:, :, j], method=method, stats=stats, padding_value=padding_value
                            )

                        for j, var in enumerate(target_vars):
                            method = str(norm_methods.get(var, "none"))
                            stats = norm_stats.get(var, {})
                            tgt[:, :, j] = _normalize_channel_preserve_padding(
                                tgt[:, :, j], method=method, stats=stats, padding_value=padding_value
                            )

                        if global_vars and glb is not None:
                            for j, var in enumerate(global_vars):
                                method = str(norm_methods.get(var, "none"))
                                stats = norm_stats.get(var, {})
                                glb[:, j] = _normalize_channel_preserve_padding(
                                    glb[:, j], method=method, stats=stats, padding_value=padding_value
                                )

                        # Write into shard buffers
                        batch_n = int(seq_in.shape[0])
                        start_pos = write_pos
                        end_pos = write_pos + batch_n

                        seq_in_np2 = seq_in.numpy().astype(np.float32, copy=False)
                        tgt_np2 = tgt.numpy().astype(np.float32, copy=False)
                        glb_np2 = glb.numpy().astype(np.float32, copy=False) if glb is not None else None

                        if end_pos <= shard_size:
                            seq_shard[start_pos:end_pos] = seq_in_np2
                            tgt_shard[start_pos:end_pos] = tgt_np2
                            if glb_shard is not None and glb_np2 is not None:
                                glb_shard[start_pos:end_pos] = glb_np2
                            write_pos = end_pos
                            pbar.update(batch_n)
                        else:
                            # Fill remainder of current shard
                            first_n = shard_size - start_pos
                            if first_n > 0:
                                seq_shard[start_pos:shard_size] = seq_in_np2[:first_n]
                                tgt_shard[start_pos:shard_size] = tgt_np2[:first_n]
                                if glb_shard is not None and glb_np2 is not None:
                                    glb_shard[start_pos:shard_size] = glb_np2[:first_n]

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
                                seq_shard[0:remaining] = seq_in_np2[first_n:]
                                tgt_shard[0:remaining] = tgt_np2[first_n:]
                                if glb_shard is not None and glb_np2 is not None:
                                    glb_shard[0:remaining] = glb_np2[first_n:]

                            write_pos = remaining
                            pbar.update(batch_n)

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
    _save_preprocessing_summary(processed_dir, config, splits, norm_metadata, current_hash)
    hash_path.write_text(current_hash)

    total_time = time.time() - preprocessing_start_time
    logger.info(f"Preprocessing completed successfully in {total_time:.2f}s.")

    return True


__all__ = ["preprocess_data"]
