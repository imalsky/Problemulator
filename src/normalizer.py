#!/usr/bin/env python3
"""GPU-aware data normalization with numerical stability."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from hardware import setup_device
from utils import get_precision_config

logger = logging.getLogger(__name__)

class DataNormalizer:
    """
    Handles data normalization with multiple methods and numerical stability.

    Supports: standard, log-standard, iqr, log-min-max, max-out,
    signed-log, scaled_signed_offset_log, symlog, bool, none.
    """

    METHODS = {
        "iqr", "log-min-max", "max-out", "signed-log",
        "scaled_signed_offset_log", "symlog", "standard",
        "log-standard", "bool", "none",
    }

    QUANTILE_METHODS = {"iqr", "symlog"}

    def __init__(self, *, config_data: Dict[str, Any]):
        """Initialize normalizer with configuration."""
        self.config = config_data
        precision = get_precision_config(self.config)

        backend = str(self.config["miscellaneous_settings"]["device_backend"])
        self.device = setup_device(backend)
        self.dtype = precision["stats_dtype"]
        self.norm_config = self.config["normalization"]
        self.eps = float(self.norm_config["epsilon"])
        self.keys_to_process, self.key_methods = self._get_keys_and_methods()
        self._approximated_quantile_keys = set()

        # Cached config values
        self._quantile_memory_limit = int(self.norm_config["quantile_max_values_in_memory"])
        self._stats_chunk_size = int(self.norm_config["stats_chunk_size"])
        self._stats_max_block_bytes = int(self.norm_config["stats_max_block_bytes"])
        self._stats_max_span_multiplier = int(self.norm_config["stats_max_span_multiplier"])
        self._pad_comparison_epsilon = float(self.norm_config["padding_comparison_epsilon"])
        self._symlog_min_threshold_multiplier = float(
            self.norm_config["symlog_min_threshold_multiplier"]
        )
        self._log_min_max_clamp_min = float(self.norm_config["log_min_max_clamp_min"])
        self._log_min_max_clamp_max = float(self.norm_config["log_min_max_clamp_max"])

        logger.info(f"DataNormalizer initialized on device '{self.device}'.")

    def _get_keys_and_methods(self) -> Tuple[Set[str], Dict[str, str]]:
        """Extract variables and their normalization methods from config."""
        spec = self.config["data_specification"]
        all_vars = set()

        for key in ["input_variables", "global_variables", "target_variables"]:
            all_vars.update(spec[key])

        user_key_methods = self.norm_config["key_methods"]
        if not isinstance(user_key_methods, dict):
            raise ValueError("normalization.key_methods must be a dictionary.")

        key_methods = {}
        for key in all_vars:
            if key not in user_key_methods:
                raise ValueError(
                    f"Missing explicit normalization method for variable '{key}'."
                )
            method = str(user_key_methods[key]).lower()
            if method not in self.METHODS:
                raise ValueError(f"Unsupported method '{method}' for key '{key}'.")
            key_methods[key] = method

        return all_vars, key_methods

    def calculate_stats(
            self, raw_hdf5_paths: List[Path], train_indices: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """Calculate normalization statistics from training data."""
        logger.info(
            f"Starting statistics calculation from {len(train_indices)} training samples..."
        )
        start_time = time.time()

        if not train_indices:
            raise ValueError("Cannot calculate statistics from empty training indices.")

        file_map = {path.stem: path for path in raw_hdf5_paths if path.is_file()}
        if not file_map:
            raise RuntimeError("No valid HDF5 files found.")

        available_keys = self._get_available_keys(file_map)
        keys_to_load = self.keys_to_process.intersection(available_keys)

        if len(keys_to_load) != len(self.keys_to_process):
            missing = sorted(self.keys_to_process - available_keys)
            raise KeyError(f"Required variables missing from HDF5 files: {missing}")

        accumulators = self._initialize_accumulators(keys_to_load)

        dataset_metadata = {
            "total_train_profiles": len(train_indices),
            "variables": list(keys_to_load),
            "sequence_lengths": {
                key: {"min": float("inf"), "max": float("-inf")} for key in keys_to_load
            },
        }

        grouped_indices = self._group_indices_by_file(train_indices)

        for file_stem, indices in grouped_indices.items():
            if file_stem not in file_map:
                raise KeyError(f"Unknown file stem '{file_stem}' in training indices.")

            h5_path = file_map[file_stem]
            with h5py.File(h5_path, "r", swmr=True, libver="latest") as hf:
                missing_in_file = sorted(k for k in keys_to_load if k not in hf)
                if missing_in_file:
                    raise KeyError(
                        f"Required variables missing in file '{h5_path.name}': {missing_in_file}"
                    )

                idx_arr = np.asarray(indices, dtype=np.int64)
                idx_arr.sort()
                num_chunks = (len(idx_arr) + self._stats_chunk_size - 1) // self._stats_chunk_size

                for i in tqdm(
                        range(0, len(idx_arr), self._stats_chunk_size),
                        desc=f"Stats for {file_stem}",
                        total=num_chunks,
                        leave=False,
                ):
                    chunk_indices = idx_arr[i:i + self._stats_chunk_size]
                    if chunk_indices.size == 0:
                        continue

                    # chunk_indices is already sorted (sliced from sorted idx_arr).
                    start = int(chunk_indices[0])
                    end = int(chunk_indices[-1]) + 1
                    span = end - start

                    # Group variables by rank to batch HDF5 reads.
                    vars_by_rank: Dict[int, List[str]] = {}
                    for key in keys_to_load:
                        rank = hf[key].ndim
                        vars_by_rank.setdefault(rank, []).append(key)

                    batch_data: Dict[str, Tensor] = {}
                    for _, var_list in vars_by_rank.items():
                        for key in var_list:
                            ds = hf[key]
                            row_elems = int(np.prod(ds.shape[1:])) if ds.ndim > 1 else 1
                            est_bytes = span * row_elems * ds.dtype.itemsize

                            if (
                                span <= self._stats_max_span_multiplier * int(chunk_indices.size)
                                and est_bytes <= self._stats_max_block_bytes
                            ):
                                block = ds[start:end]
                                data_chunk_np = block[chunk_indices - start]
                            else:
                                data_chunk_np = ds[chunk_indices]

                            batch_data[key] = torch.as_tensor(
                                data_chunk_np, device=self.device, dtype=self.dtype
                            )

                    # Update all accumulators in one call per chunk.
                    if batch_data:
                        self._update_accumulators_with_batch(
                            batch_data,
                            accumulators,
                            dataset_metadata,
                            source_context=f"file={h5_path.name}, chunk_start={start}, chunk_end={end}",
                        )

        logger.info(f"Finished statistics calculation in {time.time() - start_time:.2f}s.")

        computed_stats = self._finalize_stats(accumulators)
        if not computed_stats:
            raise RuntimeError("No statistics computed due to invalid data.")

        metadata = {
            "normalization_methods": self.key_methods,
            "per_key_stats": computed_stats,
        }

        for key in dataset_metadata["sequence_lengths"]:
            min_len = dataset_metadata["sequence_lengths"][key]["min"]
            max_len = dataset_metadata["sequence_lengths"][key]["max"]
            if min_len == float("inf"):
                dataset_metadata["sequence_lengths"][key] = None
            else:
                dataset_metadata["sequence_lengths"][key] = {
                    "min": int(min_len),
                    "max": int(max_len),
                }

        metadata["dataset_metadata"] = dataset_metadata

        logger.info("Statistics calculation complete.")
        return metadata

    def _get_available_keys(self, file_map: Dict[str, Path]) -> Set[str]:
        """Get all available keys across HDF5 files."""
        available = set()
        for path in file_map.values():
            with h5py.File(path, "r") as hf:
                available.update(hf.keys())
        return available

    def _group_indices_by_file(
            self, indices: List[Tuple[str, int]]
    ) -> Dict[str, List[int]]:
        """Group indices by file for efficient loading."""
        grouped = {}
        for file_stem, idx in indices:
            grouped.setdefault(file_stem, []).append(idx)
        return grouped

    def _initialize_accumulators(
            self, keys_to_process: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Initialize statistics accumulators for each variable."""
        accumulators = {}
        memory_limit = int(self._quantile_memory_limit)

        for key in keys_to_process:
            method = self.key_methods[key]
            if method in ("none", "bool"):
                continue

            acc: Dict[str, Any] = {}

            if method in ("standard", "log-standard", "signed-log"):
                acc.update({
                    "count": 0,
                    "mean": torch.tensor(0.0, dtype=self.dtype, device=self.device),
                    "m2": torch.tensor(0.0, dtype=self.dtype, device=self.device),
                })

            if method in self.QUANTILE_METHODS:
                # Fixed-size reservoir buffer (prevents repeated torch.cat reallocations).
                acc["values"] = torch.empty(memory_limit, dtype=self.dtype, device=self.device)
                acc["values_filled"] = 0
                acc["total_values_seen"] = 0

            if method in ("max-out", "scaled_signed_offset_log", "log-min-max"):
                acc.update({
                    "min": torch.tensor(float("inf"), dtype=self.dtype, device=self.device),
                    "max": torch.tensor(float("-inf"), dtype=self.dtype, device=self.device),
                })

            if acc:
                accumulators[key] = acc

        return accumulators

    @staticmethod
    def _reservoir_update(
            *,
            buf: Tensor,
            filled: int,
            total_seen: int,
            vals: Tensor,
            memory_limit: int,
    ) -> Tuple[int, int]:
        """Update a fixed-size reservoir buffer with new values."""
        if vals.numel() == 0 or memory_limit <= 0:
            return filled, total_seen

        vals = vals.reshape(-1)
        n = int(vals.numel())

        # Fill reservoir up to memory_limit
        if filled < memory_limit:
            take = min(n, memory_limit - filled)
            if take > 0:
                buf[filled:filled + take].copy_(vals[:take])
                filled += take
                total_seen += take
                vals = vals[take:]
                n -= take

        # Reservoir sampling for remaining values
        if n > 0:
            # For each remaining element r (1-indexed), sample j ~ Uniform{0, ..., total_seen + r - 1}
            ar = torch.arange(1, n + 1, device=vals.device, dtype=torch.long)
            u = torch.rand(n, device=vals.device)
            j = torch.floor(u * (total_seen + ar).to(u.dtype)).to(torch.long)

            mask = j < memory_limit
            if mask.any():
                j_sel = j[mask]
                v_sel = vals[mask]

                # Fast path: direct indexed writes (duplicates are extremely rare; semantics remain sampling-correct).
                buf.index_put_((j_sel,), v_sel, accumulate=False)

            total_seen += n

        return filled, total_seen

    def _update_accumulators_with_batch(
            self,
            batch: Dict[str, Tensor],
            accumulators: Dict,
            dataset_metadata: Dict,
            *,
            source_context: str,
    ) -> None:
        """Update accumulators with a batch of data using online algorithms."""
        memory_limit = int(self._quantile_memory_limit)

        padding_value = float(self.config["data_specification"]["padding_value"])

        for key, data_batch in batch.items():
            method = self.key_methods.get(key)
            if method is None:
                raise KeyError(f"Missing normalization method for required key '{key}'.")

            # Track sequence lengths for metadata.
            if data_batch.ndim == 2:
                current_len = data_batch.shape[1]
                dataset_metadata["sequence_lengths"][key]["min"] = min(
                    dataset_metadata["sequence_lengths"][key]["min"], current_len
                )
                dataset_metadata["sequence_lengths"][key]["max"] = max(
                    dataset_metadata["sequence_lengths"][key]["max"], current_len
                )

            data = data_batch.flatten()
            if data.numel() == 0:
                raise ValueError(
                    f"Empty data encountered for key '{key}' during stats update ({source_context})."
                )

            finite_mask = torch.isfinite(data)
            if not bool(finite_mask.all()):
                bad_count = int((~finite_mask).sum().item())
                raise ValueError(
                    f"Non-finite values detected for key '{key}' ({bad_count} values) "
                    f"during stats update ({source_context})."
                )

            sentinel_mask = torch.abs(data - padding_value) <= self._pad_comparison_epsilon
            if bool(sentinel_mask.any()):
                bad_count = int(sentinel_mask.sum().item())
                raise ValueError(
                    f"Padding sentinel {padding_value} detected in raw stats input for key '{key}' "
                    f"({bad_count} values) ({source_context})."
                )

            if key not in accumulators:
                if method in ("none", "bool"):
                    continue
                raise KeyError(f"No accumulator configured for required key '{key}'.")

            key_acc = accumulators[key]

            data_for_stats = data

            if method == "log-standard":
                if torch.any(data <= 0):
                    raise ValueError(
                        f"Variable '{key}' uses log-standard but contains values <= 0 "
                        f"({source_context})."
                    )
                data_for_stats = torch.log10(data)
            elif method == "log-min-max":
                if torch.any(data <= 0):
                    raise ValueError(
                        f"Variable '{key}' uses log-min-max but contains values <= 0 "
                        f"({source_context})."
                    )
            elif method == "signed-log":
                data_for_stats = torch.sign(data) * torch.log10(
                    torch.abs(data) + 1.0
                )

            # Update online mean/variance using Welford's parallel algorithm
            if "count" in key_acc:
                n_new = data_for_stats.numel()
                if n_new > 0:
                    count_old = key_acc["count"]
                    mean_old = key_acc["mean"]
                    m2_old = key_acc["m2"]

                    batch_mean = data_for_stats.mean()
                    batch_var = torch.var(data_for_stats, unbiased=False)
                    batch_m2 = batch_var * n_new

                    delta = batch_mean - mean_old
                    count_new = count_old + n_new
                    mean_new = mean_old + delta * (n_new / count_new)
                    m2_new = (
                            m2_old + batch_m2 + delta ** 2 * count_old * n_new / count_new
                    )

                    key_acc["count"] = count_new
                    key_acc["mean"] = mean_new
                    key_acc["m2"] = m2_new

            # Collect values for quantile computation (reservoir sampling)
            if "values" in key_acc:
                filled = int(key_acc.get("values_filled", 0))
                t_prev = int(key_acc.get("total_values_seen", 0))
                filled, t_prev = self._reservoir_update(
                    buf=key_acc["values"],
                    filled=filled,
                    total_seen=t_prev,
                    vals=data,
                    memory_limit=memory_limit,
                )
                key_acc["values_filled"] = filled
                key_acc["total_values_seen"] = t_prev

                # Mark as approximated if we've seen more than memory_limit
                if t_prev > memory_limit:
                    self._approximated_quantile_keys.add(key)

            # Update min/max
            if "max" in key_acc:
                if method == "scaled_signed_offset_log":
                    log_vals = torch.sign(data) * torch.log10(
                        torch.abs(data) + 1.0
                    )
                    key_acc["min"] = torch.min(key_acc["min"], log_vals.min())
                    key_acc["max"] = torch.max(key_acc["max"], log_vals.max())
                elif method == "log-min-max":
                    log_vals = torch.log10(data)
                    key_acc["min"] = torch.min(key_acc["min"], log_vals.min())
                    key_acc["max"] = torch.max(key_acc["max"], log_vals.max())
                else:
                    key_acc["min"] = torch.min(key_acc["min"], data.min())
                    key_acc["max"] = torch.max(key_acc["max"], data.max())

    def _finalize_stats(self, accumulators: Dict) -> Dict[str, Any]:
        """Finalize statistics from accumulators with zero-variance protection."""
        final_stats = {}

        for key, method in self.key_methods.items():
            stats: Dict[str, Any] = {"method": method, "epsilon": self.eps}

            if method in ("none", "bool"):
                final_stats[key] = stats
                continue

            if key not in accumulators:
                raise ValueError(
                    f"No statistics accumulator produced for required key '{key}'."
                )

            key_acc = accumulators[key]

            # Finalize mean/variance with zero-variance protection
            if "count" in key_acc and key_acc["count"] > 1:
                mean = key_acc["mean"].item()
                variance = key_acc["m2"].item() / (key_acc["count"] - 1)
                std = max(math.sqrt(variance), self.eps)  # Prevent zero std

                if method == "standard":
                    stats.update({"mean": mean, "std": std})
                elif method == "log-standard":
                    stats.update({"log_mean": mean, "log_std": std})
                elif method == "signed-log":
                    stats.update({"mean": mean, "std": std})
            elif "count" in key_acc:
                mean = key_acc["mean"].item() if key_acc["count"] > 0 else 0.0
                if method == "standard":
                    stats.update({"mean": mean, "std": self.eps})
                elif method == "log-standard":
                    stats.update({"log_mean": mean, "log_std": self.eps})
                elif method == "signed-log":
                    stats.update({"mean": mean, "std": self.eps})

            # Compute quantile statistics
            if "values" in key_acc:
                filled = int(key_acc.get("values_filled", key_acc["values"].numel()))
                if filled > 0:
                    if key in self._approximated_quantile_keys:
                        logger.info(
                            f"Approximating quantiles for '{key}' using sample of "
                            f"{filled:,} values "
                            f"(out of {key_acc.get('total_values_seen', filled):,} total)."
                        )

                    all_values = key_acc["values"][:filled]
                    stats.update(self._compute_quantile_stats(all_values, key, method))

            # Finalize other statistics
            if method == "max-out":
                max_val = max(
                    abs(key_acc["min"].item()),
                    abs(key_acc["max"].item()),
                    self.eps  # Prevent zero max_val
                )
                stats["max_val"] = max_val

            elif method == "scaled_signed_offset_log":
                m = max(
                    abs(key_acc["min"].item()),
                    abs(key_acc["max"].item()),
                    self.eps
                )
                stats["m"] = m

            elif method == "log-min-max" and "min" in key_acc:
                min_val = key_acc["min"].item()
                max_val = key_acc["max"].item()
                if max_val - min_val < self.eps:
                    max_val = min_val + self.eps
                stats.update({
                    "min": min_val,
                    "max": max_val,
                    "clamp_min": self._log_min_max_clamp_min,
                    "clamp_max": self._log_min_max_clamp_max,
                })

            final_stats[key] = stats

        return final_stats

    def _compute_quantile_stats(
            self, values: Tensor, key: str, method: str
    ) -> dict:
        """Compute quantile-based statistics with numerical stability."""
        stats: Dict[str, float] = {}

        if method == "iqr":
            q_tensor = torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype, device=values.device)
            q_vals = torch.quantile(values, q_tensor)
            q1, med, q3 = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
            iqr = max(q3 - q1, self.eps)  # Prevent zero IQR
            stats.update({"median": med, "iqr": iqr})

        elif method == "symlog":
            percentile = float(self.norm_config["symlog_percentile"])
            thr = torch.quantile(torch.abs(values), percentile).item()
            thr = max(thr, self.eps * self._symlog_min_threshold_multiplier)

            abs_v = torch.abs(values)
            mask = abs_v > thr
            transformed = torch.zeros_like(values)

            transformed[mask] = torch.sign(values[mask]) * (
                    torch.log10(abs_v[mask] / thr) + 1
            )
            transformed[~mask] = values[~mask] / thr

            sf = transformed.abs().max().item() if transformed.numel() > 0 else 1.0
            stats.update({"threshold": thr, "scale_factor": max(sf, 1.0)})

        return stats

    @staticmethod
    def normalize_tensor(
        x: Tensor, method: str, stats: Dict[str, Any], *, normalized_value_clamp: float
    ) -> Tensor:
        """Apply normalization to a tensor."""
        if not x.is_floating_point():
            raise TypeError(
                f"normalize_tensor requires a floating-point tensor, got {x.dtype}."
            )

        if method in ("none", "bool") or not stats:
            return x

        result = x

        try:
            if method == "standard":
                result = (x - stats["mean"]) / stats["std"]

            elif method == "log-standard":
                if torch.any(x <= 0):
                    raise ValueError("log-standard normalization received values <= 0.")
                x_safe = torch.log10(x)
                result = (x_safe - stats["log_mean"]) / stats["log_std"]

            elif method == "signed-log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1.0)
                result = (y - stats["mean"]) / stats["std"]

            elif method == "log-min-max":
                if torch.any(x <= 0):
                    raise ValueError("log-min-max normalization received values <= 0.")
                log_x = torch.log10(x)
                denom = stats["max"] - stats["min"]
                if denom <= 0:
                    raise ValueError("log-min-max stats have zero range")
                normed = (log_x - stats["min"]) / denom
                result = torch.clamp(normed, stats["clamp_min"], stats["clamp_max"])

            elif method == "max-out":
                result = x / stats["max_val"]

            elif method == "iqr":
                result = (x - stats["median"]) / stats["iqr"]

            elif method == "scaled_signed_offset_log":
                y = torch.sign(x) * torch.log10(torch.abs(x) + 1)
                result = y / stats["m"]

            elif method == "symlog":
                thr, sf = stats["threshold"], stats["scale_factor"]
                abs_x = torch.abs(x)
                linear_mask = abs_x <= thr
                y = torch.zeros_like(x)
                y[linear_mask] = x[linear_mask] / thr
                y[~linear_mask] = torch.sign(x[~linear_mask]) * (
                        torch.log10(abs_x[~linear_mask] / thr) + 1.0
                )
                result = y / sf

            else:
                raise ValueError(f"Unsupported normalization method '{method}'.")

        except KeyError as e:
            raise KeyError(f"Missing stat '{e}' for normalization method '{method}'.") from e

        # Clamp unbounded normalizations
        if method in ("standard", "log-standard", "signed-log", "iqr"):
            result = torch.clamp(
                result, -normalized_value_clamp, normalized_value_clamp
            )

        return result

    @staticmethod
    def denormalize_tensor(x: Tensor, method: str, stats: Dict[str, Any]) -> Tensor:
        """Reverse normalization on a tensor."""
        if not x.is_floating_point():
            raise TypeError(
                f"denormalize_tensor requires a floating-point tensor, got {x.dtype}."
            )

        if method in ("none", "bool"):
            return x

        if not stats:
            raise ValueError(f"No stats for denormalization with method '{method}'")

        dtype, device = x.dtype, x.device
        def to_t(val: float) -> Tensor:
            """Convert scalar to tensor with correct dtype/device."""
            return torch.as_tensor(val, dtype=dtype, device=device)

        if method == "standard":
            return x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))

        elif method == "log-standard":
            return 10 ** (x.mul(to_t(stats["log_std"])).add(to_t(stats["log_mean"])))

        elif method == "signed-log":
            unscaled_log = x.mul(to_t(stats["std"])).add(to_t(stats["mean"]))
            return torch.sign(unscaled_log) * (10 ** torch.abs(unscaled_log) - 1.0)

        elif method == "log-min-max":
            clamp_min = to_t(stats["clamp_min"])
            clamp_max = to_t(stats["clamp_max"])
            unscaled = (
                torch.clamp(x, clamp_min, clamp_max)
                .mul(to_t(stats["max"] - stats["min"]))
                .add(to_t(stats["min"]))
            )
            return 10 ** unscaled

        elif method == "max-out":
            return x.mul(to_t(stats["max_val"]))

        elif method == "iqr":
            return x.mul(to_t(stats["iqr"])).add(to_t(stats["median"]))

        elif method == "scaled_signed_offset_log":
            ytmp = x.mul(to_t(stats["m"]))
            return torch.sign(ytmp) * (10 ** torch.abs(ytmp) - 1)

        elif method == "symlog":
            unscaled = x.mul(to_t(stats["scale_factor"]))
            abs_unscaled = torch.abs(unscaled)
            linear_mask = abs_unscaled <= 1.0
            thr = to_t(stats["threshold"])
            y = torch.zeros_like(x)
            y[linear_mask] = unscaled[linear_mask].mul(thr)
            y[~linear_mask] = (
                    torch.sign(unscaled[~linear_mask])
                    * thr
                    * (10 ** (abs_unscaled[~linear_mask] - 1.0))
            )
            return y

        else:
            raise ValueError(f"Unsupported denormalization method '{method}'")

__all__ = ["DataNormalizer"]
