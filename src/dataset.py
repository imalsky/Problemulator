#!/usr/bin/env python3
"""
Load preprocessed atmospheric-profile shards for training and evaluation.

This module consumes processed ``.npy`` shards plus split metadata emitted by
the preprocessing stage. It does not read raw HDF5 training inputs directly.

Padding convention:
- The padding sentinel comes from ``data_specification.padding_value``.
- Masks use PyTorch's convention: ``True`` means "padding timestep".
- A timestep is padded only when every feature in that row equals the sentinel.
"""
from __future__ import annotations

import json
import logging
import psutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class AtmosphericDataset(Dataset):
    """
    Dataset for preprocessed atmospheric-profile shards.

    The dataset can either load an entire split into RAM or serve samples from
    shard-backed storage with a bounded cache, depending on config and the
    estimated split size.
    """
    
    def __init__(
        self,
        dir_path: Path,
        config: Dict[str, Any],
        indices: Optional[Sequence[int]],
    ) -> None:
        """
        Initialize dataset with automatic memory management.

        Args:
            dir_path: Directory containing processed data shards
            config: Configuration dictionary
            indices: Optional list of sample indices to use (0-based within split).
                     If None, use all samples in split order.
        """
        super().__init__()

        if not dir_path.is_dir():
            raise RuntimeError(f"Directory not found: {dir_path}")

        self.dir_path = dir_path
        self.config = config

        self.indices = list(indices) if indices is not None else None
        misc_cfg = self.config["miscellaneous_settings"]
        self.loading_mode = str(misc_cfg["dataset_loading_mode"]).lower()
        self.max_cached_shards = int(misc_cfg["dataset_max_cached_shards"])
        self.large_shard_mmap_bytes = int(misc_cfg["dataset_large_shard_mmap_bytes"])
        self.ram_safety_fraction = float(misc_cfg["dataset_ram_safety_fraction"])
        self.copy_mmap_slices = bool(misc_cfg["dataset_copy_mmap_slices"])
        
        # Extract data specification
        data_spec = self.config["data_specification"]
        self.input_variables = data_spec["input_variables"]
        self.target_variables = data_spec["target_variables"]
        self.global_variables = data_spec["global_variables"]
        # Load shard metadata first to determine structure
        self._load_metadata()
        
        # Validate directory structure based on metadata
        self._validate_structure()
        
        # Decide loading strategy and load data
        self._estimate_memory_and_load()
        
        logger.info(f"AtmosphericDataset initialized: {len(self)} samples from {dir_path} ")

    def _assert_finite_array(self, array: np.ndarray, *, array_name: str, source: str) -> None:
        """Hard-fail when processed shards contain NaN or Inf values."""
        finite_mask = np.isfinite(array)
        if bool(finite_mask.all()):
            return

        bad_count = int((~finite_mask).sum())
        raise RuntimeError(
            f"Non-finite values detected in processed {array_name} from {source} "
            f"({bad_count} values)."
        )

    def _validate_loaded_shard(
        self,
        *,
        seq_data: np.ndarray,
        tgt_data: np.ndarray,
        glb_data: Optional[np.ndarray],
        shard_name: str,
    ) -> None:
        """Validate one processed shard after load."""
        self._assert_finite_array(
            seq_data,
            array_name="sequence_inputs",
            source=f"{self.dir_path / 'sequence_inputs' / shard_name}",
        )
        self._assert_finite_array(
            tgt_data,
            array_name="targets",
            source=f"{self.dir_path / 'targets' / shard_name}",
        )
        if glb_data is not None:
            self._assert_finite_array(
                glb_data,
                array_name="globals",
                source=f"{self.dir_path / 'globals' / shard_name}",
            )
    
    def _validate_structure(self) -> None:
        """Validate that required directories exist based on metadata."""
        required_dirs = ["sequence_inputs", "targets"]
        # Check metadata instead of config for globals
        if self.has_globals:
            required_dirs.append("globals")
        
        missing = [d for d in required_dirs if not (self.dir_path / d).exists()]
        if missing:
            raise RuntimeError(f"Missing directories in {self.dir_path}: {missing}")
    
    def _load_metadata(self) -> None:
        """Load metadata about shard structure."""
        metadata_path = self.dir_path / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing metadata.json in {self.dir_path}")
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.shard_size = self.metadata["shard_size"]
        self.total_samples = self.metadata["total_samples"]
        self.num_shards = self.metadata["num_shards"]
        self.has_globals = self.metadata["has_globals"]
        self.sequence_length = self.metadata["sequence_length"]
    
    def _estimate_memory_and_load(self) -> None:
        """
        Estimate memory requirements and choose loading strategy.
        
        If dataset fits in available RAM, load everything for fast access.
        Otherwise, use disk caching with memory mapping.
        """
        # Get available shard files
        seq_dir = self.dir_path / "sequence_inputs"
        self.seq_shards = sorted(seq_dir.glob("shard_*.npy"))
        
        if len(self.seq_shards) == 0:
            raise RuntimeError(f"No shard files found in {seq_dir}")
        
        # Peek at first shard to get dtype and dimensions
        seq_shard = np.load(self.seq_shards[0], mmap_mode='r')
        seq_dtype = seq_shard.dtype
        seq_bytes_per_sample = seq_shard.itemsize * int(np.prod(seq_shard.shape[1:]))
        
        tgt_path0 = self.dir_path / "targets" / self.seq_shards[0].name
        if not tgt_path0.exists():
            raise RuntimeError(f"Missing target shard file: {tgt_path0}")
        tgt_shard = np.load(tgt_path0, mmap_mode='r')
        tgt_dtype = tgt_shard.dtype
        tgt_bytes_per_sample = tgt_shard.itemsize * int(np.prod(tgt_shard.shape[1:]))
        
        total_bytes_per_sample = seq_bytes_per_sample + tgt_bytes_per_sample
        
        # Add globals if present
        glb_dtype = None
        if self.has_globals:
            glb_path0 = self.dir_path / "globals" / self.seq_shards[0].name
            if not glb_path0.exists():
                raise RuntimeError(f"Missing globals shard file: {glb_path0}")
            glb_shard = np.load(glb_path0, mmap_mode='r')
            glb_dtype = glb_shard.dtype
            glb_bytes_per_sample = glb_shard.itemsize * int(glb_shard.shape[1])
            total_bytes_per_sample += glb_bytes_per_sample
        
        # Store dtypes for later use
        self._seq_dtype = seq_dtype
        self._tgt_dtype = tgt_dtype
        self._glb_dtype = glb_dtype
        
        # Validate indices against total_samples for *both* modes (hard-fail policy).
        if self.indices is None:
            self._valid_indices: Sequence[int] = range(self.total_samples)
            self._identity_indices = True
            invalid_indices: List[int] = []
        else:
            self._valid_indices = self.indices
            self._identity_indices = False
            invalid_indices = [i for i in self._valid_indices if not (0 <= i < self.total_samples)]
        if invalid_indices:
            preview = invalid_indices[:10]
            raise RuntimeError(
                f"Found {len(invalid_indices)} invalid indices outside [0, {self.total_samples}). "
                f"First examples: {preview}"
            )
        
        # Calculate total memory needed
        total_bytes_needed = total_bytes_per_sample * len(self._valid_indices)
        total_gb_needed = total_bytes_needed / (1024**3)
        
        # Get available memory and apply configured safety margin.
        available_memory = psutil.virtual_memory().available
        available_gb = available_memory / (1024**3)
        safe_available_gb = available_gb * self.ram_safety_fraction
        
        logger.info(
            f"Memory estimate: {total_gb_needed:.2f} GB needed for {len(self._valid_indices)} samples, "
            f"{safe_available_gb:.2f} GB safely available"
        )
        
        # Choose loading strategy from explicit config.
        selected_mode = self.loading_mode
        if selected_mode == "auto":
            selected_mode = "ram" if total_gb_needed < safe_available_gb else "disk"
            logger.info("Auto-selected dataset loading mode: %s", selected_mode)

        if selected_mode == "disk":
            self._setup_disk_cache()
            self.ram_mode = False
        elif selected_mode == "ram":
            logger.info("Loading entire dataset into RAM...")
            self._load_all_to_ram()
            self.ram_mode = True
        else:
            raise ValueError(f"Unsupported dataset loading mode '{selected_mode}'.")
    
    def _load_all_to_ram(self) -> None:
        """Load entire dataset into RAM for fast access."""
        n_samples = len(self._valid_indices)
        
        # Early exit if nothing to load
        if n_samples == 0:
            self.ram_sequences = np.zeros(
                (0, self.sequence_length, len(self.input_variables)), dtype=self._seq_dtype
            )
            self.ram_targets = np.zeros(
                (0, self.sequence_length, len(self.target_variables)), dtype=self._tgt_dtype
            )
            self.ram_globals = (
                np.zeros((0, len(self.global_variables)), dtype=self._glb_dtype)
                if self.has_globals else None
            )
            return
        
        # Allocate arrays with correct dtype
        seq_shape = (n_samples, self.sequence_length, len(self.input_variables))
        tgt_shape = (n_samples, self.sequence_length, len(self.target_variables))
        
        self.ram_sequences = np.zeros(seq_shape, dtype=self._seq_dtype)
        self.ram_targets = np.zeros(tgt_shape, dtype=self._tgt_dtype)
        
        if self.has_globals:
            glb_shape = (n_samples, len(self.global_variables))
            self.ram_globals = np.zeros(glb_shape, dtype=self._glb_dtype)
        
        # Fast path when consuming the full split in natural order.
        if self._identity_indices:
            processed = 0
            for shard_idx in range(self.num_shards):
                shard_name = f"shard_{shard_idx:06d}.npy"
                seq_data = np.load(self.dir_path / "sequence_inputs" / shard_name)
                tgt_data = np.load(self.dir_path / "targets" / shard_name)
                glb_data = None
                if self.has_globals:
                    glb_data = np.load(self.dir_path / "globals" / shard_name)
                self._validate_loaded_shard(
                    seq_data=seq_data,
                    tgt_data=tgt_data,
                    glb_data=glb_data,
                    shard_name=shard_name,
                )
                n_rows = int(seq_data.shape[0])
                end = processed + n_rows
                self.ram_sequences[processed:end] = seq_data
                self.ram_targets[processed:end] = tgt_data
                if self.has_globals and glb_data is not None:
                    self.ram_globals[processed:end] = glb_data
                processed = end
                if processed % 10000 == 0 or processed == n_samples:
                    logger.info(f"Loaded {processed}/{n_samples} samples into RAM")
            return

        # Group validated indices by shard for efficient loading
        indices_by_shard: Dict[int, List[Tuple[int, int]]] = {}
        for idx, original_idx in enumerate(self._valid_indices):
            shard_idx = original_idx // self.shard_size
            within_shard_idx = original_idx % self.shard_size
            
            if shard_idx not in indices_by_shard:
                indices_by_shard[shard_idx] = []
            indices_by_shard[shard_idx].append((idx, within_shard_idx))
        
        # Load data shard by shard
        processed = 0
        for shard_idx in sorted(indices_by_shard.keys()):
            shard_name = f"shard_{shard_idx:06d}.npy"
            
            # Load entire shard at once
            seq_data = np.load(self.dir_path / "sequence_inputs" / shard_name)
            tgt_data = np.load(self.dir_path / "targets" / shard_name)
            glb_data = None
            if self.has_globals:
                glb_data = np.load(self.dir_path / "globals" / shard_name)
            self._validate_loaded_shard(
                seq_data=seq_data,
                tgt_data=tgt_data,
                glb_data=glb_data,
                shard_name=shard_name,
            )
            
            # Vectorized extraction from this shard
            shard_pairs = indices_by_shard[shard_idx]
            pairs_arr = np.array(shard_pairs, dtype=np.int64)
            dest_idx = pairs_arr[:, 0]
            src_idx = pairs_arr[:, 1]

            self.ram_sequences[dest_idx] = seq_data[src_idx]
            self.ram_targets[dest_idx] = tgt_data[src_idx]
            if self.has_globals and glb_data is not None:
                self.ram_globals[dest_idx] = glb_data[src_idx]
            processed += int(dest_idx.size)
            
            if processed % 10000 == 0 or processed == n_samples:
                logger.info(f"Loaded {processed}/{n_samples} samples into RAM")
    
    def _setup_disk_cache(self) -> None:
        """Setup disk caching with memory mapping for large files."""
        # Map validated indices to shards (skip list materialization for identity indexing).
        self.effective_indices = []
        if not self._identity_indices:
            for idx in self._valid_indices:
                shard_idx = idx // self.shard_size
                within_shard_idx = idx % self.shard_size
                self.effective_indices.append((idx, shard_idx, within_shard_idx))
        
        # Initialize bounded shard cache (LRU).
        self._shard_cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self._cache_size = min(len(self.seq_shards), self.max_cached_shards)
        
        # Pre-load frequently accessed shards
        logger.info(f"Pre-loading up to {self._cache_size} shards into cache...")
        if self._identity_indices:
            unique_shard_indices = list(range(self.num_shards))
        else:
            unique_shard_indices = sorted(set(idx[1] for idx in self.effective_indices))
        
        for i, shard_idx in enumerate(unique_shard_indices[:self._cache_size]):
            self._load_shard(shard_idx)
            if (i + 1) % 20 == 0:
                logger.info(f"Pre-loaded {i + 1} shards")
    
    def _load_shard(self, shard_idx: int) -> Dict[str, Any]:
        """Load a shard with bounded LRU caching."""
        cached = self._shard_cache.pop(shard_idx, None)
        if cached is not None:
            self._shard_cache[shard_idx] = cached
            return cached

        shard_name = f"shard_{shard_idx:06d}.npy"
        seq_path = self.dir_path / "sequence_inputs" / shard_name
        tgt_path = self.dir_path / "targets" / shard_name
        
        if not (seq_path.exists() and tgt_path.exists()):
            raise RuntimeError(f"Missing shard files for shard {shard_idx}")
        
        # Memory map large files and keep references in a bounded cache.
        if seq_path.stat().st_size > self.large_shard_mmap_bytes:
            shard_data = {
                "sequence_inputs": np.load(seq_path, mmap_mode='r'),
                "targets": np.load(tgt_path, mmap_mode='r'),
                "mmap_backed": True,
            }
            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if not glb_path.exists():
                    raise RuntimeError(f"Missing globals for shard {shard_idx}")
                shard_data["globals"] = np.load(glb_path, mmap_mode='r')
        else:
            shard_data = {
                "sequence_inputs": np.load(seq_path),
                "targets": np.load(tgt_path),
                "mmap_backed": False,
            }
            if self.has_globals:
                glb_path = self.dir_path / "globals" / shard_name
                if not glb_path.exists():
                    raise RuntimeError(f"Missing globals for shard {shard_idx}")
                shard_data["globals"] = np.load(glb_path)
        self._validate_loaded_shard(
            seq_data=shard_data["sequence_inputs"],
            tgt_data=shard_data["targets"],
            glb_data=shard_data.get("globals"),
            shard_name=shard_name,
        )

        # LRU eviction
        if len(self._shard_cache) >= self._cache_size:
            self._shard_cache.popitem(last=False)
        self._shard_cache[shard_idx] = shard_data
        return shard_data
    
    def __len__(self) -> int:
        """Return number of loadable samples in dataset."""
        if self.ram_mode:
            return len(self.ram_sequences)
        if self._identity_indices:
            return self.total_samples
        return len(self.effective_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_dict, target_array)
        """
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        if self.ram_mode:
            # Fast path: direct RAM access
            seq_in_np = self.ram_sequences[idx]
            tgt_np = self.ram_targets[idx]
            
            # Create input dictionary
            inputs = {"sequence": seq_in_np}
            
            if self.has_globals:
                glb_np = self.ram_globals[idx]
                inputs["global_features"] = glb_np
            
            targets = tgt_np
            
        else:
            # Disk cache path
            if self._identity_indices:
                shard_idx = idx // self.shard_size
                within_shard_idx = idx % self.shard_size
            else:
                _, shard_idx, within_shard_idx = self.effective_indices[idx]
            
            # Load the shard
            shard_data = self._load_shard(shard_idx)
            
            # Extract the specific sample
            seq_in_np = shard_data["sequence_inputs"][within_shard_idx]
            tgt_np = shard_data["targets"][within_shard_idx]
            
            # Optional copy policy for mmap-backed slices only.
            if self.copy_mmap_slices and bool(shard_data.get("mmap_backed", False)):
                seq_in_np = seq_in_np.copy()
                tgt_np = tgt_np.copy()
            
            inputs = {"sequence": seq_in_np}
            
            if self.has_globals and "globals" in shard_data:
                glb_np = shard_data["globals"][within_shard_idx]
                if self.copy_mmap_slices and bool(shard_data.get("mmap_backed", False)):
                    glb_np = glb_np.copy()
                inputs["global_features"] = glb_np
            
            targets = tgt_np
        
        return inputs, targets


def create_dataset(
    dir_path: Path,
    config: Dict[str, Any],
    indices: Optional[Sequence[int]],
) -> AtmosphericDataset:
    """
    Create a dataset instance.
    
    Args:
        dir_path: Directory containing processed data
        config: Configuration dictionary
        indices: Optional sample indices to use; None means all samples
        
    Returns:
        AtmosphericDataset instance
    """
    logger.info(f"Creating dataset from {dir_path}...")
    return AtmosphericDataset(
        dir_path=dir_path,
        config=config,
        indices=indices,
    )


def pad_collate(
    batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]],
    padding_value: float,
    padding_epsilon: float,
    tensor_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor]:
    """
    Collate a batch and derive the timestep padding mask.

    The sequence tensor is treated as the authoritative source for padding.
    Preprocessing guarantees that targets are padded on the same timesteps, so
    the returned ``target_masks`` reuses the sequence mask instead of
    recomputing it from the targets.

    Args:
        batch: List of (inputs, targets) tuples
        padding_value: Sentinel value for padding
        padding_epsilon: Tolerance for padding comparison
        
    Returns:
        Tuple ``(batched_inputs, masks, batched_targets, target_masks)`` where:
        - ``batched_inputs["sequence"]`` has shape ``[batch, seq_len, input_dim]``
        - ``masks["sequence"]`` has shape ``[batch, seq_len]``
        - ``batched_targets`` has shape ``[batch, seq_len, target_dim]``
        - ``target_masks`` has shape ``[batch, seq_len]``
        All masks use ``True`` for padding positions.
    """
    inputs, targets = zip(*batch)

    def _stack_to_tensor(values: Sequence[np.ndarray | Tensor]) -> Tensor:
        first = values[0]
        if isinstance(first, torch.Tensor):
            tensor_values = [
                v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                for v in values
            ]
            tensor = torch.stack(tensor_values)
        else:
            stacked_np = np.stack(values, axis=0)
            tensor = torch.from_numpy(stacked_np)
        if tensor_dtype is not None and tensor.dtype != tensor_dtype:
            tensor = tensor.to(dtype=tensor_dtype)
        return tensor
    
    # Stack sequences
    seq = _stack_to_tensor([d["sequence"] for d in inputs])
    
    # Padding is defined per timestep: every feature in that row must match the
    # configured sentinel before the timestep is considered padded.
    seq_mask = (torch.abs(seq - padding_value) <= padding_epsilon).all(dim=-1)
    
    # Build batched inputs and masks
    batched = {"sequence": seq}
    masks = {"sequence": seq_mask}  # True = padding
    
    # Handle global features if present
    if "global_features" in inputs[0]:
        batched["global_features"] = _stack_to_tensor(
            [d["global_features"] for d in inputs]
        )
    
    # Stack targets. Padding is defined by the input-sequence contract, so the
    # sequence mask is reused for the regression targets as well.
    tgt = _stack_to_tensor(targets)

    if tgt.shape[:2] != seq.shape[:2]:
        raise RuntimeError(
            f"Target shape {tuple(tgt.shape[:2])} does not match sequence shape {tuple(seq.shape[:2])}."
        )

    tgt_mask = seq_mask
    if bool(tgt_mask.all()):
        raise RuntimeError("All-padding batch encountered during collation.")
    
    return batched, masks, tgt, tgt_mask


def create_collate_fn(
    padding_value: float,
    padding_epsilon: float,
    tensor_dtype: Optional[torch.dtype] = None,
) -> Callable:
    """
    Create a collate function with specified padding value and epsilon.

    Args:
        padding_value: Sentinel value for padding
        padding_epsilon: Tolerance for padding comparison

    Returns:
        Partial collate function
    """
    return partial(
        pad_collate,
        padding_value=padding_value,
        padding_epsilon=padding_epsilon,
        tensor_dtype=tensor_dtype,
    )


__all__ = ["AtmosphericDataset", "create_dataset", "create_collate_fn"]
