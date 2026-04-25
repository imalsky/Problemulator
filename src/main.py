#!/usr/bin/env python3
"""main.py - Entry point for configured pipeline execution."""
from __future__ import annotations

import os
import sys
import argparse
import logging
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Prevent MKL/OpenMP library conflicts before importing torch.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from dataset import create_collate_fn
from generate_splits import generate_splits_from_config
from hardware import setup_device
from preprocess import preprocess_data
from train import ModelTrainer
from utils import (
    ensure_dirs,
    get_config_str,
    get_precision_config,
    load_config,
    load_splits,
    save_json,
    seed_everything,
    setup_logging,
)

# Default paths are anchored to project root (sibling of src/, config/, unit_tests/),
# not the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "transformer.jsonc"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_FINGERPRINT_FILENAME = "processed_fingerprint.json"
PROCESSED_FINGERPRINT_VERSION = 2

logger = logging.getLogger(__name__)


def _resolve_from_project_root(path: Path) -> Path:
    """Resolve relative paths from project root instead of process CWD."""
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _assert_not_inside_src(path: Path, label: str) -> None:
    """Reject data/models paths that resolve under src/."""
    src_root = (PROJECT_ROOT / "src").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(src_root)
    except ValueError:
        return
    raise ValueError(
        f"{label} must not be inside {src_root}. "
        "Use project-root sibling directories like 'data/' and 'models/'."
    )


def _sha256_bytes(data: bytes) -> str:
    """Return SHA256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Return SHA256 hex digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _preprocess_relevant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only config fields that affect processed artifact contents."""
    misc = config["miscellaneous_settings"]
    precision = config["precision"]
    return {
        "data_paths_config": config["data_paths_config"],
        "data_specification": config["data_specification"],
        "normalization": config["normalization"],
        "model_hyperparameters": {
            "max_sequence_length": config["model_hyperparameters"]["max_sequence_length"],
        },
        "precision": {
            "input_dtype": precision["input_dtype"],
            "stats_accumulation_dtype": precision["stats_accumulation_dtype"],
        },
        "miscellaneous_settings": {
            "device_backend": misc["device_backend"],
            "shard_size": misc["shard_size"],
            "hdf5_read_chunk_size": misc["hdf5_read_chunk_size"],
        },
    }


def _stable_config_sha256(config: Dict[str, Any]) -> str:
    """Hash preprocessing-relevant config deterministically."""
    relevant = _preprocess_relevant_config(config)
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_bytes(payload.encode("utf-8"))


def _build_processed_fingerprint(
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    splits_path: Path,
) -> Dict[str, Any]:
    """Build reproducibility fingerprint for processed artifacts."""
    split_stat = splits_path.stat()
    raw_entries = []
    for path in raw_hdf5_paths:
        stat = path.stat()
        raw_entries.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    return {
        "version": PROCESSED_FINGERPRINT_VERSION,
        "config_sha256": _stable_config_sha256(config),
        "splits": {
            "path": str(splits_path.resolve()),
            "size_bytes": int(split_stat.st_size),
            "mtime_ns": int(split_stat.st_mtime_ns),
            "sha256": _sha256_file(splits_path),
        },
        "raw_hdf5_files": raw_entries,
    }


def _load_json_dict(path: Path) -> Dict[str, Any]:
    """Load JSON and enforce object root."""
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(loaded).__name__}.")
    return loaded


def _validate_processed_split_dir(split_dir: Path) -> None:
    """Validate one processed split directory structure and shard counts."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing processed split directory: {split_dir}")

    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")
    metadata = _load_json_dict(metadata_path)

    required_keys = {
        "split",
        "total_samples",
        "shard_size",
        "num_shards",
        "sequence_length",
        "has_globals",
    }
    missing_keys = required_keys - set(metadata.keys())
    if missing_keys:
        raise ValueError(f"Split metadata missing keys {sorted(missing_keys)} in {metadata_path}")

    num_shards = int(metadata["num_shards"])
    if num_shards < 0:
        raise ValueError(f"Invalid num_shards={num_shards} in {metadata_path}")

    seq_dir = split_dir / "sequence_inputs"
    tgt_dir = split_dir / "targets"
    if not seq_dir.is_dir() or not tgt_dir.is_dir():
        raise FileNotFoundError(f"Missing sequence_inputs/targets directories in {split_dir}")

    seq_shards = sorted(seq_dir.glob("shard_*.npy"))
    tgt_shards = sorted(tgt_dir.glob("shard_*.npy"))
    if len(seq_shards) != num_shards or len(tgt_shards) != num_shards:
        raise RuntimeError(
            f"Shard count mismatch in {split_dir}: expected {num_shards}, "
            f"found sequence_inputs={len(seq_shards)}, targets={len(tgt_shards)}."
        )

    has_globals = bool(metadata["has_globals"])
    glb_dir = split_dir / "globals"
    if has_globals:
        if not glb_dir.is_dir():
            raise FileNotFoundError(f"Missing globals directory in {split_dir}")
        glb_shards = sorted(glb_dir.glob("shard_*.npy"))
        if len(glb_shards) != num_shards:
            raise RuntimeError(
                f"Globals shard count mismatch in {split_dir}: expected {num_shards}, "
                f"found {len(glb_shards)}."
            )


def _validate_processed_artifacts(processed_dir: Path) -> None:
    """Validate required processed artifact structure."""
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"Processed directory does not exist: {processed_dir}")

    norm_meta_path = processed_dir / "normalization_metadata.json"
    if not norm_meta_path.is_file():
        raise FileNotFoundError(f"Missing normalization metadata: {norm_meta_path}")
    _ = _load_json_dict(norm_meta_path)

    for split_name in ("train", "val", "test"):
        _validate_processed_split_dir(processed_dir / split_name)


def _compare_processed_fingerprints(
    expected: Dict[str, Any],
    existing: Dict[str, Any],
) -> Tuple[bool, str]:
    """Compare fingerprints and return (match, reason_if_mismatch)."""
    if existing.get("version") != expected["version"]:
        return False, "fingerprint version mismatch"

    if existing.get("config_sha256") != expected["config_sha256"]:
        return False, "config differs from processed artifacts"

    existing_splits = existing.get("splits", {})
    expected_splits = expected["splits"]
    if not isinstance(existing_splits, dict):
        return False, "invalid splits section in processed fingerprint"
    if existing_splits.get("sha256") != expected_splits["sha256"]:
        return False, "dataset_splits file content differs"

    if existing.get("raw_hdf5_files") != expected["raw_hdf5_files"]:
        return False, "raw HDF5 file list/size/mtime differs"

    return True, ""


def _can_reuse_processed_data(
    *,
    config: Dict[str, Any],
    raw_hdf5_paths: List[Path],
    splits_path: Path,
    processed_dir: Path,
) -> Tuple[bool, str]:
    """Return whether processed artifacts can be safely reused."""
    try:
        _validate_processed_artifacts(processed_dir)
    except Exception as exc:
        return False, f"processed artifact validation failed ({exc})"

    fingerprint_path = processed_dir / PROCESSED_FINGERPRINT_FILENAME
    if not fingerprint_path.is_file():
        return False, f"missing processed fingerprint file ({fingerprint_path})"

    try:
        existing = _load_json_dict(fingerprint_path)
    except Exception as exc:
        return False, f"invalid processed fingerprint file ({exc})"

    expected = _build_processed_fingerprint(config, raw_hdf5_paths, splits_path)
    return _compare_processed_fingerprints(expected=expected, existing=existing)


def _max_split_index_per_stem(
    existing: Dict[str, Any], stems: List[str]
) -> Dict[str, int]:
    """Return the largest sample index referenced for each stem in the splits file."""
    max_index: Dict[str, int] = {stem: -1 for stem in stems}
    for split_name in ("train", "validation", "test"):
        items = existing.get(split_name, []) or []
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                continue
            stem_idx, sample_idx = item
            if not isinstance(stem_idx, int) or not isinstance(sample_idx, int):
                continue
            if stem_idx < 0 or stem_idx >= len(stems):
                continue
            stem = stems[stem_idx]
            if sample_idx > max_index[stem]:
                max_index[stem] = sample_idx
    return max_index


def _row_count_per_stem(
    config: Dict[str, Any], data_root_dir: Path, stems: List[str]
) -> Dict[str, int]:
    """Read the leading-dim row count for each configured raw HDF5 file."""
    import h5py

    profile_key = str(config["data_specification"]["input_variables"][0])
    raw_dir = data_root_dir / "raw"
    counts: Dict[str, int] = {}
    for filename in config["data_paths_config"]["hdf5_dataset_filename"]:
        path = raw_dir / filename
        if not path.is_file():
            continue
        stem = Path(filename).stem
        if stem not in stems:
            continue
        with h5py.File(path, "r", swmr=True, libver="latest") as hf:
            if profile_key in hf:
                counts[stem] = int(hf[profile_key].shape[0])
    return counts


def _ensure_splits_file(config: Dict[str, Any], data_root_dir: Path) -> Path:
    """Ensure splits file exists and matches the configured raw HDF5 files."""
    splits_filename = get_config_str(
        config, "data_paths_config", "dataset_splits_filename", "dataset splits"
    )
    splits_path = data_root_dir / splits_filename

    h5_filenames = config["data_paths_config"]["hdf5_dataset_filename"]
    configured_stems = {Path(name).stem for name in h5_filenames}

    regenerate_reason: str | None = None
    if not splits_path.is_file():
        regenerate_reason = "splits file is missing"
    else:
        try:
            existing = _load_json_dict(splits_path)
            existing_stems = list(existing.get("file_stems", []) or [])
        except Exception as exc:
            regenerate_reason = f"could not read existing splits file ({exc})"
            existing_stems = []
        if regenerate_reason is None and set(existing_stems) != configured_stems:
            regenerate_reason = (
                f"splits file stems {sorted(existing_stems)} do not match "
                f"configured raw file stems {sorted(configured_stems)}"
            )
        if regenerate_reason is None:
            try:
                max_index_per_stem = _max_split_index_per_stem(existing, existing_stems)
                row_count_per_stem = _row_count_per_stem(config, data_root_dir, existing_stems)
            except Exception as exc:
                regenerate_reason = f"could not validate splits against raw files ({exc})"
            else:
                for stem, max_idx in max_index_per_stem.items():
                    available = row_count_per_stem.get(stem, 0)
                    if max_idx >= available:
                        regenerate_reason = (
                            f"splits reference index {max_idx} for stem '{stem}' but the "
                            f"raw file now has only {available} rows"
                        )
                        break

    if regenerate_reason is None:
        return splits_path

    logger.warning(
        "Regenerating splits file %s: %s.",
        splits_path,
        regenerate_reason,
    )
    generated_path = generate_splits_from_config(config=config, data_dir=data_root_dir)
    if generated_path.resolve() != splits_path.resolve():
        raise RuntimeError(
            f"Split generation returned unexpected path {generated_path}; expected {splits_path}."
        )
    if not splits_path.is_file():
        raise RuntimeError(f"Split generation did not create expected file: {splits_path}")
    return splits_path


def _get_raw_hdf5_paths(config: Dict[str, Any], raw_dir: Path) -> List[Path]:
    """
    Get list of raw HDF5 file paths from configuration.
    
    Args:
        config: Configuration dictionary
        raw_dir: Directory containing raw HDF5 files
        
    Returns:
        List of HDF5 file paths
        
    Raises:
        ValueError: If no files specified
        FileNotFoundError: If files are missing
    """
    h5_filenames = config["data_paths_config"]["hdf5_dataset_filename"]

    raw_paths = [raw_dir / fname for fname in h5_filenames]
    missing = [p for p in raw_paths if not p.is_file()]
    
    if missing:
        raise FileNotFoundError(f"Missing raw HDF5 files: {missing}")
    
    return raw_paths


def _is_out_of_range_split_error(exc: Exception) -> bool:
    """Return whether an exception reflects stale split indices against raw files."""
    return isinstance(exc, ValueError) and "Out-of-range index in split" in str(exc)


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Atmospheric profile transformer pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (contains raw/ and processed/).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory for saved models.",
    )

    return parser.parse_args()


def run_normalize(
        config: Dict[str, Any],
        raw_hdf5_paths: List[Path],
        processed_dir: Path,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Run data normalization step.

    Args:
        config: Configuration dictionary
        raw_hdf5_paths: List of raw HDF5 files
        processed_dir: Directory for processed data

    Returns:
        Dictionary of data splits
    """
    logger.info("=== Running Data Normalization ===")

    _ = _ensure_splits_file(config, processed_dir.parent)

    # Load required splits file
    splits, splits_path = load_splits(
        config, processed_dir.parent
    )

    rebuild_processed = bool(config["miscellaneous_settings"]["rebuild_processed_data"])

    # Rebuild policy is explicit in config.
    import shutil
    if rebuild_processed:
        if processed_dir.exists():
            logger.info("Removing existing processed data before normalization.")
            shutil.rmtree(processed_dir)
    else:
        if processed_dir.exists() and any(processed_dir.iterdir()):
            reusable, reason = _can_reuse_processed_data(
                config=config,
                raw_hdf5_paths=raw_hdf5_paths,
                splits_path=splits_path,
                processed_dir=processed_dir,
            )
            if reusable:
                logger.info(
                    "Reusing existing processed artifacts in %s "
                    "(fingerprint and structure validation passed).",
                    processed_dir,
                )
                logger.info("=== Data Normalization Complete (reused existing artifacts) ===")
                return splits
            raise RuntimeError(
                "processed_dir exists and rebuild_processed_data=false, but existing artifacts "
                f"cannot be safely reused: {reason}. "
                "Set rebuild_processed_data=true to rebuild processed artifacts."
            )

    # Run preprocessing
    try:
        preprocess_data(
            config=config,
            raw_hdf5_paths=raw_hdf5_paths,
            splits=splits,
            processed_dir=processed_dir,
        )
    except Exception as exc:
        if not _is_out_of_range_split_error(exc):
            raise

        logger.warning(
            "Regenerating splits file %s after stale index validation failure: %s",
            splits_path,
            exc,
        )
        splits_path = generate_splits_from_config(config=config, data_dir=processed_dir.parent)
        splits, splits_path = load_splits(config, processed_dir.parent)
        preprocess_data(
            config=config,
            raw_hdf5_paths=raw_hdf5_paths,
            splits=splits,
            processed_dir=processed_dir,
        )

    fingerprint = _build_processed_fingerprint(config, raw_hdf5_paths, splits_path)
    fingerprint_path = processed_dir / PROCESSED_FINGERPRINT_FILENAME
    if not save_json(fingerprint, fingerprint_path, compact=True):
        raise RuntimeError(f"Failed to save processed fingerprint: {fingerprint_path}")
    logger.info("Saved processed fingerprint: %s", fingerprint_path)

    logger.info("=== Data Normalization Complete ===")
    return splits


def run_train(
    config: Dict[str, Any],
    device: torch.device,
    model_save_dir: Path,
    processed_dir: Path,
    raw_hdf5_paths: List[Path],
) -> None:
    """
    Run the training pipeline.
    
    Args:
        config: Configuration dictionary
        device: Compute device
        model_save_dir: Directory for saving models
        processed_dir: Directory with processed data
        raw_hdf5_paths: List of raw HDF5 files
    """
    logger.info("=== Running Model Training ===")
    
    # Ensure data is preprocessed
    splits = run_normalize(config, raw_hdf5_paths, processed_dir)
    
    # Get padding value and epsilon from config
    padding_val = float(config["data_specification"]["padding_value"])
    padding_eps = float(config["normalization"]["padding_comparison_epsilon"])
    input_dtype = get_precision_config(config)["input_dtype"]
    collate_fn = create_collate_fn(padding_val, padding_eps, tensor_dtype=input_dtype)
    trainer = ModelTrainer(
        config=config,
        device=device,
        save_dir=model_save_dir,
        processed_dir=processed_dir,
        splits=splits,
        collate_fn=collate_fn,
    )
    trainer.train()
    trainer.test()
    
    logger.info("=== Model Training Complete ===")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = _parse_arguments()
    args.config = _resolve_from_project_root(args.config)
    args.data_dir = _resolve_from_project_root(args.data_dir)
    args.models_dir = _resolve_from_project_root(args.models_dir)
    
    # Setup logging first so early failures are recorded.
    setup_logging()
    
    try:
        _assert_not_inside_src(args.data_dir, "--data-dir")
        _assert_not_inside_src(args.models_dir, "--models-dir")

        # Setup directories
        if not ensure_dirs(args.data_dir, args.models_dir):
            raise RuntimeError("Failed to create required data/models directories.")

        logger.info(f"Using config: {args.config.resolve()}")

        # Load configuration
        config = load_config(args.config)
        command = config["miscellaneous_settings"]["execution_mode"]
        logger.info(f"Configured execution_mode: {command}")
        
        # Set random seed for reproducibility
        seed = int(config["miscellaneous_settings"]["random_seed"])
        seed_everything(seed)
        
        # Setup compute device
        backend = str(config["miscellaneous_settings"]["device_backend"])
        device = setup_device(backend)
        
        # Apply explicit matmul precision from config (no capability-based auto-switching).
        matmul_precision = str(config["precision"]["float32_matmul_precision"]).lower()
        if matmul_precision != "none":
            if not hasattr(torch, "set_float32_matmul_precision"):
                raise RuntimeError(
                    "Configured float32 matmul precision, but this PyTorch build "
                    "does not support torch.set_float32_matmul_precision."
                )
            torch.set_float32_matmul_precision(matmul_precision)
            logger.info(
                "Set float32 matmul precision to '%s' from config.",
                matmul_precision,
            )
        
        # Setup paths
        raw_dir = args.data_dir / "raw"
        processed_dir = args.data_dir / "processed"
        raw_hdf5_paths = _get_raw_hdf5_paths(config, raw_dir)
        
        # Create model save directory
        model_folder = get_config_str(
            config, "output_paths_config", "fixed_model_foldername", "model training"
        )
        model_save_dir = args.models_dir / model_folder
        if not ensure_dirs(model_save_dir):
            raise RuntimeError(f"Failed to create model save directory: {model_save_dir}")
        
        # Setup logging to file
        log_file = model_save_dir / f"{command}_run.log"
        setup_logging(log_file=log_file, force=True)
        
        # Save configuration
        if not save_json(config, model_save_dir / f"{command}_config.json"):
            raise RuntimeError("Failed to save run configuration JSON.")
        
        # Execute the appropriate command
        if command == "normalize":
            run_normalize(config, raw_hdf5_paths, processed_dir)
        
        elif command == "train":
            run_train(
                config, device, model_save_dir, processed_dir,
                raw_hdf5_paths
            )
        
        else:
            raise ValueError(
                f"Unknown execution_mode '{command}'. Expected normalize/train."
            )
        
        logger.info(f"{command.capitalize()} completed successfully.")
        return 0
        
    except RuntimeError as e:
        logger.error(f"Pipeline error: {e}", exc_info=False)
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        return 130
        
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
