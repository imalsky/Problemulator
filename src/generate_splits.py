#!/usr/bin/env python3
"""generate_splits.py - Dataset split generation utility."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from utils import compress_splits, get_config_str, load_config, save_json, setup_logging

# Default paths are anchored to project root (sibling of src/, config/, unit_tests/),
# not the process working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "transformer.jsonc"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


def _resolve_from_project_root(path: Path) -> Path:
    """Resolve relative paths from project root instead of process CWD."""
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _assert_not_inside_src(path: Path, label: str) -> None:
    """Reject data paths that resolve under src/."""
    src_root = (PROJECT_ROOT / "src").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(src_root)
    except ValueError:
        return
    raise ValueError(
        f"{label} must not be inside {src_root}. "
        "Use project-root sibling directories like 'data/'."
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate required dataset_splits.json from configured raw HDF5 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="Root data directory (contains raw/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional split seed override. Defaults to miscellaneous_settings.random_seed.",
    )
    return parser.parse_args()


def _get_sample_count(hdf5_path: Path, profile_key: str) -> int:
    """Return number of profile rows in a configured HDF5 file."""
    with h5py.File(hdf5_path, "r", swmr=True, libver="latest") as hf:
        if profile_key not in hf:
            raise ValueError(
                f"Configured profile key '{profile_key}' missing in {hdf5_path.name}."
            )
        profile_ds = hf[profile_key]
        if len(profile_ds.shape) != 2:
            raise ValueError(
                f"Configured profile key '{profile_key}' in {hdf5_path.name} must be rank-2."
            )
        sample_count = int(profile_ds.shape[0])
        if sample_count <= 0:
            raise ValueError(f"Configured profile key '{profile_key}' in {hdf5_path.name} is empty.")
        return sample_count


def _build_all_pairs(raw_hdf5_paths: List[Path], profile_key: str) -> List[Tuple[str, int]]:
    """Build full (file_stem, sample_index) list across all configured raw files."""
    pairs: List[Tuple[str, int]] = []
    for path in raw_hdf5_paths:
        sample_count = _get_sample_count(path, profile_key)
        stem = path.stem
        pairs.extend((stem, idx) for idx in range(sample_count))
        logger.info("Indexed %d rows from %s", sample_count, path.name)
    return pairs


def _split_pairs(
    all_pairs: List[Tuple[str, int]],
    *,
    seed: int,
    split_fractions: Dict[str, float],
) -> Dict[str, List[Tuple[str, int]]]:
    """Shuffle all pairs with a fixed seed and split into train/validation/test."""
    if not all_pairs:
        raise ValueError("No sample pairs found; cannot generate splits.")

    ratio_sum = sum(float(split_fractions[name]) for name in ("train", "validation", "test"))
    if abs(ratio_sum - 1.0) > 1e-12:
        raise RuntimeError(f"Split fractions must sum to 1.0, got {ratio_sum}.")

    n_total = len(all_pairs)
    rng = np.random.default_rng(seed)
    shuffled = [all_pairs[i] for i in rng.permutation(n_total)]

    n_train = int(n_total * float(split_fractions["train"]))
    n_validation = int(n_total * float(split_fractions["validation"]))
    n_test = n_total - n_train - n_validation

    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError(
            "Generated split has an empty partition. "
            f"Counts: train={n_train}, validation={n_validation}, test={n_test}."
        )

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }


def generate_splits_from_config(
    config: Dict[str, Any],
    data_dir: Path,
    *,
    seed_override: Optional[int] = None,
) -> Path:
    """
    Generate compact dataset_splits.json from an already-loaded config.

    Returns:
        Path to the generated splits file.
    """
    data_dir = _resolve_from_project_root(Path(data_dir))
    _assert_not_inside_src(data_dir, "--data-dir")

    input_variables = config["data_specification"]["input_variables"]
    if not isinstance(input_variables, list) or not input_variables:
        raise ValueError("Config key 'data_specification.input_variables' must be non-empty.")
    profile_key = str(input_variables[0])

    h5_filenames = config["data_paths_config"]["hdf5_dataset_filename"]
    raw_hdf5_paths = [data_dir / "raw" / name for name in h5_filenames]
    missing_files = [p for p in raw_hdf5_paths if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing configured raw HDF5 files: {missing_files}")

    seed = int(config["miscellaneous_settings"]["random_seed"])
    if seed_override is not None:
        seed = int(seed_override)
    split_fractions = {
        name: float(config["data_paths_config"]["dataset_split_fractions"][name])
        for name in ("train", "validation", "test")
    }

    splits_filename = get_config_str(
        config, "data_paths_config", "dataset_splits_filename", "dataset splits"
    )
    output_path = data_dir / splits_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building splits from %d raw files with seed=%d", len(raw_hdf5_paths), seed)
    all_pairs = _build_all_pairs(raw_hdf5_paths, profile_key)
    splits = _split_pairs(all_pairs, seed=seed, split_fractions=split_fractions)

    compact_splits = compress_splits(splits)
    if not save_json(compact_splits, output_path, compact=True):
        raise RuntimeError(f"Failed to write split file: {output_path}")

    logger.info("Saved compact split file: %s", output_path.resolve())
    logger.info(
        "Split sizes: %d train, %d validation, %d test.",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return output_path


def main() -> int:
    """Generate compact dataset_splits.json required by preprocessing/training."""
    args = _parse_arguments()
    args.config = _resolve_from_project_root(args.config)
    args.data_dir = _resolve_from_project_root(args.data_dir)
    setup_logging()

    try:
        _assert_not_inside_src(args.data_dir, "--data-dir")

        config = load_config(args.config)
        _ = generate_splits_from_config(
            config=config,
            data_dir=args.data_dir,
            seed_override=args.seed,
        )
        return 0

    except Exception as exc:
        logger.critical("Failed to generate dataset splits: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
