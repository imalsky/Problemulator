#!/usr/bin/env python3
"""Benchmark dataset construction and DataLoader collation throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from dataset import create_collate_fn, create_dataset
from utils import get_precision_config, load_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.jsonc"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark processed-dataset construction and collation throughput."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the validated config file.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed train/val/test shards.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_DIR_MAP),
        default="train",
        help="Which processed split to benchmark.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Maximum number of batches to iterate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional DataLoader batch size override.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker count override.",
    )
    return parser.parse_args()


def benchmark_dataset_collate(
    *,
    config: Dict[str, Any],
    processed_dir: Path,
    split: str,
    num_batches: int,
    batch_size: int | None,
    num_workers: int | None,
) -> Dict[str, float]:
    """Benchmark processed-dataset construction plus DataLoader collation."""
    split_dir = processed_dir / SPLIT_DIR_MAP[split]
    padding_value = float(config["data_specification"]["padding_value"])
    padding_epsilon = float(config["normalization"]["padding_comparison_epsilon"])
    input_dtype = get_precision_config(config)["input_dtype"]

    dataset_start = time.perf_counter()
    dataset = create_dataset(split_dir, config, indices=None)
    dataset_init_s = time.perf_counter() - dataset_start

    loader_batch_size = (
        int(batch_size)
        if batch_size is not None
        else int(config["training_hyperparameters"]["batch_size"])
    )
    loader_num_workers = (
        int(num_workers)
        if num_workers is not None
        else int(config["miscellaneous_settings"]["num_workers"])
    )
    collate_fn = create_collate_fn(
        padding_value=padding_value,
        padding_epsilon=padding_epsilon,
        tensor_dtype=input_dtype,
    )

    loader_start = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=loader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )
    loader_init_s = time.perf_counter() - loader_start

    iterated_batches = 0
    iterated_samples = 0
    iterate_start = time.perf_counter()
    for batch in loader:
        _, _, targets, _ = batch
        iterated_batches += 1
        iterated_samples += int(targets.shape[0])
        if iterated_batches >= num_batches:
            break
    iterate_s = time.perf_counter() - iterate_start

    return {
        "dataset_init_s": dataset_init_s,
        "loader_init_s": loader_init_s,
        "iterate_s": iterate_s,
        "iterated_batches": float(iterated_batches),
        "iterated_samples": float(iterated_samples),
        "batches_per_s": float(iterated_batches / max(iterate_s, 1e-12)),
        "samples_per_s": float(iterated_samples / max(iterate_s, 1e-12)),
    }


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be a positive integer.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    config = load_config(args.config)
    summary = benchmark_dataset_collate(
        config=config,
        processed_dir=args.processed_dir,
        split=args.split,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
