#!/usr/bin/env python3
"""Unit tests for config validation and explicit backend behavior."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_splits import _split_pairs
from hardware import setup_device
from utils import get_precision_config, load_config, load_splits, validate_config


def _load_checked_in_config() -> dict:
    """Load the checked-in JSONC config as plain JSON for test mutation."""
    config_path = PROJECT_ROOT / "config" / "config.jsonc"
    return load_config(config_path)


class ConfigAndHardwareTests(unittest.TestCase):
    """Validate the fixed runtime policy around config and device selection."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = _load_checked_in_config()

    def test_checked_in_config_validates(self) -> None:
        """The repository default config should pass strict validation."""
        validate_config(copy.deepcopy(self.base_config))

    def test_checked_in_config_defaults_to_cuda(self) -> None:
        """The checked-in config now matches the SLURM/CUDA launcher."""
        self.assertEqual(
            self.base_config["miscellaneous_settings"]["device_backend"],
            "cuda",
        )

    def test_validate_config_rejects_none_normalization(self) -> None:
        """Required variables may not use the removed 'none' normalization mode."""
        config = copy.deepcopy(self.base_config)
        config["normalization"]["key_methods"]["temperature_k"] = "none"

        with self.assertRaisesRegex(ValueError, "Unsupported normalization method"):
            validate_config(config)

    def test_get_precision_config_rejects_float64_on_mps(self) -> None:
        """MPS precision policy should fail before runtime when float64 is requested."""
        config = copy.deepcopy(self.base_config)
        config["miscellaneous_settings"]["device_backend"] = "mps"
        config["precision"]["input_dtype"] = "float64"
        config["precision"]["stats_accumulation_dtype"] = "float64"
        config["precision"]["model_dtype"] = "float64"
        config["precision"]["forward_dtype"] = "float64"
        config["precision"]["loss_dtype"] = "float64"
        config["precision"]["optimizer_state_dtype"] = "float64"

        with self.assertRaisesRegex(ValueError, "MPS backend does not support float64"):
            get_precision_config(config)

    def test_validate_config_rejects_bfloat16_processed_input_dtype(self) -> None:
        """Processed shards must stay representable as NumPy float arrays."""
        config = copy.deepcopy(self.base_config)
        config["precision"]["input_dtype"] = "bfloat16"

        with self.assertRaisesRegex(ValueError, "precision.input_dtype must be float16"):
            validate_config(config)

    def test_validate_config_rejects_large_hdf5_chunk_size(self) -> None:
        """Preprocessing buffer sizing must fail at config-load time."""
        config = copy.deepcopy(self.base_config)
        shard_size = config["miscellaneous_settings"]["shard_size"]
        config["miscellaneous_settings"]["hdf5_read_chunk_size"] = shard_size

        with self.assertRaisesRegex(ValueError, "must be less than"):
            validate_config(config)

    def test_validate_config_rejects_invalid_split_fractions(self) -> None:
        """Configured split fractions must be explicit and sum to 1.0."""
        config = copy.deepcopy(self.base_config)
        config["data_paths_config"]["dataset_split_fractions"]["test"] = 0.20

        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            validate_config(config)

    def test_validate_config_accepts_lstm_model_type(self) -> None:
        """LSTM configs should validate when the LSTM subsection is complete."""
        config = copy.deepcopy(self.base_config)
        config["model_hyperparameters"]["model_type"] = "lstm"

        validate_config(config)

    def test_validate_config_accepts_transformer_without_inactive_lstm_section(self) -> None:
        """Only the active architecture section should be required."""
        config = copy.deepcopy(self.base_config)
        del config["model_hyperparameters"]["lstm"]

        validate_config(config)

    def test_validate_config_accepts_lstm_without_inactive_transformer_section(self) -> None:
        """The inactive transformer section should not be required for LSTM runs."""
        config = copy.deepcopy(self.base_config)
        config["model_hyperparameters"]["model_type"] = "lstm"
        del config["model_hyperparameters"]["transformer"]

        validate_config(config)

    def test_validate_config_rejects_invalid_model_type(self) -> None:
        """Model type must be limited to the supported architecture set."""
        config = copy.deepcopy(self.base_config)
        config["model_hyperparameters"]["model_type"] = "cnn"

        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_config(config)

    def test_validate_config_rejects_missing_transformer_section(self) -> None:
        """Transformer configs must include the transformer hyperparameter section."""
        config = copy.deepcopy(self.base_config)
        del config["model_hyperparameters"]["transformer"]

        with self.assertRaisesRegex(ValueError, "model_hyperparameters.transformer"):
            validate_config(config)

    def test_validate_config_rejects_missing_lstm_section_for_lstm_runs(self) -> None:
        """LSTM configs must include the LSTM hyperparameter section."""
        config = copy.deepcopy(self.base_config)
        config["model_hyperparameters"]["model_type"] = "lstm"
        del config["model_hyperparameters"]["lstm"]

        with self.assertRaisesRegex(ValueError, "model_hyperparameters.lstm"):
            validate_config(config)

    def test_validate_config_rejects_unsupported_scheduler_type(self) -> None:
        """Scheduler type must be limited to the supported set."""
        config = copy.deepcopy(self.base_config)
        config["training_hyperparameters"]["scheduler_type"] = "onecycle"

        with self.assertRaisesRegex(ValueError, "scheduler_type"):
            validate_config(config)

    def test_validate_config_rejects_invalid_plateau_factor(self) -> None:
        """Plateau LR decay factor must stay strictly between zero and one."""
        config = copy.deepcopy(self.base_config)
        config["training_hyperparameters"]["plateau_factor"] = 1.0

        with self.assertRaisesRegex(ValueError, "plateau_factor"):
            validate_config(config)

    def test_validate_config_rejects_invalid_plateau_threshold_mode(self) -> None:
        """Plateau threshold mode should remain in the explicit supported set."""
        config = copy.deepcopy(self.base_config)
        config["training_hyperparameters"]["plateau_threshold_mode"] = "percent"

        with self.assertRaisesRegex(ValueError, "plateau_threshold_mode"):
            validate_config(config)

    def test_split_pairs_uses_configured_fractions(self) -> None:
        """Split generation should honor the configured train/val/test ratios."""
        all_pairs = [("file_a", idx) for idx in range(20)]

        splits = _split_pairs(
            all_pairs,
            seed=42,
            split_fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        )

        self.assertEqual(len(splits["train"]), 10)
        self.assertEqual(len(splits["validation"]), 5)
        self.assertEqual(len(splits["test"]), 5)

    def test_load_splits_rejects_cross_partition_duplicates(self) -> None:
        """Loaded split files must hard-fail when the same sample appears twice."""
        config = {
            "data_paths_config": {
                "dataset_splits_filename": "dataset_splits.json",
            }
        }
        payload = {
            "file_stems": ["file_a"],
            "train": [[0, 0]],
            "validation": [[0, 0]],
            "test": [[0, 1]],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            splits_path = Path(tmpdir) / "dataset_splits.json"
            with splits_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                load_splits(config, Path(tmpdir))

    def test_load_splits_rejects_duplicates_within_split(self) -> None:
        """Loaded split files must reject repeated samples inside one partition."""
        config = {
            "data_paths_config": {
                "dataset_splits_filename": "dataset_splits.json",
            }
        }
        payload = {
            "file_stems": ["file_a"],
            "train": [[0, 0], [0, 0]],
            "validation": [[0, 1]],
            "test": [[0, 2]],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            splits_path = Path(tmpdir) / "dataset_splits.json"
            with splits_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            with self.assertRaisesRegex(ValueError, "Duplicate sample reference"):
                load_splits(config, Path(tmpdir))

    def test_setup_device_cpu_returns_cpu(self) -> None:
        """CPU backend selection should stay explicit and deterministic."""
        self.assertEqual(setup_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
