#!/usr/bin/env python3
"""Unit tests for config validation and explicit backend behavior."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hardware import setup_device
from utils import get_precision_config, validate_config


def _load_checked_in_config() -> dict:
    """Load the checked-in JSONC config as plain JSON for test mutation."""
    config_path = PROJECT_ROOT / "config" / "config.jsonc"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    def test_setup_device_cpu_returns_cpu(self) -> None:
        """CPU backend selection should stay explicit and deterministic."""
        self.assertEqual(setup_device("cpu").type, "cpu")


if __name__ == "__main__":
    unittest.main()
