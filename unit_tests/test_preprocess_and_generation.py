#!/usr/bin/env python3
"""Unit tests for preprocessing and generation-path schema preservation."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
GEN_DATA_DIR = REPO_ROOT / "gen_data"
for path in (SRC_DIR, GEN_DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_profiles import (
    MAX_VALID_TEMPERATURE_K,
    SCALAR_KEYS,
    _append_profile_batch as append_generated_profile_batch,
    _create_profile_hdf5,
)
from normalizer import DataNormalizer
from preprocess import preprocess_data
from run_picaso import (
    INPUT_KEYS,
    OUTPUT_KEYS,
    _append_output_batch,
    _collate_valid_results,
    _create_output_hdf5,
    _iter_profile_records,
    _load_profile_chunk,
    _select_input_batch_rows,
)
from utils import load_config


def _load_checked_in_config() -> dict:
    """Load the checked-in config for test mutation."""
    config_path = PROJECT_ROOT / "config" / "config.jsonc"
    return load_config(config_path)


class PreprocessAndGenerationTests(unittest.TestCase):
    """Cover the optimized preprocessing and generation paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = _load_checked_in_config()

    def _build_cpu_test_config(self) -> dict:
        """Return a config tailored for small CPU-side preprocessing fixtures."""
        config = copy.deepcopy(self.base_config)
        config["miscellaneous_settings"]["device_backend"] = "cpu"
        config["miscellaneous_settings"]["torch_compile"] = False
        config["miscellaneous_settings"]["shard_size"] = 2
        config["miscellaneous_settings"]["hdf5_read_chunk_size"] = 1
        config["miscellaneous_settings"]["num_workers"] = 0
        config["model_hyperparameters"]["max_sequence_length"] = 4
        return config

    def test_preprocess_data_preserves_schema_padding_and_normalized_values(self) -> None:
        """Small synthetic preprocessing runs should preserve outputs and normalization."""
        config = self._build_cpu_test_config()
        padding_value = float(config["data_specification"]["padding_value"])
        clamp = float(config["normalization"]["normalized_value_clamp"])

        raw_arrays = {
            "pressure_bar": np.asarray(
                [
                    [1.0e-5, 1.0e-4, 1.0e-3],
                    [2.0e-5, 2.0e-4, 2.0e-3],
                    [3.0e-5, 3.0e-4, 3.0e-3],
                ],
                dtype=np.float32,
            ),
            "temperature_k": np.asarray(
                [
                    [500.0, 600.0, 700.0],
                    [550.0, 650.0, 750.0],
                    [600.0, 700.0, 800.0],
                ],
                dtype=np.float32,
            ),
            "orbital_distance_m": np.asarray([1.0e11, 1.5e11, 2.0e11], dtype=np.float32),
            "net_thermal_flux": np.asarray(
                [
                    [10.0, 20.0, 30.0],
                    [11.0, 21.0, 31.0],
                    [12.0, 22.0, 32.0],
                ],
                dtype=np.float32,
            ),
            "net_reflected_flux": np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [1.1, 2.1, 3.1],
                    [1.2, 2.2, 3.2],
                ],
                dtype=np.float32,
            ),
        }
        splits = {
            "train": [("fixture", 0)],
            "validation": [("fixture", 1)],
            "test": [("fixture", 2)],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "fixture.h5"
            processed_dir = tmp_path / "processed"

            with h5py.File(raw_path, "w") as h5_file:
                for key, value in raw_arrays.items():
                    h5_file.create_dataset(key, data=value)

            success = preprocess_data(
                config=config,
                raw_hdf5_paths=[raw_path],
                splits=splits,
                processed_dir=processed_dir,
            )

            self.assertTrue(success)

            train_seq = np.load(processed_dir / "train" / "sequence_inputs" / "shard_000000.npy")
            train_tgt = np.load(processed_dir / "train" / "targets" / "shard_000000.npy")
            train_glb = np.load(processed_dir / "train" / "globals" / "shard_000000.npy")
            with (processed_dir / "train" / "metadata.json").open("r", encoding="utf-8") as handle:
                train_meta = json.load(handle)
            with (processed_dir / "normalization_metadata.json").open(
                "r", encoding="utf-8"
            ) as handle:
                norm_meta = json.load(handle)

            self.assertEqual(train_meta["sequence_length"], 4)
            self.assertEqual(tuple(train_seq.shape), (1, 4, 2))
            self.assertEqual(tuple(train_tgt.shape), (1, 4, 2))
            self.assertEqual(tuple(train_glb.shape), (1, 1))
            self.assertEqual(train_seq.dtype, np.float32)
            self.assertEqual(train_tgt.dtype, np.float32)
            self.assertEqual(train_glb.dtype, np.float32)
            self.assertTrue(np.all(train_seq[0, 3, :] == padding_value))
            self.assertTrue(np.all(train_tgt[0, 3, :] == padding_value))

            norm_methods = norm_meta["normalization_methods"]
            norm_stats = norm_meta["per_key_stats"]
            expected_inputs = np.stack(
                [
                    DataNormalizer.normalize_array(
                        raw_arrays[var][0].copy(),
                        norm_methods[var],
                        norm_stats[var],
                        normalized_value_clamp=clamp,
                    )
                    for var in config["data_specification"]["input_variables"]
                ],
                axis=-1,
            )
            expected_targets = np.stack(
                [
                    DataNormalizer.normalize_array(
                        raw_arrays[var][0].copy(),
                        norm_methods[var],
                        norm_stats[var],
                        normalized_value_clamp=clamp,
                    )
                    for var in config["data_specification"]["target_variables"]
                ],
                axis=-1,
            )
            expected_global = np.stack(
                [
                    DataNormalizer.normalize_array(
                        raw_arrays[var][0:1].copy(),
                        norm_methods[var],
                        norm_stats[var],
                        normalized_value_clamp=clamp,
                    )[0]
                    for var in config["data_specification"]["global_variables"]
                ],
                axis=-1,
            )

            np.testing.assert_allclose(train_seq[0, :3, :], expected_inputs, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(train_tgt[0, :3, :], expected_targets, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(train_glb[0], expected_global, rtol=1e-6, atol=1e-6)

    def test_create_profiles_hdf5_schema_is_stable(self) -> None:
        """Chunk-oriented generation writes should preserve the established HDF5 schema."""
        n_pressure = 4
        batch_one = {
            "pressure_bar": np.asarray([[1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]], dtype=np.float32),
            "temperature_k": np.asarray([[400.0, 500.0, 600.0, 700.0]], dtype=np.float32),
            "planet_radius_m": np.asarray([7.0e7], dtype=np.float32),
            "planet_mass_kg": np.asarray([1.9e27], dtype=np.float32),
            "star_radius_m": np.asarray([7.0e8], dtype=np.float32),
            "star_temperature_k": np.asarray([6000.0], dtype=np.float32),
            "star_metallicity": np.asarray([0.0], dtype=np.float32),
            "star_logg": np.asarray([4.5], dtype=np.float32),
            "orbital_distance_m": np.asarray([1.0e11], dtype=np.float32),
        }
        batch_two = {
            key: np.concatenate([value, value + value.dtype.type(1)], axis=0)
            if value.ndim == 1
            else np.concatenate([value, value + 1.0], axis=0)
            for key, value in batch_one.items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.h5"
            _create_profile_hdf5(output_path, n_pressure=n_pressure, chunk_rows=2)

            with h5py.File(output_path, "a") as h5_file:
                self.assertEqual(append_generated_profile_batch(h5_file, batch_one), 1)
                self.assertEqual(append_generated_profile_batch(h5_file, batch_two), 2)
                h5_file.attrs["requested_profiles"] = 3
                h5_file.attrs["valid_profiles"] = 3
                h5_file.attrs["max_valid_temperature_k"] = float(MAX_VALID_TEMPERATURE_K)

            with h5py.File(output_path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["generator"], "create_profiles.py")
                self.assertEqual(h5_file.attrs["schema_version"], "1.0")
                self.assertEqual(int(h5_file.attrs["pressure_levels"]), n_pressure)
                self.assertEqual(int(h5_file.attrs["requested_profiles"]), 3)
                self.assertEqual(int(h5_file.attrs["valid_profiles"]), 3)
                self.assertEqual(float(h5_file.attrs["max_valid_temperature_k"]), MAX_VALID_TEMPERATURE_K)
                for key in ("pressure_bar", "temperature_k"):
                    self.assertIn(key, h5_file)
                    self.assertEqual(h5_file[key].dtype, np.float32)
                    self.assertEqual(h5_file[key].shape[0], 3)
                for key in SCALAR_KEYS:
                    self.assertIn(key, h5_file)
                    self.assertEqual(h5_file[key].dtype, np.float32)
                    self.assertEqual(h5_file[key].shape[0], 3)

    def test_run_picaso_output_schema_is_stable(self) -> None:
        """PICASO orchestration helpers should preserve output attrs, keys, and dtypes."""
        n_rows = 2
        n_levels = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.h5"
            output_path = tmp_path / "output.h5"

            with h5py.File(input_path, "w") as h5_file:
                h5_file.create_dataset(
                    "pressure_bar",
                    data=np.asarray(
                        [
                            [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
                            [2.0e-5, 2.0e-4, 2.0e-3, 2.0e-2],
                        ],
                        dtype=np.float32,
                    ),
                )
                h5_file.create_dataset(
                    "temperature_k",
                    data=np.asarray(
                        [
                            [500.0, 600.0, 700.0, 800.0],
                            [550.0, 650.0, 750.0, 850.0],
                        ],
                        dtype=np.float32,
                    ),
                )
                for key, value in {
                    "planet_radius_m": [7.0e7, 7.1e7],
                    "planet_mass_kg": [1.9e27, 1.91e27],
                    "star_radius_m": [7.0e8, 7.0e8],
                    "star_temperature_k": [6000.0, 6000.0],
                    "star_metallicity": [0.0, 0.0],
                    "star_logg": [4.5, 4.5],
                    "orbital_distance_m": [1.0e11, 1.2e11],
                }.items():
                    h5_file.create_dataset(key, data=np.asarray(value, dtype=np.float32))

            with h5py.File(input_path, "r") as h5_file:
                profile_batch = _load_profile_chunk(h5_file, 0, n_rows)

            records = list(_iter_profile_records(profile_batch))
            self.assertEqual(len(records), n_rows)
            self.assertEqual(records[0].pressure_bar.shape[0], n_levels)

            results_raw = [
                {
                    "pressure_bar_picaso": np.asarray([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2], dtype=np.float32),
                    "net_thermal_flux": np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
                    "net_reflected_flux": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                },
                None,
            ]
            valid_indices, valid_result_batch = _collate_valid_results(results_raw)
            self.assertEqual(valid_indices.tolist(), [0])
            self.assertIsNotNone(valid_result_batch)
            assert valid_result_batch is not None
            valid_profile_batch = _select_input_batch_rows(profile_batch, valid_indices)

            output_file = _create_output_hdf5(
                output_path,
                {key: valid_profile_batch[key][0] for key in INPUT_KEYS},
                {key: valid_result_batch[key][0] for key in OUTPUT_KEYS},
                chunk_rows=2,
                opacity_method="preweighted",
                opacity_source=tmp_path / "mock_opacity.hdf5",
                stellar_mode="database",
                stellar_value="phoenix",
            )
            _append_output_batch(output_file, valid_profile_batch, valid_result_batch)
            output_file.attrs["requested_profiles"] = n_rows
            output_file.attrs["successful_profiles"] = 1
            output_file.close()

            with h5py.File(output_path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["generator"], "run_picaso.py")
                self.assertEqual(h5_file.attrs["schema_version"], "1.0")
                self.assertEqual(h5_file.attrs["opacity_method"], "preweighted")
                self.assertEqual(h5_file.attrs["stellar_source_mode"], "database")
                self.assertEqual(h5_file.attrs["stellar_source_value"], "phoenix")
                self.assertEqual(float(h5_file.attrs["shortwave_incidence_mu"]), 1.0)
                self.assertEqual(h5_file.attrs["chemistry_mode"], "premixed_ck")
                self.assertEqual(int(h5_file.attrs["requested_profiles"]), n_rows)
                self.assertEqual(int(h5_file.attrs["successful_profiles"]), 1)
                for key in INPUT_KEYS + OUTPUT_KEYS:
                    self.assertIn(key, h5_file)
                    self.assertEqual(h5_file[key].dtype, np.float32)
                    self.assertEqual(h5_file[key].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
