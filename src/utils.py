#!/usr/bin/env python3
"""
utils.py - Helper functions for configuration, logging, and data handling.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# JSON5 is required for .jsonc config files (comment support).
try:
    import json5 as _json_backend
except ImportError:
    _json_backend = None

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
METADATA_FILENAME = "normalization_metadata.json"
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"  # Handle UTF-8 with BOM
ALLOWED_NORMALIZATION_METHODS = {
    "iqr",
    "log-min-max",
    "max-out",
    "signed-log",
    "scaled_signed_offset_log",
    "symlog",
    "standard",
    "log-standard",
    "bool",
}
REQUIRED_SPLIT_KEYS = {"train", "validation", "test"}
REQUIRED_CONFIG_SECTIONS = {
    "miscellaneous_settings",
    "data_paths_config",
    "data_specification",
    "normalization",
    "precision",
    "model_hyperparameters",
    "training_hyperparameters",
    "output_paths_config",
}
SUPPORTED_DEVICE_BACKENDS = {"cpu", "mps", "cuda"}
SUPPORTED_OPTIMIZER = "adamw"
SUPPORTED_SCHEDULER = "cosine"
SUPPORTED_FLOAT32_MATMUL_PRECISION = {"highest", "high", "medium", "none"}
SUPPORTED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
SUPPORTED_DATASET_LOADING_MODES = {"auto", "ram", "disk"}
TORCH_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

logger = logging.getLogger(__name__)

def parse_torch_dtype(
    dtype_name: str,
    *,
    allow_none: bool = False,
    field_name: str = "dtype",
) -> Optional[torch.dtype]:
    """Parse a configured dtype name into a torch.dtype."""
    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string.")

    lowered = dtype_name.strip().lower()
    if allow_none and lowered == "none":
        return None

    if lowered not in TORCH_DTYPE_MAP:
        allowed = sorted(TORCH_DTYPE_MAP.keys()) + (["none"] if allow_none else [])
        raise ValueError(f"Unsupported {field_name} '{dtype_name}'. Allowed: {allowed}.")

    return TORCH_DTYPE_MAP[lowered]


def get_precision_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return validated precision settings with resolved torch dtypes."""
    precision = config["precision"]
    training = config["training_hyperparameters"]
    backend = str(config["miscellaneous_settings"]["device_backend"]).lower()

    input_name = str(precision["input_dtype"]).lower()
    stats_name = str(precision["stats_accumulation_dtype"]).lower()
    model_name = str(precision["model_dtype"]).lower()
    forward_name = str(precision["forward_dtype"]).lower()
    loss_name = str(precision["loss_dtype"]).lower()
    optimizer_name = str(precision["optimizer_state_dtype"]).lower()
    amp_name = str(precision["amp_autocast_dtype"]).lower()
    matmul_name = str(precision["float32_matmul_precision"]).lower()

    input_dtype = parse_torch_dtype(input_name, field_name="precision.input_dtype")
    stats_dtype = parse_torch_dtype(stats_name, field_name="precision.stats_accumulation_dtype")
    model_dtype = parse_torch_dtype(model_name, field_name="precision.model_dtype")
    forward_dtype = parse_torch_dtype(forward_name, field_name="precision.forward_dtype")
    loss_dtype = parse_torch_dtype(loss_name, field_name="precision.loss_dtype")
    amp_dtype = parse_torch_dtype(
        amp_name, allow_none=True, field_name="precision.amp_autocast_dtype"
    )

    if stats_dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "precision.stats_accumulation_dtype must be float32 or float64."
        )

    use_amp = bool(training["use_amp"])
    if use_amp:
        if backend != "cuda":
            raise ValueError("AMP is supported only when device_backend='cuda'.")
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "use_amp=True requires amp_autocast_dtype to be float16 or bfloat16."
            )
        if model_dtype != torch.float32:
            raise ValueError(
                "use_amp=True requires model_dtype=float32 for stable AdamW training."
            )
    else:
        if amp_dtype is not None:
            raise ValueError("use_amp=False requires amp_autocast_dtype='none'.")

    if forward_dtype != model_dtype:
        raise ValueError(
            "precision.forward_dtype must match precision.model_dtype."
        )

    if matmul_name not in SUPPORTED_FLOAT32_MATMUL_PRECISION:
        raise ValueError(
            "precision.float32_matmul_precision must be one of "
            f"{sorted(SUPPORTED_FLOAT32_MATMUL_PRECISION)}."
        )

    # AdamW state dtype is coupled to parameter dtype in this codebase.
    if optimizer_name != model_name:
        raise ValueError(
            "precision.optimizer_state_dtype must match precision.model_dtype "
            "for AdamW in this implementation."
        )

    if backend == "mps":
        mps_unsupported: Dict[str, torch.dtype] = {
            "precision.input_dtype": input_dtype,
            "precision.stats_accumulation_dtype": stats_dtype,
            "precision.model_dtype": model_dtype,
            "precision.forward_dtype": forward_dtype,
            "precision.loss_dtype": loss_dtype,
        }
        float64_fields = sorted(
            name for name, dtype in mps_unsupported.items() if dtype == torch.float64
        )
        if float64_fields:
            raise ValueError(
                "MPS backend does not support float64 for these fields: "
                f"{float64_fields}."
            )

    return {
        "input_dtype": input_dtype,
        "input_dtype_name": input_name,
        "stats_dtype": stats_dtype,
        "stats_dtype_name": stats_name,
        "model_dtype": model_dtype,
        "model_dtype_name": model_name,
        "forward_dtype": forward_dtype,
        "forward_dtype_name": forward_name,
        "loss_dtype": loss_dtype,
        "loss_dtype_name": loss_name,
        "optimizer_state_dtype": model_dtype,
        "optimizer_state_dtype_name": optimizer_name,
        "amp_dtype": amp_dtype,
        "amp_dtype_name": amp_name,
        "float32_matmul_precision": matmul_name,
        "use_amp": use_amp,
    }


def setup_logging(
    level: int = logging.INFO,
    log_file: Union[str, Path, None] = None,
    force: bool = False,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional file path for logging output
        force: If True, remove existing handlers before setup
    """
    root_logger = logging.getLogger()
    
    # Force reset if requested
    if force:
        while root_logger.handlers:
            handler = root_logger.handlers.pop()
            handler.close()
    
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Add console handler if none exists
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        try:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_file_path, mode="a", encoding=UTF8_ENCODING
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Logging to console and file: {log_file_path.resolve()}")
        except OSError as e:
            print(f"Error setting up file logging for {log_file}: {e}. Using console only.")
    else:
        print("Logging to console only.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate a configuration file (JSON or JSON5).
    
    Args:
        path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        RuntimeError: If config is invalid or malformed
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # .jsonc files require json5 for comment support — no silent fallback.
        is_jsonc = config_path.suffix.lower() == ".jsonc"
        if is_jsonc and _json_backend is None:
            raise RuntimeError(
                f"Config file '{config_path}' uses .jsonc format but the json5 "
                "package is not installed. Install it: pip install json5"
            )

        with open(config_path, "r", encoding=UTF8_SIG_ENCODING) as f:
            if _json_backend is not None:
                config_dict = _json_backend.load(f)
            else:
                config_dict = json.load(f)

        validate_config(config_dict)

        backend = "JSON5" if _json_backend is not None else "JSON"
        logger.info(f"Loaded and validated {backend} config from {config_path}.")
        return config_dict

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate config {config_path}: {e}") from e


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a dictionary.")

    missing_sections = REQUIRED_CONFIG_SECTIONS - set(config.keys())
    if missing_sections:
        raise ValueError(
            f"Missing required config sections: {sorted(missing_sections)}"
        )

    def require_section(section: str) -> Dict[str, Any]:
        section_obj = config.get(section)
        if not isinstance(section_obj, dict):
            raise ValueError(f"Config section '{section}' must be a dictionary.")
        return section_obj

    def require_list_of_str(section: Dict[str, Any], key: str, *, allow_empty: bool = False) -> List[str]:
        value = section.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            raise ValueError(f"Config key '{key}' must be a list of non-empty strings.")
        if not allow_empty and not value:
            raise ValueError(f"Config key '{key}' cannot be empty.")
        return value

    def require_split_fractions(section: Dict[str, Any], key: str) -> Dict[str, float]:
        value = section.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Config key '{key}' must be a dictionary.")

        missing_keys = REQUIRED_SPLIT_KEYS - set(value.keys())
        if missing_keys:
            raise ValueError(
                f"Missing keys in '{key}': {sorted(missing_keys)}"
            )

        unknown_keys = set(value.keys()) - REQUIRED_SPLIT_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in '{key}': {sorted(unknown_keys)}"
            )

        split_fractions: Dict[str, float] = {}
        for split_name in ("train", "validation", "test"):
            fraction = value[split_name]
            if not isinstance(fraction, (int, float)):
                raise ValueError(
                    f"Split fraction '{key}.{split_name}' must be numeric."
                )

            fraction_f = float(fraction)
            if not (0.0 < fraction_f < 1.0):
                raise ValueError(
                    f"Split fraction '{key}.{split_name}' must be in (0, 1)."
                )
            split_fractions[split_name] = fraction_f

        if abs(sum(split_fractions.values()) - 1.0) > 1e-12:
            raise ValueError(f"Split fractions in '{key}' must sum to 1.0.")

        return split_fractions

    # data_paths_config
    data_paths = require_section("data_paths_config")
    h5_list = require_list_of_str(data_paths, "hdf5_dataset_filename")
    splits_file = data_paths.get("dataset_splits_filename")
    if not isinstance(splits_file, str) or not splits_file.strip():
        raise ValueError(
            "Config key 'data_paths_config.dataset_splits_filename' must be a non-empty string."
        )
    require_split_fractions(data_paths, "dataset_split_fractions")
    if not h5_list:
        raise ValueError("Config key 'hdf5_dataset_filename' cannot be empty.")

    # data_specification
    data_spec = require_section("data_specification")
    input_vars = require_list_of_str(data_spec, "input_variables")
    target_vars = require_list_of_str(data_spec, "target_variables")
    global_vars = require_list_of_str(data_spec, "global_variables", allow_empty=True)
    strictly_positive_vars = require_list_of_str(
        data_spec, "strictly_positive_variables", allow_empty=True
    )

    padding_value = data_spec.get("padding_value")
    if not isinstance(padding_value, (int, float)):
        raise ValueError("Config key 'data_specification.padding_value' must be numeric.")

    all_vars = input_vars + global_vars + target_vars
    if len(all_vars) != len(set(all_vars)):
        raise ValueError("Duplicate variable names found across input/global/target variables.")
    if len(strictly_positive_vars) != len(set(strictly_positive_vars)):
        raise ValueError(
            "Duplicate variable names found in data_specification.strictly_positive_variables."
        )
    unknown_positive = sorted(set(strictly_positive_vars) - set(all_vars))
    if unknown_positive:
        raise ValueError(
            "Unknown keys in data_specification.strictly_positive_variables: "
            f"{unknown_positive}"
        )

    variable_units = data_spec.get("variable_units")
    if not isinstance(variable_units, dict):
        raise ValueError("Config key 'data_specification.variable_units' must be a dictionary.")
    missing_units = sorted(set(all_vars) - set(variable_units.keys()))
    if missing_units:
        raise ValueError(
            "Each variable must have declared units in data_specification.variable_units. "
            f"Missing: {missing_units}"
        )
    unknown_units = sorted(set(variable_units.keys()) - set(all_vars))
    if unknown_units:
        raise ValueError(
            "Unknown keys in data_specification.variable_units: "
            f"{unknown_units}"
        )
    for var_name, unit in variable_units.items():
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(
                f"Unit for variable '{var_name}' must be a non-empty string."
            )

    # normalization
    normalization = require_section("normalization")
    key_methods = normalization.get("key_methods")
    if not isinstance(key_methods, dict):
        raise ValueError("Config key 'normalization.key_methods' must be a dictionary.")
    missing_methods = sorted(set(all_vars) - set(key_methods.keys()))
    if missing_methods:
        raise ValueError(
            "Each variable must have an explicit normalization method. "
            f"Missing: {missing_methods}"
        )
    unknown_methods = sorted(set(key_methods.keys()) - set(all_vars))
    if unknown_methods:
        raise ValueError(
            "Unknown keys in normalization.key_methods: "
            f"{unknown_methods}"
        )
    for var_name, method in key_methods.items():
        if not isinstance(method, str):
            raise ValueError(f"Normalization method for '{var_name}' must be a string.")
        lowered = method.lower()
        if lowered not in ALLOWED_NORMALIZATION_METHODS:
            raise ValueError(f"Unsupported normalization method '{method}' for '{var_name}'.")
    for required_key in (
        "epsilon",
        "padding_comparison_epsilon",
        "normalized_value_clamp",
        "log_min_max_clamp_min",
        "log_min_max_clamp_max",
        "symlog_min_threshold_multiplier",
        "stats_max_block_bytes",
        "stats_max_span_multiplier",
        "quantile_max_values_in_memory",
        "symlog_percentile",
        "stats_chunk_size",
    ):
        if required_key not in normalization:
            raise ValueError(f"Missing 'normalization.{required_key}' in config.")

    for required_float_key in (
        "epsilon",
        "padding_comparison_epsilon",
        "normalized_value_clamp",
        "log_min_max_clamp_min",
        "log_min_max_clamp_max",
        "symlog_min_threshold_multiplier",
        "symlog_percentile",
    ):
        if not isinstance(normalization[required_float_key], (int, float)):
            raise ValueError(f"'normalization.{required_float_key}' must be numeric.")
    if float(normalization["epsilon"]) <= 0:
        raise ValueError("'normalization.epsilon' must be > 0.")
    if float(normalization["padding_comparison_epsilon"]) <= 0:
        raise ValueError("'normalization.padding_comparison_epsilon' must be > 0.")
    if float(normalization["normalized_value_clamp"]) <= 0:
        raise ValueError("'normalization.normalized_value_clamp' must be > 0.")
    if float(normalization["symlog_min_threshold_multiplier"]) <= 0:
        raise ValueError("'normalization.symlog_min_threshold_multiplier' must be > 0.")
    if not (0.0 < float(normalization["symlog_percentile"]) <= 1.0):
        raise ValueError("'normalization.symlog_percentile' must be in (0, 1].")

    log_min_max_clamp_min = float(normalization["log_min_max_clamp_min"])
    log_min_max_clamp_max = float(normalization["log_min_max_clamp_max"])
    if log_min_max_clamp_min >= log_min_max_clamp_max:
        raise ValueError(
            "'normalization.log_min_max_clamp_min' must be less than "
            "'normalization.log_min_max_clamp_max'."
        )

    if (
        not isinstance(normalization["stats_max_block_bytes"], int)
        or normalization["stats_max_block_bytes"] <= 0
    ):
        raise ValueError("'normalization.stats_max_block_bytes' must be a positive integer.")
    if (
        not isinstance(normalization["stats_max_span_multiplier"], int)
        or normalization["stats_max_span_multiplier"] <= 0
    ):
        raise ValueError("'normalization.stats_max_span_multiplier' must be a positive integer.")
    if (
        not isinstance(normalization["quantile_max_values_in_memory"], int)
        or normalization["quantile_max_values_in_memory"] <= 0
    ):
        raise ValueError(
            "'normalization.quantile_max_values_in_memory' must be a positive integer."
        )
    if not isinstance(normalization["stats_chunk_size"], int) or normalization["stats_chunk_size"] <= 0:
        raise ValueError("'normalization.stats_chunk_size' must be a positive integer.")

    # precision
    precision = require_section("precision")
    required_precision_keys = {
        "input_dtype",
        "stats_accumulation_dtype",
        "model_dtype",
        "forward_dtype",
        "loss_dtype",
        "optimizer_state_dtype",
        "amp_autocast_dtype",
        "float32_matmul_precision",
    }
    missing_precision_keys = required_precision_keys - set(precision.keys())
    if missing_precision_keys:
        raise ValueError(
            f"Missing required precision keys: {sorted(missing_precision_keys)}"
        )
    for key in required_precision_keys:
        if not isinstance(precision[key], str) or not str(precision[key]).strip():
            raise ValueError(f"'precision.{key}' must be a non-empty string.")

    # model_hyperparameters
    model_params = require_section("model_hyperparameters")
    required_model_keys = {
        "d_model",
        "nhead",
        "num_encoder_layers",
        "dim_feedforward",
        "dropout",
        "attention_dropout",
        "film_clamp",
        "max_sequence_length",
        "output_head_divisor",
        "output_head_dropout_factor",
    }
    missing_model_keys = required_model_keys - set(model_params.keys())
    if missing_model_keys:
        raise ValueError(
            f"Missing required model_hyperparameters keys: {sorted(missing_model_keys)}"
        )

    d_model = model_params["d_model"]
    nhead = model_params["nhead"]
    if not isinstance(d_model, int) or d_model <= 0:
        raise ValueError("'model_hyperparameters.d_model' must be a positive integer.")
    if not isinstance(nhead, int) or nhead <= 0:
        raise ValueError("'model_hyperparameters.nhead' must be a positive integer.")
    if d_model % nhead != 0:
        raise ValueError(f"'d_model' ({d_model}) must be divisible by 'nhead' ({nhead}).")
    if not isinstance(model_params["max_sequence_length"], int) or model_params["max_sequence_length"] <= 0:
        raise ValueError("'model_hyperparameters.max_sequence_length' must be a positive integer.")

    for positive_key in ("num_encoder_layers", "dim_feedforward", "output_head_divisor"):
        value = model_params[positive_key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"'model_hyperparameters.{positive_key}' must be a positive integer."
            )

    for prob_key in ("dropout", "attention_dropout", "output_head_dropout_factor"):
        value = model_params[prob_key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"'model_hyperparameters.{prob_key}' must be numeric.")
        value_f = float(value)
        if not (0.0 <= value_f <= 1.0):
            raise ValueError(f"'model_hyperparameters.{prob_key}' must be in [0, 1].")

    film_clamp = model_params["film_clamp"]
    if not isinstance(film_clamp, (int, float)):
        raise ValueError("'model_hyperparameters.film_clamp' must be numeric.")
    if float(film_clamp) <= 0.0:
        raise ValueError("'model_hyperparameters.film_clamp' must be > 0.")

    output_head_divisor = int(model_params["output_head_divisor"])
    if output_head_divisor > d_model:
        raise ValueError(
            "'model_hyperparameters.output_head_divisor' must be <= d_model "
            f"to keep output head width >= 1 (got divisor={output_head_divisor}, d_model={d_model})."
        )

    # training_hyperparameters
    train_params = require_section("training_hyperparameters")
    required_train_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "gradient_clip_val",
        "scheduler_type",
        "min_lr",
        "warmup_epochs",
        "warmup_start_factor",
        "early_stopping_patience",
        "min_delta",
        "use_amp",
        "dataset_fraction_to_use",
    }
    missing_train_keys = required_train_keys - set(train_params.keys())
    if missing_train_keys:
        raise ValueError(
            f"Missing required training_hyperparameters keys: {sorted(missing_train_keys)}"
        )

    optimizer_name = str(train_params["optimizer"]).lower()
    if optimizer_name != SUPPORTED_OPTIMIZER:
        raise ValueError(f"Only '{SUPPORTED_OPTIMIZER}' optimizer is supported.")

    scheduler_name = str(train_params["scheduler_type"]).lower()
    if scheduler_name != SUPPORTED_SCHEDULER:
        raise ValueError(f"Only '{SUPPORTED_SCHEDULER}' scheduler_type is supported.")

    if "gradient_accumulation_steps" in train_params:
        raise ValueError(
            "gradient_accumulation_steps is no longer supported. "
            "Remove it from training_hyperparameters."
        )

    for positive_int_key in ("epochs", "batch_size", "early_stopping_patience"):
        value = train_params[positive_int_key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"'training_hyperparameters.{positive_int_key}' must be a positive integer."
            )
    if not isinstance(train_params["warmup_epochs"], int):
        raise ValueError("'training_hyperparameters.warmup_epochs' must be an integer.")

    for non_negative_float_key in (
        "weight_decay",
        "gradient_clip_val",
        "min_lr",
        "min_delta",
    ):
        value = train_params[non_negative_float_key]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"'training_hyperparameters.{non_negative_float_key}' must be numeric."
            )
        if float(value) < 0.0:
            raise ValueError(
                f"'training_hyperparameters.{non_negative_float_key}' must be >= 0."
            )

    learning_rate = train_params["learning_rate"]
    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0.0:
        raise ValueError("'training_hyperparameters.learning_rate' must be > 0.")

    warmup_start_factor = train_params["warmup_start_factor"]
    if not isinstance(warmup_start_factor, (int, float)):
        raise ValueError("'training_hyperparameters.warmup_start_factor' must be numeric.")
    if not (0.0 < float(warmup_start_factor) <= 1.0):
        raise ValueError("'training_hyperparameters.warmup_start_factor' must be in (0, 1].")

    if not isinstance(train_params["use_amp"], bool):
        raise ValueError("'training_hyperparameters.use_amp' must be a boolean.")

    warmup_epochs = int(train_params["warmup_epochs"])
    epochs = int(train_params["epochs"])
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be >= 0 and strictly less than epochs.")

    if float(train_params["min_lr"]) > float(learning_rate):
        raise ValueError("'training_hyperparameters.min_lr' must be <= learning_rate.")

    dataset_fraction = float(train_params["dataset_fraction_to_use"])
    if not (0.0 < dataset_fraction <= 1.0):
        raise ValueError("dataset_fraction_to_use must be in (0, 1].")

    # miscellaneous_settings
    misc = require_section("miscellaneous_settings")
    required_misc_keys = {
        "random_seed",
        "num_workers",
        "detect_anomaly",
        "shard_size",
        "hdf5_read_chunk_size",
        "device_backend",
        "torch_compile",
        "compile_mode",
        "execution_mode",
        "rebuild_processed_data",
        "dataset_loading_mode",
        "dataset_max_cached_shards",
        "dataset_large_shard_mmap_bytes",
        "dataset_ram_safety_fraction",
        "dataset_copy_mmap_slices",
    }
    missing_misc_keys = required_misc_keys - set(misc.keys())
    if missing_misc_keys:
        raise ValueError(
            f"Missing required miscellaneous_settings keys: {sorted(missing_misc_keys)}"
        )
    if not isinstance(misc["random_seed"], int):
        raise ValueError("'miscellaneous_settings.random_seed' must be an integer.")
    if not isinstance(misc["num_workers"], int) or misc["num_workers"] < 0:
        raise ValueError("'miscellaneous_settings.num_workers' must be >= 0.")
    if not isinstance(misc["detect_anomaly"], bool):
        raise ValueError("'miscellaneous_settings.detect_anomaly' must be a boolean.")
    if not isinstance(misc["shard_size"], int) or misc["shard_size"] <= 0:
        raise ValueError("'miscellaneous_settings.shard_size' must be a positive integer.")
    if (
        not isinstance(misc["hdf5_read_chunk_size"], int)
        or misc["hdf5_read_chunk_size"] <= 0
    ):
        raise ValueError(
            "'miscellaneous_settings.hdf5_read_chunk_size' must be a positive integer."
        )
    if misc["hdf5_read_chunk_size"] >= misc["shard_size"]:
        raise ValueError(
            "'miscellaneous_settings.hdf5_read_chunk_size' must be less than "
            "'miscellaneous_settings.shard_size'."
        )

    if not isinstance(misc["torch_compile"], bool):
        raise ValueError("'miscellaneous_settings.torch_compile' must be a boolean.")
    compile_mode = misc["compile_mode"]
    if not isinstance(compile_mode, str) or not compile_mode.strip():
        raise ValueError("'miscellaneous_settings.compile_mode' must be a non-empty string.")
    if compile_mode not in SUPPORTED_COMPILE_MODES:
        raise ValueError(
            "'miscellaneous_settings.compile_mode' must be one of "
            f"{sorted(SUPPORTED_COMPILE_MODES)}."
        )

    device_backend = str(misc["device_backend"]).lower()
    if device_backend not in SUPPORTED_DEVICE_BACKENDS:
        raise ValueError(
            "miscellaneous_settings.device_backend must be one of "
            f"{sorted(SUPPORTED_DEVICE_BACKENDS)}."
        )
    if bool(misc["torch_compile"]) and device_backend != "cuda":
        raise ValueError("torch_compile=true requires miscellaneous_settings.device_backend='cuda'.")

    if not isinstance(misc["rebuild_processed_data"], bool):
        raise ValueError("'miscellaneous_settings.rebuild_processed_data' must be a boolean.")

    loading_mode = str(misc["dataset_loading_mode"]).lower()
    if loading_mode not in SUPPORTED_DATASET_LOADING_MODES:
        raise ValueError(
            "'miscellaneous_settings.dataset_loading_mode' must be one of "
            f"{sorted(SUPPORTED_DATASET_LOADING_MODES)}."
        )
    if (
        not isinstance(misc["dataset_max_cached_shards"], int)
        or misc["dataset_max_cached_shards"] <= 0
    ):
        raise ValueError(
            "'miscellaneous_settings.dataset_max_cached_shards' must be a positive integer."
        )
    if (
        not isinstance(misc["dataset_large_shard_mmap_bytes"], int)
        or misc["dataset_large_shard_mmap_bytes"] <= 0
    ):
        raise ValueError(
            "'miscellaneous_settings.dataset_large_shard_mmap_bytes' must be a positive integer."
        )
    ram_safety = misc["dataset_ram_safety_fraction"]
    if not isinstance(ram_safety, (int, float)) or not (0.0 < float(ram_safety) <= 1.0):
        raise ValueError(
            "'miscellaneous_settings.dataset_ram_safety_fraction' must be in (0, 1]."
        )
    if not isinstance(misc["dataset_copy_mmap_slices"], bool):
        raise ValueError("'miscellaneous_settings.dataset_copy_mmap_slices' must be a boolean.")

    execution_mode = misc["execution_mode"]
    if execution_mode not in {"normalize", "train"}:
        raise ValueError(
            "miscellaneous_settings.execution_mode must be one of "
            "{'normalize', 'train'}."
        )

    # Validate precision combinations now that training params are known.
    precision_cfg = get_precision_config(config)
    if precision_cfg["input_dtype"] not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(
            "precision.input_dtype must be float16, float32, or float64 so "
            "processed shards can be written as NumPy arrays."
        )

    # output_paths_config
    output_paths = require_section("output_paths_config")
    model_folder = output_paths.get("fixed_model_foldername")
    if not isinstance(model_folder, str) or not model_folder.strip():
        raise ValueError(
            "Config key 'output_paths_config.fixed_model_foldername' must be a non-empty string."
        )


def ensure_dirs(*paths: Union[str, Path, None]) -> bool:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths
        
    Returns:
        True if successful, False otherwise
    """
    try:
        for path in paths:
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for NumPy/PyTorch types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj.resolve())
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def save_json(data: Dict[str, Any], path: Union[str, Path], compact: bool = False) -> bool:
    """
    Save dictionary to JSON file with custom serialization.

    Args:
        data: Dictionary to save
        path: Output file path
        compact: If True, use minimal formatting for smaller files

    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = Path(path)
        ensure_dirs(json_path.parent)

        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            if compact:
                # Minimal formatting for compact files
                json.dump(data, f, default=_json_serializer,
                          ensure_ascii=False, separators=(',', ':'))
            else:
                # Human-readable formatting
                json.dump(data, f, indent=2, default=_json_serializer,
                          ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return True

    except (OSError, TypeError) as e:
        logger.error(f"Failed to save JSON to {path}: {e}", exc_info=True)
        return False


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Global random seed set to {seed}.")


def get_config_str(config: Dict[str, Any], section: str, key: str, op_desc: str) -> str:
    """
    Safely extract a string value from nested config dictionary.
    
    Args:
        config: Configuration dictionary
        section: Section name in config
        key: Key within section
        op_desc: Operation description for error messages
        
    Returns:
        The string value
        
    Raises:
        ValueError: If section/key missing or invalid
    """
    if section not in config or not isinstance(config[section], dict):
        raise ValueError(f"Config section '{section}' missing or invalid for {op_desc}.")
    
    path_val = config[section].get(key)
    if not isinstance(path_val, str) or not path_val.strip():
        raise ValueError(f"Config key '{key}' in '{section}' missing or empty for {op_desc}.")
    
    return path_val.strip()


def load_splits(
    config: Dict[str, Any],
    data_root_dir: Path,
) -> Tuple[Dict[str, List[Tuple[str, int]]], Path]:
    """
    Load dataset splits from required compact split file.

    Missing file is a hard failure at this utility level.

    Args:
        config: Configuration dictionary
        data_root_dir: Root data directory

    Returns:
        Tuple of (splits dictionary, splits file path)
    """
    splits_filename = get_config_str(
        config, "data_paths_config", "dataset_splits_filename", "dataset splits"
    )
    splits_path = data_root_dir / splits_filename

    logger.info(f"Loading dataset splits from: {splits_path}")
    if not splits_path.is_file():
        raise FileNotFoundError(
            f"Required splits file is missing: {splits_path}. "
            "Generate it with python src/generate_splits.py "
            "or call through the main pipeline, which auto-generates missing splits."
        )

    with open(splits_path, "r", encoding=UTF8_ENCODING) as f:
        loaded_data = json.load(f)

    if not isinstance(loaded_data, dict):
        raise ValueError("Splits file must be a JSON object.")

    if "file_stems" not in loaded_data:
        raise ValueError(
            "Legacy split format is not supported. "
            "Expected compact format with 'file_stems'."
        )

    required_keys = REQUIRED_SPLIT_KEYS | {"file_stems"}
    missing_keys = required_keys - set(loaded_data.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in compact split file: {sorted(missing_keys)}")

    file_stems = loaded_data["file_stems"]
    if not isinstance(file_stems, list) or not all(isinstance(s, str) and s for s in file_stems):
        raise ValueError("'file_stems' must be a list of non-empty strings.")
    if len(file_stems) != len(set(file_stems)):
        raise ValueError("'file_stems' must not contain duplicates.")

    seen_items: Dict[Tuple[int, int], str] = {}
    for split_name in ("train", "validation", "test"):
        items = loaded_data[split_name]
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"Split '{split_name}' must be a non-empty list.")
        split_seen: set[Tuple[int, int]] = set()
        for item in items:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], int)
                or not isinstance(item[1], int)
            ):
                raise ValueError(
                    f"Invalid item in split '{split_name}': {item!r}. "
                    "Expected [file_stem_index, sample_index]."
                )
            if item[0] < 0 or item[0] >= len(file_stems):
                raise ValueError(f"Invalid file_stem index {item[0]} in split '{split_name}'.")
            if item[1] < 0:
                raise ValueError(f"Negative sample index {item[1]} in split '{split_name}'.")

            sample_ref = (item[0], item[1])
            if sample_ref in split_seen:
                raise ValueError(
                    f"Duplicate sample reference in split '{split_name}': "
                    f"({file_stems[item[0]]}, {item[1]})."
                )

            previous_split = seen_items.get(sample_ref)
            if previous_split is not None:
                raise ValueError(
                    "Split partitions must be disjoint; sample "
                    f"({file_stems[item[0]]}, {item[1]}) appears in both "
                    f"'{previous_split}' and '{split_name}'."
                )

            split_seen.add(sample_ref)
            seen_items[sample_ref] = split_name

    splits = decompress_splits(loaded_data)

    logger.info(f"Loaded splits from {splits_path}")
    logger.info(
        f"Split sizes: {len(splits['train'])} train, "
        f"{len(splits['validation'])} val, {len(splits['test'])} test."
    )
    return splits, splits_path


def compress_splits(splits: Dict[str, List[Tuple[str, int]]]) -> Dict:
    """Convert verbose format to compact format."""
    # Extract unique file stems
    file_stems = sorted(set(stem for split in splits.values()
                            for stem, _ in split))
    stem_to_idx = {stem: i for i, stem in enumerate(file_stems)}

    compressed = {
        "file_stems": file_stems,
        "train": [[stem_to_idx[s], i] for s, i in splits["train"]],
        "validation": [[stem_to_idx[s], i] for s, i in splits["validation"]],
        "test": [[stem_to_idx[s], i] for s, i in splits["test"]]
    }
    return compressed

def decompress_splits(compressed: Dict) -> Dict[str, List[Tuple[str, int]]]:
    """Convert compact format back to verbose format."""
    file_stems = compressed["file_stems"]
    return {
        "train": [(file_stems[s], i) for s, i in compressed["train"]],
        "validation": [(file_stems[s], i) for s, i in compressed["validation"]],
        "test": [(file_stems[s], i) for s, i in compressed["test"]]
    }

__all__ = [
    "TORCH_DTYPE_MAP",
    "LOG_FORMAT",
    "METADATA_FILENAME",
    "parse_torch_dtype",
    "get_precision_config",
    "setup_logging",
    "load_config",
    "validate_config",
    "ensure_dirs",
    "save_json",
    "seed_everything",
    "get_config_str",
    "load_splits",
    "compress_splits",
    "decompress_splits",
]
