#!/usr/bin/env python3
"""Build a final-model config by merging a tuning trial's winning architecture
with production training settings from a v2 base config.

Workflow:
    1. After a sweep finishes, read its leaderboard.json to identify the best
       transformer trial and the best LSTM trial.
    2. For each, run this script with --trial-dir pointing at the trial dir
       and --base-config pointing at the v2 config that supplies production
       optimizer/scheduler/loss settings.
    3. The output JSONC config is written to --out (default
       config/_final_run.jsonc) and is consumed by train.sh in TRIAL_DIR mode.

What carries over from the trial:
    - model_hyperparameters (the entire winning architecture block)

What carries over from the base config:
    - data_paths_config, data_specification, normalization, precision,
      training_hyperparameters (notably full-data epochs, patience, warmup,
      EMA, lr, weight_decay, scheduler, batch_size), loss, miscellaneous_settings

What this script forces:
    - training_hyperparameters.dataset_fraction_to_use = 1.0   (full data)
    - miscellaneous_settings.execution_mode             = "train"
    - miscellaneous_settings.rebuild_processed_data     = False
    - output_paths_config.fixed_model_foldername        = derived from trial

If the v2 base config is later edited (e.g. lr changes), the next
_final_run.jsonc will pick up the change. That is intentional: the v2 config
is the single source of truth for production training settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from utils import load_config, save_json, validate_config

logger = logging.getLogger(__name__)

DEFAULT_OUT = PROJECT_ROOT / "config" / "_final_run.jsonc"


def _read_trial_config(trial_dir: Path) -> Dict[str, Any]:
    cfg_path = trial_dir / "trial_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing trial_config.json under {trial_dir}.")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _derive_foldername(trial_dir: Path, model_type: str) -> str:
    trial_name = trial_dir.name
    if not trial_name.startswith("trial_"):
        raise ValueError(
            f"Trial dir name '{trial_name}' does not match expected 'trial_NNNN'."
        )
    study_name = trial_dir.parent.name
    return f"final_{model_type}_{study_name}_{trial_name}"


def build_final_config(
    trial_dir: Path,
    base_config_path: Path,
) -> Dict[str, Any]:
    """Merge trial model block with base-config production settings."""
    trial_cfg = _read_trial_config(trial_dir)
    base_cfg = load_config(base_config_path)

    if "model_hyperparameters" not in trial_cfg:
        raise KeyError(
            f"trial_config.json under {trial_dir} has no 'model_hyperparameters' block."
        )
    trial_model = trial_cfg["model_hyperparameters"]
    model_type = str(trial_model.get("model_type", "")).lower()
    if model_type not in {"transformer", "lstm"}:
        raise ValueError(
            f"Unexpected model_type '{model_type}' in trial config "
            f"{trial_dir}/trial_config.json."
        )

    # Sanity-check the base config matches the trial's model type. Mismatches
    # are usually a copy-paste error and would silently produce a misnamed run.
    base_model_type = str(base_cfg["model_hyperparameters"].get("model_type", "")).lower()
    if base_model_type != model_type:
        logger.warning(
            "Base config model_type=%s but trial model_type=%s. The trial's "
            "model block wins, but consider passing --base-config matching "
            "the trial's family.",
            base_model_type,
            model_type,
        )

    final_cfg = copy.deepcopy(base_cfg)
    final_cfg["model_hyperparameters"] = copy.deepcopy(trial_model)

    final_cfg["training_hyperparameters"]["dataset_fraction_to_use"] = 1.0
    final_cfg["miscellaneous_settings"]["execution_mode"] = "train"
    final_cfg["miscellaneous_settings"]["rebuild_processed_data"] = False
    final_cfg["output_paths_config"]["fixed_model_foldername"] = _derive_foldername(
        trial_dir, model_type
    )

    validate_config(final_cfg)
    return final_cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a final-model JSONC config from a tuning trial directory. "
            "Combines the trial's winning architecture with production training "
            "settings from a v2 base config."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        required=True,
        help="Path to a Problemulator/models/tune_<study>/trial_NNNN directory.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help=(
            "Path to a v2 base config (transformer_v2.jsonc or lstm_v2.jsonc) "
            "that supplies production training/loss/optimizer settings."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path for the final-run JSONC config.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    trial_dir = args.trial_dir.resolve()
    base_config = args.base_config.resolve()
    out_path = args.out.resolve()

    if not trial_dir.is_dir():
        raise NotADirectoryError(f"Trial directory not found: {trial_dir}")
    if not base_config.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_cfg = build_final_config(trial_dir, base_config)
    if not save_json(final_cfg, out_path):
        raise RuntimeError(f"Failed to write final config to {out_path}.")

    logger.info(
        "Wrote final-run config: %s (model_type=%s, foldername=%s)",
        out_path,
        final_cfg["model_hyperparameters"]["model_type"],
        final_cfg["output_paths_config"]["fixed_model_foldername"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
