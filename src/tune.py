#!/usr/bin/env python3
"""tune.py - compact Optuna architecture search for Problemulator.

This version is intentionally narrower than the original broad tuner.

Design goals:
- Faster transformer-vs-LSTM comparison.
- Balanced model-family sampling: ``--model-family both`` alternates
  transformer/LSTM trials instead of letting TPE prematurely abandon one branch.
- Do not tune optimizer-scale knobs such as learning rate, weight decay, or
  batch size. Those remain whatever the base config says.
- Tune only the high-impact architecture/dropout/activation knobs.
- Include exact zero in all dropout choices so Optuna can select no dropout.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Prevent MKL/OpenMP library conflicts before importing torch (mirrors main.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import optuna
import torch

from dataset import create_collate_fn
from hardware import setup_device
from main import (
    PROJECT_ROOT,
    _assert_not_inside_src,
    _get_raw_hdf5_paths,
    _resolve_from_project_root,
    run_normalize,
)
from train import ModelTrainer, NonFiniteTensorError
from utils import (
    ensure_dirs,
    get_precision_config,
    load_config,
    save_json,
    seed_everything,
    setup_logging,
    validate_config,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Search-loop constants.
# -----------------------------------------------------------------------------

# Median pruning is simpler than Hyperband for this compact comparison. It avoids
# the earlier failure mode where many transformer trials were killed too early.
PRUNE_WARMUP_EPOCHS = 10

LEADERBOARD_TOP_K = 5
GC_CALLBACK_EVERY_N_TRIALS = 5

MODEL_SIDE_NONFINITE_LABELS = {
    "model predictions",
    "unreduced loss",
    "unreduced fractional loss",
    "masked loss",
}

DATA_FRACTION_DEFAULT = 0.1
EPOCHS_DEFAULT = 25
EARLY_STOPPING_PATIENCE_DEFAULT = 8
WARMUP_EPOCHS_DEFAULT = 2
SAMPLER_SEED_DEFAULT = 42
N_TRIALS_DEFAULT = 64
TIMEOUT_DEFAULT_SECONDS = 248_400  # ~69 h; timeout still acts as the real cap.

# -----------------------------------------------------------------------------
# Grid search space (32 cells per architecture = 64 cells total).
# -----------------------------------------------------------------------------
#
# Each cell is the cartesian product of:
#   - 2 dropout values (0.0, 0.05)
#   - 2 activation functions
#       transformer: ffn_type in {gelu, swiglu}
#       lstm:        head_activation in {gelu, silu}
#   - 8 model-size variants per architecture (see TRANSFORMER_SIZES / LSTM_SIZES)
#
# attention_dropout and film_clamp are intentionally NOT swept (user request).
# Other arch-specific knobs (nhead, use_qk_norm, qkv_bias, bidirectional,
# output_head_divisor, output_head_dropout_factor) are also fixed -- the goal
# of this sweep is to compare model sizes / dropouts / activations directly,
# not to re-tune every dial. Defaults below match the v3 base configs.

DROPOUT_CHOICES: Tuple[float, ...] = (0.0, 0.05)
TRANSFORMER_FFN_TYPE_CHOICES: Tuple[str, ...] = ("gelu", "swiglu")
LSTM_HEAD_ACTIVATION_CHOICES: Tuple[str, ...] = ("gelu", "silu")

# Eight transformer architectures spanning ~1.6M to ~9.6M trainable params
# at ffn_type=gelu. SwiGLU swap inflates each cell by ~1.5x (the FFN cost
# fraction), reaching up to ~12.8M -- accommodated by MAX_TRAINABLE_PARAMS=13M.
# Each is (d_model, num_layers, dim_feedforward); attention_dropout=0,
# output_head_divisor=2, output_head_dropout_factor=0.
#                                                       GELU      SwiGLU
TRANSFORMER_SIZES: Tuple[Dict[str, int], ...] = (
    {"d_model": 256, "num_layers": 3, "dim_feedforward": 512},   #  1.62M  / 2.02M
    {"d_model": 256, "num_layers": 4, "dim_feedforward": 1024},  #  3.20M  / 4.25M
    {"d_model": 256, "num_layers": 5, "dim_feedforward": 1024},  #  3.99M  / 5.31M
    {"d_model": 256, "num_layers": 7, "dim_feedforward": 1024},  #  5.57M  / 7.42M
    {"d_model": 256, "num_layers": 7, "dim_feedforward": 2048},  #  9.25M  / 12.93M
    {"d_model": 512, "num_layers": 3, "dim_feedforward": 1024},  #  6.45M  / 8.03M
    {"d_model": 512, "num_layers": 4, "dim_feedforward": 1024},  #  8.56M  / 10.66M
    {"d_model": 512, "num_layers": 3, "dim_feedforward": 2048},  #  9.60M  / 12.75M (= transformer_main_v3)
)

# Eight bidirectional LSTM architectures spanning ~0.61M to ~9.73M trainable
# params. Each is (d_model, num_layers); output_head_divisor=1,
# output_head_dropout_factor=0.25.
LSTM_SIZES: Tuple[Dict[str, int], ...] = (
    {"d_model": 128, "num_layers": 2},  # ~0.61M
    {"d_model": 192, "num_layers": 2},  # ~1.38M
    {"d_model": 256, "num_layers": 2},  # ~2.44M
    {"d_model": 256, "num_layers": 4},  # ~4.81M
    {"d_model": 320, "num_layers": 3},  # ~5.66M
    {"d_model": 448, "num_layers": 2},  # ~7.45M
    {"d_model": 384, "num_layers": 3},  # ~8.14M
    {"d_model": 512, "num_layers": 2},  # ~9.73M (matches lstm_main_v3)
)

# Fixed knobs (NOT swept in this round). Values match the v3 base configs.
FIXED_FILM_CLAMP: float = 1000.0
FIXED_ATTENTION_DROPOUT: float = 0.0
TRANSFORMER_FIXED_NHEAD: int = 4
TRANSFORMER_FIXED_USE_QK_NORM: bool = True
TRANSFORMER_FIXED_QKV_BIAS: bool = True
TRANSFORMER_FIXED_OUTPUT_HEAD_DIVISOR: int = 2
TRANSFORMER_FIXED_OUTPUT_HEAD_DROPOUT_FACTOR: float = 0.0
TRANSFORMER_FIXED_HEAD_ACTIVATION: str = "gelu"
LSTM_FIXED_BIDIRECTIONAL: bool = True
LSTM_FIXED_OUTPUT_HEAD_DIVISOR: int = 1
LSTM_FIXED_OUTPUT_HEAD_DROPOUT_FACTOR: float = 0.25
LSTM_FIXED_FFN_TYPE: str = "gelu"  # placeholder; LSTM ignores ffn_type

# Hard cap on trainable parameters. Headroom above the user-stated ~10M
# target is left for transformer trials that pick ffn_type='swiglu', which has
# ~1.5x more FFN params than GELU at the same dim_feedforward. The 8 sizes
# below all fit under this cap with EITHER activation. The two main configs
# (transformer_main_v3: 9.60M GELU; lstm_main_v3: 9.73M) sit at the upper end.
MAX_TRAINABLE_PARAMS: int = 13_000_000

# TPE knobs retained for legacy non-grid model_family modes (transformer, lstm,
# both, sequential, config). The 'grid' mode uses GridSampler instead.
TPE_N_STARTUP_TRIALS_LEGACY: int = 12
TPE_N_EI_CANDIDATES_LEGACY: int = 32


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge: dicts merge, scalars/lists in ``overrides`` replace base."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class TunerCallbackTrainer(ModelTrainer):
    """ModelTrainer subclass that reports per-epoch val loss to Optuna."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
        processed_dir: Path,
        splits: Dict[str, List[Tuple[str, int]]],
        collate_fn: Callable,
        trial: optuna.Trial,
    ) -> None:
        super().__init__(
            config=config,
            device=device,
            save_dir=save_dir,
            processed_dir=processed_dir,
            splits=splits,
            collate_fn=collate_fn,
        )
        self._trial = trial

    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        elapsed_time: float,
        improvement,
        lr: float,
    ) -> None:
        super()._log_epoch_results(epoch, train_loss, val_loss, elapsed_time, improvement, lr)

        if not math.isfinite(val_loss):
            raise optuna.TrialPruned(f"Non-finite val_loss at epoch {epoch}.")

        self._trial.report(val_loss, step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}, val={val_loss:.3e}.")


def _choose_model_type(
    trial: optuna.Trial,
    requested_family: str,
    n_trials_lstm: Optional[int] = None,
) -> str:
    """Select model family.

    ``grid`` samples model_type as part of the grid (used with GridSampler).
    ``both`` alternates by trial number (even=transformer, odd=lstm).
    ``sequential`` runs LSTM trials first (trial.number < n_trials_lstm),
    then transformer trials.
    ``transformer`` or ``lstm`` runs a single architecture.
    """
    family = str(requested_family).lower()
    if family == "config":
        raise ValueError("'config' model family must be resolved before sampling.")
    if family == "grid":
        return str(trial.suggest_categorical("model_type", ["lstm", "transformer"]))
    if family == "both":
        return "transformer" if trial.number % 2 == 0 else "lstm"
    if family == "sequential":
        if n_trials_lstm is None:
            raise ValueError(
                "model_family='sequential' requires n_trials_lstm to be provided."
            )
        return "lstm" if trial.number < int(n_trials_lstm) else "transformer"
    if family in {"transformer", "lstm"}:
        return family
    raise ValueError(f"Unknown model family '{requested_family}'.")


def _build_transformer_block(size_idx: int, dropout: float, ffn_type: str) -> Dict[str, Any]:
    """Construct a transformer model_hyperparameters block from grid coordinates."""
    size = TRANSFORMER_SIZES[size_idx]
    return {
        "model_type": "transformer",
        "d_model": int(size["d_model"]),
        "dropout": float(dropout),
        "film_clamp": float(FIXED_FILM_CLAMP),
        "output_head_divisor": int(TRANSFORMER_FIXED_OUTPUT_HEAD_DIVISOR),
        "output_head_dropout_factor": float(TRANSFORMER_FIXED_OUTPUT_HEAD_DROPOUT_FACTOR),
        "head_activation": TRANSFORMER_FIXED_HEAD_ACTIVATION,
        "transformer": {
            "nhead": TRANSFORMER_FIXED_NHEAD,
            "num_layers": int(size["num_layers"]),
            "dim_feedforward": int(size["dim_feedforward"]),
            "attention_dropout": float(FIXED_ATTENTION_DROPOUT),
            "use_qk_norm": TRANSFORMER_FIXED_USE_QK_NORM,
            "qkv_bias": TRANSFORMER_FIXED_QKV_BIAS,
            "ffn_type": ffn_type,
        },
    }


def _build_lstm_block(size_idx: int, dropout: float, head_activation: str) -> Dict[str, Any]:
    """Construct an LSTM model_hyperparameters block from grid coordinates."""
    size = LSTM_SIZES[size_idx]
    return {
        "model_type": "lstm",
        "d_model": int(size["d_model"]),
        "dropout": float(dropout),
        "film_clamp": float(FIXED_FILM_CLAMP),
        "output_head_divisor": int(LSTM_FIXED_OUTPUT_HEAD_DIVISOR),
        "output_head_dropout_factor": float(LSTM_FIXED_OUTPUT_HEAD_DROPOUT_FACTOR),
        "head_activation": head_activation,
        "transformer": {
            "nhead": TRANSFORMER_FIXED_NHEAD,
            "num_layers": int(TRANSFORMER_SIZES[0]["num_layers"]),
            "dim_feedforward": int(TRANSFORMER_SIZES[0]["dim_feedforward"]),
            "attention_dropout": float(FIXED_ATTENTION_DROPOUT),
            "use_qk_norm": TRANSFORMER_FIXED_USE_QK_NORM,
            "qkv_bias": TRANSFORMER_FIXED_QKV_BIAS,
            "ffn_type": LSTM_FIXED_FFN_TYPE,
        },
        "lstm": {
            "num_layers": int(size["num_layers"]),
            "bidirectional": LSTM_FIXED_BIDIRECTIONAL,
        },
    }


def _suggest_search_space(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Sample a per-trial model_hyperparameters block from the grid.

    Search space per trial: dropout, activation, model size. ``model_type``
    is sampled inside ``_choose_model_type`` when model_family='grid'.
    Everything else (optimizer, scheduler, attention_dropout, film_clamp,
    output_head_*) stays fixed.
    """
    del base_config  # Kept in signature for backward compat.

    n_trials_lstm = getattr(args, "n_trials_lstm", None)
    model_type = _choose_model_type(trial, args.model_family, n_trials_lstm)
    trial.set_user_attr("model_type", model_type)

    dropout = float(trial.suggest_categorical("dropout", list(DROPOUT_CHOICES)))

    if args.model_family == "grid":
        # Grid mode: one shared activation_idx, mapped per-arch in build helpers.
        activation_idx = int(trial.suggest_categorical("activation_idx", [0, 1]))
    else:
        activation_idx = None

    if model_type == "transformer":
        size_idx = int(trial.suggest_categorical("size_idx", list(range(len(TRANSFORMER_SIZES)))))
        if activation_idx is not None:
            ffn_type = TRANSFORMER_FFN_TYPE_CHOICES[activation_idx]
        else:
            ffn_type = str(trial.suggest_categorical("ffn_type", list(TRANSFORMER_FFN_TYPE_CHOICES)))
        model_block = _build_transformer_block(size_idx, dropout, ffn_type)
    else:  # lstm
        size_idx = int(trial.suggest_categorical("size_idx", list(range(len(LSTM_SIZES)))))
        if activation_idx is not None:
            head_activation = LSTM_HEAD_ACTIVATION_CHOICES[activation_idx]
        else:
            head_activation = str(
                trial.suggest_categorical("head_activation", list(LSTM_HEAD_ACTIVATION_CHOICES))
            )
        model_block = _build_lstm_block(size_idx, dropout, head_activation)

    return {"model_hyperparameters": model_block}


def _forced_overrides(args: argparse.Namespace, trial_number: int, study_name: str) -> Dict[str, Any]:
    """Tuner-mandated config keys that are not sampled.

    Important: optimizer-scale parameters are intentionally not touched here.
    The base config remains responsible for learning_rate, weight_decay,
    batch_size, Adam betas, EMA decay, and scheduler choice.
    """
    return {
        "training_hyperparameters": {
            "epochs": int(args.epochs),
            "dataset_fraction_to_use": float(args.data_fraction),
            "early_stopping_patience": int(args.patience),
            "warmup_epochs": int(args.warmup_epochs),
        },
        "miscellaneous_settings": {
            "torch_compile": False,
            "num_workers": int(args.num_workers),
            "rebuild_processed_data": False,
            "execution_mode": "train",
            # Fixed seed across trials => identical sampled data subset for every trial.
            "random_seed": int(args.sampler_seed),
        },
        "output_paths_config": {
            "fixed_model_foldername": f"tune_{study_name}/trial_{trial_number:04d}",
        },
    }


def _trial_dir(models_root: Path, study_name: str, trial_number: int) -> Path:
    return models_root / f"tune_{study_name}" / f"trial_{trial_number:04d}"


def _study_root(models_root: Path, study_name: str) -> Path:
    return models_root / f"tune_{study_name}"


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or ("cuda" in msg and "memory" in msg)
    return False


def _build_objective(
    base_config: Dict[str, Any],
    device: torch.device,
    processed_dir: Path,
    splits: Dict[str, List[Tuple[str, int]]],
    collate_fn: Callable,
    models_root: Path,
    study_name: str,
    args: argparse.Namespace,
) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective bound to invariants resolved once at startup."""

    def objective(trial: optuna.Trial) -> float:
        suggested = _suggest_search_space(trial, base_config, args)
        forced = _forced_overrides(args, trial.number, study_name)

        merged = _deep_merge(base_config, suggested)
        merged = _deep_merge(merged, forced)
        validate_config(merged)

        save_dir = _trial_dir(models_root, study_name, trial.number)
        if not ensure_dirs(save_dir):
            raise RuntimeError(f"Failed to create trial dir: {save_dir}")

        if not save_json(merged, save_dir / "trial_config.json"):
            raise RuntimeError(f"Failed to save trial_config.json in {save_dir}")

        trial.set_user_attr("save_dir", str(save_dir.resolve()))
        trial.set_user_attr("started_at", time.time())

        trainer = None
        try:
            trainer = TunerCallbackTrainer(
                config=merged,
                device=device,
                save_dir=save_dir,
                processed_dir=processed_dir,
                splits=splits,
                collate_fn=collate_fn,
                trial=trial,
            )

            if trainer.model is not None:
                n_params = sum(p.numel() for p in trainer.model.parameters())
                n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
                trial.set_user_attr("num_parameters", int(n_params))
                trial.set_user_attr("num_trainable", int(n_trainable))
                if n_trainable > MAX_TRAINABLE_PARAMS:
                    raise optuna.TrialPruned(
                        f"Trial would train {n_trainable:,} parameters, exceeding the "
                        f"{MAX_TRAINABLE_PARAMS:,} cap. Sampled architecture pruned to "
                        "keep the sweep within the ~10M budget."
                    )

            best_val = trainer.train()
            return float(best_val)

        except optuna.TrialPruned:
            raise
        except NonFiniteTensorError as exc:
            if exc.label not in MODEL_SIDE_NONFINITE_LABELS:
                raise
            logger.warning(
                "Trial %d pruned due to model-side numerical instability: %s",
                trial.number,
                exc,
            )
            raise optuna.TrialPruned(f"Model-side non-finite values: {exc}") from exc
        except BaseException as exc:
            if _is_oom_error(exc):
                logger.warning(
                    "Trial %d marked pruned due to GPU memory pressure: %s",
                    trial.number,
                    exc,
                )
                raise optuna.TrialPruned(f"OOM: {exc}") from exc
            raise
        finally:
            if trainer is not None:
                del trainer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def _garbage_collect_callback_factory(
    models_root: Path,
    study_name: str,
    top_k: int,
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """Periodically delete checkpoints for completed trials outside top-K."""

    def _callback(study: optuna.Study, _frozen: optuna.trial.FrozenTrial) -> None:
        completed = [
            t
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if t.value is not None and math.isfinite(t.value)
        ]
        if len(completed) % GC_CALLBACK_EVERY_N_TRIALS != 0 or not completed:
            return

        completed.sort(key=lambda t: t.value)
        keep_numbers = {t.number for t in completed[:top_k]}
        study_dir = _study_root(models_root, study_name)
        if not study_dir.is_dir():
            return

        deleted = 0
        for trial_dir in study_dir.glob("trial_*"):
            try:
                trial_idx = int(trial_dir.name.split("_")[-1])
            except ValueError:
                continue
            if trial_idx in keep_numbers:
                continue

            ckpt = trial_dir / "best_model.pt"
            if ckpt.is_file():
                try:
                    ckpt.unlink()
                    deleted += 1
                except OSError as exc:
                    logger.warning("Failed to delete %s: %s", ckpt, exc)

        if deleted:
            logger.info(
                "Checkpoint GC: deleted %d non-top-%d checkpoints (kept trials %s).",
                deleted,
                top_k,
                sorted(keep_numbers),
            )

    return _callback


def _emit_leaderboard(study: optuna.Study, output_dir: Path) -> None:
    """Write leaderboard.json and all_trials.json."""

    trials = study.get_trials(
        deepcopy=False,
        states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
        ),
    )

    def _serialize(t: optuna.trial.FrozenTrial) -> Dict[str, Any]:
        duration_s = None
        if t.datetime_start is not None and t.datetime_complete is not None:
            duration_s = (t.datetime_complete - t.datetime_start).total_seconds()

        value = None
        if t.value is not None and math.isfinite(t.value):
            value = float(t.value)

        return {
            "number": t.number,
            "state": t.state.name,
            "value": value,
            "params": dict(t.params),
            "user_attrs": dict(t.user_attrs),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_s": duration_s,
        }

    serialized = [_serialize(t) for t in trials]
    completed = [s for s in serialized if s["state"] == "COMPLETE" and s["value"] is not None]
    completed.sort(key=lambda s: s["value"])

    best = completed[0] if completed else None
    leaderboard = {
        "study_name": study.study_name,
        "direction": "minimize",
        "n_trials_total": len(trials),
        "n_trials_complete": len(completed),
        "n_trials_pruned": sum(1 for s in serialized if s["state"] == "PRUNED"),
        "n_trials_failed": sum(1 for s in serialized if s["state"] == "FAIL"),
        "best_trial_number": best["number"] if best else None,
        "best_value": best["value"] if best else None,
        "best_params": best["params"] if best else None,
        "best_user_attrs": best["user_attrs"] if best else None,
        "top_k": completed[:LEADERBOARD_TOP_K],
        "note": (
            "Compact architecture/dropout/activation tuner. Optimizer-scale "
            "settings such as learning_rate, weight_decay, and batch_size are "
            "fixed from the base config."
        ),
    }

    if not save_json(leaderboard, output_dir / "leaderboard.json"):
        raise RuntimeError("Failed to write leaderboard.json")
    if not save_json({"trials": serialized}, output_dir / "all_trials.json"):
        raise RuntimeError("Failed to write all_trials.json")

    logger.info("Wrote leaderboard with %d completed trials.", len(completed))


def _build_study(args: argparse.Namespace) -> optuna.Study:
    if args.model_family == "grid":
        # Exhaustive grid: 2 dropouts x 8 sizes x 2 activations x 2 archs = 64 cells.
        # The 'activation_idx' param is a single 0/1 selector mapped per-arch:
        #   transformer: 0 -> ffn_type="gelu",  1 -> ffn_type="swiglu"
        #   lstm:        0 -> head_activation="gelu", 1 -> head_activation="silu"
        # Using ONE shared param prevents GridSampler from blowing up to 128 cells.
        if len(TRANSFORMER_SIZES) != len(LSTM_SIZES):
            raise ValueError(
                f"GridSampler requires TRANSFORMER_SIZES ({len(TRANSFORMER_SIZES)}) and "
                f"LSTM_SIZES ({len(LSTM_SIZES)}) to share a length; otherwise the "
                "size_idx categorical would not exhaustively cover both archs."
            )
        sampler: optuna.samplers.BaseSampler = optuna.samplers.GridSampler(
            search_space={
                "model_type": ["lstm", "transformer"],
                "dropout": list(DROPOUT_CHOICES),
                "size_idx": list(range(len(TRANSFORMER_SIZES))),
                "activation_idx": [0, 1],
            },
            seed=int(args.sampler_seed),
        )
        logger.info(
            "model_family='grid': using GridSampler over %d cells "
            "(2 archs x 2 dropouts x 8 sizes x 2 activations).",
            2 * len(DROPOUT_CHOICES) * len(TRANSFORMER_SIZES) * 2,
        )
    else:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=TPE_N_STARTUP_TRIALS_LEGACY,
            n_ei_candidates=TPE_N_EI_CANDIDATES_LEGACY,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=int(args.sampler_seed),
        )

    # When comparing both architectures, MedianPruner is unfair: transformer
    # trials converge faster and dominate the shared median, causing LSTM trials
    # to be pruned before they reach their asymptotic loss. Use NopPruner so
    # every trial runs to early-stopping completion on equal footing. The same
    # reasoning applies to the strictly sequential and grid layouts.
    if args.model_family in {"both", "sequential", "grid"}:
        pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()
        logger.info(
            "model_family=%r: using NopPruner so both architectures run "
            "to early-stopping completion without cross-architecture bias.",
            args.model_family,
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=TPE_N_STARTUP_TRIALS_LEGACY,
            n_warmup_steps=int(args.prune_warmup_epochs),
            interval_steps=1,
        )

    return optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact Optuna architecture search for the Problemulator emulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "transformer_main_v3.jsonc",
        help="Base config file. Optimizer-scale settings are kept from this file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data dir containing raw/ and processed/.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory under which tune_<study>/ is written.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Optuna study name. Re-using an existing name resumes the study.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL. Defaults to sqlite:///<models>/tune_<study>/study.db.",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=("grid", "config", "both", "sequential", "transformer", "lstm"),
        default="grid",
        help=(
            "'grid' (default) runs a 64-cell exhaustive grid via GridSampler: "
            "2 archs x 2 dropouts x 8 sizes x 2 activations. "
            "'config' uses model_hyperparameters.model_type from the base config. "
            "'both' alternates transformer/LSTM trials by trial number. "
            "'sequential' runs the first --n-trials-lstm trials as LSTM, the "
            "remainder as transformer."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument(
        "--n-trials-lstm",
        type=int,
        default=None,
        help=(
            "LSTM/transformer split index for --model-family=sequential. "
            "Defaults to --n-trials // 2 (e.g. 64 -> 32 LSTM, then 32 transformer)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_DEFAULT_SECONDS,
        help="Walltime budget in seconds.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--data-fraction", type=float, default=DATA_FRACTION_DEFAULT)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE_DEFAULT)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    parser.add_argument("--prune-warmup-epochs", type=int, default=PRUNE_WARMUP_EPOCHS)
    parser.add_argument("--sampler-seed", type=int, default=SAMPLER_SEED_DEFAULT)
    parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    args.base_config = _resolve_from_project_root(args.base_config)
    args.data_dir = _resolve_from_project_root(args.data_dir)
    args.models_dir = _resolve_from_project_root(args.models_dir)

    setup_logging()

    try:
        _assert_not_inside_src(args.data_dir, "--data-dir")
        _assert_not_inside_src(args.models_dir, "--models-dir")

        study_dir = _study_root(args.models_dir, args.study_name)
        if not ensure_dirs(args.data_dir, args.models_dir, study_dir):
            raise RuntimeError("Failed to create required directories.")

        if args.storage is None:
            args.storage = f"sqlite:///{(study_dir / 'study.db').resolve()}"

        setup_logging(log_file=study_dir / "tune_run.log", force=True)
        logger.info("Tune CLI args: %s", json.dumps(vars(args), default=str))

        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        if args.warmup_epochs >= args.epochs:
            raise ValueError("--warmup-epochs must be less than --epochs.")
        if args.prune_warmup_epochs >= args.epochs:
            raise ValueError("--prune-warmup-epochs must be less than --epochs.")
        if not (0.0 < args.data_fraction <= 1.0):
            raise ValueError("--data-fraction must be in (0, 1].")
        if args.n_trials <= 0:
            raise ValueError("--n-trials must be positive.")

        base_config = load_config(args.base_config)
        if args.model_family == "config":
            args.model_family = str(base_config["model_hyperparameters"]["model_type"]).lower()
            logger.info("Resolved --model-family config to '%s'.", args.model_family)

        if args.model_family == "grid":
            grid_size = 2 * len(DROPOUT_CHOICES) * len(TRANSFORMER_SIZES) * 2
            if args.n_trials != grid_size:
                logger.info(
                    "model_family='grid': overriding --n-trials %d -> %d to cover the full grid.",
                    int(args.n_trials),
                    grid_size,
                )
                args.n_trials = grid_size
            if args.n_trials_lstm is not None:
                raise ValueError(
                    "--n-trials-lstm is not used with --model-family=grid; the 64-cell "
                    "grid balances 32 LSTM + 32 transformer trials automatically."
                )

        if args.model_family == "sequential":
            if args.n_trials_lstm is None:
                args.n_trials_lstm = int(args.n_trials) // 2
            if not (0 <= int(args.n_trials_lstm) <= int(args.n_trials)):
                raise ValueError(
                    "--n-trials-lstm must be in [0, --n-trials]; got "
                    f"{args.n_trials_lstm} with --n-trials={args.n_trials}."
                )
            logger.info(
                "model_family='sequential': trials 0..%d -> lstm, %d..%d -> transformer.",
                int(args.n_trials_lstm) - 1,
                int(args.n_trials_lstm),
                int(args.n_trials) - 1,
            )
        elif args.n_trials_lstm is not None:
            raise ValueError(
                "--n-trials-lstm is only valid with --model-family=sequential."
            )

        seed_everything(int(args.sampler_seed))

        backend = str(base_config["miscellaneous_settings"]["device_backend"])
        device = setup_device(backend)

        matmul_precision = str(base_config["precision"]["float32_matmul_precision"]).lower()
        if matmul_precision != "none" and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)
            logger.info("Set float32 matmul precision to '%s'.", matmul_precision)

        raw_dir = args.data_dir / "raw"
        processed_dir = args.data_dir / "processed"
        raw_hdf5_paths = _get_raw_hdf5_paths(base_config, raw_dir)

        # Build/validate processed artifacts once, shared across all trials.
        normalize_config = copy.deepcopy(base_config)
        normalize_config["miscellaneous_settings"]["execution_mode"] = "normalize"
        normalize_config["miscellaneous_settings"]["rebuild_processed_data"] = False
        splits = run_normalize(normalize_config, raw_hdf5_paths, processed_dir)

        padding_val = float(base_config["data_specification"]["padding_value"])
        padding_eps = float(base_config["normalization"]["padding_comparison_epsilon"])
        input_dtype = get_precision_config(base_config)["input_dtype"]
        collate_fn = create_collate_fn(padding_val, padding_eps, tensor_dtype=input_dtype)

        study = _build_study(args)
        existing_complete = sum(
            1
            for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        logger.info(
            "Optuna study '%s' loaded (existing completed trials=%d). Storage: %s",
            args.study_name,
            existing_complete,
            args.storage,
        )

        objective = _build_objective(
            base_config=base_config,
            device=device,
            processed_dir=processed_dir,
            splits=splits,
            collate_fn=collate_fn,
            models_root=args.models_dir,
            study_name=args.study_name,
            args=args,
        )
        gc_callback = _garbage_collect_callback_factory(
            args.models_dir,
            args.study_name,
            LEADERBOARD_TOP_K,
        )

        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            timeout=int(args.timeout),
            gc_after_trial=True,
            callbacks=[gc_callback],
        )

        _emit_leaderboard(study, study_dir)

        # Optuna's study.best_trial raises ValueError("Record does not exist")
        # when zero trials completed (e.g., every trial pruned). Guard against
        # that so the tuner exits cleanly with the leaderboard already written.
        try:
            best = study.best_trial
        except ValueError:
            best = None

        if best is not None:
            logger.info(
                "Best trial: number=%d value=%.6e params=%s user_attrs=%s",
                best.number,
                float(best.value),
                json.dumps(best.params, sort_keys=True),
                json.dumps(best.user_attrs, sort_keys=True, default=str),
            )
        else:
            logger.warning(
                "No completed trials in study '%s'; no best trial to report.",
                study.study_name,
            )

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception:
        logger.exception("Tuning failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
