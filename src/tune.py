#!/usr/bin/env python3
"""tune.py - Optuna hyperparameter search comparing transformer vs LSTM.

Drives ``ModelTrainer`` in-process: each trial samples a config delta on top
of a base config, runs a short training loop on a small data fraction, and
reports per-epoch validation loss to Optuna for Hyperband pruning. The study
is SQLite-backed and resumable across SLURM restarts.
"""
from __future__ import annotations

import os

# Prevent MKL/OpenMP library conflicts before importing torch (mirrors main.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
from model import create_prediction_model
from train import ModelTrainer
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

# --- Tuner mechanics. None of these affect physics; they tune the search loop. ---
HYPERBAND_REDUCTION_FACTOR = 3          # Optuna's recommended Hyperband bracket factor.
PRUNE_MIN_RESOURCE_EPOCHS = 4           # Earliest epoch eligible for pruning.
TPE_N_STARTUP_TRIALS = 16               # Random exploration before TPE conditioning kicks in.
TPE_N_EI_CANDIDATES = 24                # TPE candidate set size per ask.
LEADERBOARD_TOP_K = 25                  # Trial checkpoints retained on disk.
GC_CALLBACK_EVERY_N_TRIALS = 25         # Periodicity of checkpoint cleanup.
DATA_FRACTION_DEFAULT = 0.1
EPOCHS_DEFAULT = 35
EARLY_STOPPING_PATIENCE_DEFAULT = 8
WARMUP_EPOCHS_DEFAULT = 3               # Must stay strictly < epochs (validate_config requires it).
SAMPLER_SEED_DEFAULT = 42
N_TRIALS_DEFAULT = 2000                 # Effective cap; timeout is the real budget.
TIMEOUT_DEFAULT_SECONDS = 248_400       # ~69h; leaves ~3h margin under a 72h walltime.

# Discrete d_model set chosen so divisors of {1,2,4,8,16} cover most heads cleanly.
DISCRETE_D_MODEL: Tuple[int, ...] = (64, 96, 128, 160, 192, 224, 256, 384, 512)
NHEAD_CHOICES: Tuple[int, ...] = (1, 2, 4, 8, 16)
HEAD_DIVISOR_CHOICES: Tuple[int, ...] = (1, 2, 4)
DIM_FF_CHOICES: Tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048)
BATCH_SIZE_CHOICES: Tuple[int, ...] = (64, 128, 256, 512)
ADAM_BETA2_CHOICES: Tuple[float, ...] = (0.95, 0.99, 0.999)
EMA_DECAY_CHOICES: Tuple[float, ...] = (0.0, 0.99, 0.999)

# Param-matched comparisons: when a trial picks `param_match_mode="matched"`,
# the architecture-sizing knob (dim_feedforward for transformer, d_model for
# LSTM) is derived to land near one of these budgets so transformer/LSTM trials
# can be paired apples-to-apples post hoc. Targets span ~2 orders of magnitude
# around the typical baseline runs.
PARAM_MATCH_TARGETS: Tuple[int, ...] = (250_000, 1_000_000, 4_000_000, 15_000_000)
# Search bounds for the matched-mode solvers. Both knobs are free positive ints
# per validate_config; these bounds just keep the bisection loop bounded.
TRANSFORMER_DIM_FF_SOLVE_BOUNDS: Tuple[int, int] = (16, 32_768)
LSTM_D_MODEL_SOLVE_BOUNDS: Tuple[int, int] = (16, 2048)
LSTM_D_MODEL_SOLVE_STEP: int = 8  # Snap solved d_model to a multiple of 8.


def _divisors(value: int, candidates: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the subset of ``candidates`` that evenly divide ``value``."""
    return tuple(c for c in candidates if value % c == 0)


def _at_most(value: int, candidates: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return ``candidates`` filtered to entries ``<= value``."""
    return tuple(c for c in candidates if c <= value)


class TunerCallbackTrainer(ModelTrainer):
    """ModelTrainer subclass that reports per-epoch val loss to Optuna and prunes."""

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


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge: dicts merge, scalars/lists in ``overrides`` replace base."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _count_model_params(base_config: Dict[str, Any], model_block: Dict[str, Any]) -> int:
    """Build the model on CPU (no compile) and return its trainable param count.

    Used by the matched-budget solvers; relies on ``create_prediction_model``
    to keep the count exactly aligned with what training will instantiate.
    """
    cfg = _deep_merge(base_config, {"model_hyperparameters": model_block})
    cfg["miscellaneous_settings"]["torch_compile"] = False
    model = create_prediction_model(cfg, device=torch.device("cpu"), compile_model=False)
    n = sum(p.numel() for p in model.parameters())
    del model
    return int(n)


def _solve_transformer_dim_ff(
    base_config: Dict[str, Any],
    partial_block: Dict[str, Any],
    target_params: int,
) -> int:
    """Bisect ``transformer.dim_feedforward`` so the model lands near ``target_params``.

    Param count is monotone-increasing in ``dim_feedforward`` (FFN widths are
    purely additive). Returns the int from the search interval whose realised
    param count is closest to the target.
    """
    lo, hi = TRANSFORMER_DIM_FF_SOLVE_BOUNDS

    def count_at(dim_ff: int) -> int:
        block = copy.deepcopy(partial_block)
        block["transformer"]["dim_feedforward"] = int(dim_ff)
        return _count_model_params(base_config, block)

    n_lo = count_at(lo)
    n_hi = count_at(hi)
    if target_params <= n_lo:
        return lo
    if target_params >= n_hi:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        n_mid = count_at(mid)
        if n_mid < target_params:
            lo, n_lo = mid, n_mid
        else:
            hi, n_hi = mid, n_mid
    return lo if abs(n_lo - target_params) <= abs(n_hi - target_params) else hi


def _solve_lstm_d_model(
    base_config: Dict[str, Any],
    partial_block: Dict[str, Any],
    target_params: int,
) -> int:
    """Bisect LSTM ``d_model`` (snapped to a multiple of LSTM_D_MODEL_SOLVE_STEP).

    Param count is monotone-increasing in ``d_model`` for fixed num_layers /
    bidirectional / head_divisor. Returns the snapped int whose realised count
    is closest to the target.
    """
    step = LSTM_D_MODEL_SOLVE_STEP
    lo_raw, hi_raw = LSTM_D_MODEL_SOLVE_BOUNDS
    lo = max(step, ((lo_raw + step - 1) // step) * step)
    hi = (hi_raw // step) * step

    def count_at(d: int) -> int:
        block = copy.deepcopy(partial_block)
        block["d_model"] = int(d)
        return _count_model_params(base_config, block)

    n_lo = count_at(lo)
    n_hi = count_at(hi)
    if target_params <= n_lo:
        return lo
    if target_params >= n_hi:
        return hi
    while hi - lo > step:
        mid = (((lo + hi) // 2) // step) * step
        if mid <= lo:
            mid = lo + step
        if mid >= hi:
            mid = hi - step
        n_mid = count_at(mid)
        if n_mid < target_params:
            lo, n_lo = mid, n_mid
        else:
            hi, n_hi = mid, n_mid
    return lo if abs(n_lo - target_params) <= abs(n_hi - target_params) else hi


def _suggest_search_space(
    trial: optuna.Trial, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Sample a per-trial config delta. Branches conditionally on model_type.

    When ``param_match_mode="matched"`` is sampled, the architecture-sizing
    knob is derived from a discrete target budget (PARAM_MATCH_TARGETS) so
    transformer and LSTM trials can be paired apples-to-apples on parameter
    count. ``free`` mode preserves the original sampling space unchanged.
    """
    model_type = trial.suggest_categorical("model_type", ["transformer", "lstm"])
    param_match_mode = trial.suggest_categorical(
        "param_match_mode", ["free", "matched"]
    )

    target_params: int = 0
    if param_match_mode == "matched":
        target_params = int(
            trial.suggest_categorical(
                "matched_target_params", list(PARAM_MATCH_TARGETS)
            )
        )

    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    head_dropout_factor = trial.suggest_float("output_head_dropout_factor", 0.0, 0.7)
    film_clamp = trial.suggest_float("film_clamp", 1e2, 1e4, log=True)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", list(BATCH_SIZE_CHOICES))
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.5, 5.0)
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "plateau"])
    warmup_start_factor = trial.suggest_float("warmup_start_factor", 0.01, 0.5)
    warmup_unit = trial.suggest_categorical("warmup_unit", ["epoch", "step"])
    adam_beta2 = trial.suggest_categorical("adam_beta2", list(ADAM_BETA2_CHOICES))
    ema_decay = trial.suggest_categorical("ema_decay", list(EMA_DECAY_CHOICES))

    if model_type == "transformer":
        d_model = trial.suggest_categorical("d_model", list(DISCRETE_D_MODEL))
        # In matched mode, fix the head_divisor to 1 so the only varying
        # transformer-sizing knob is dim_feedforward (cleaner pairing signal).
        if param_match_mode == "matched":
            head_divisor = 1
        else:
            head_divisor = trial.suggest_categorical(
                "output_head_divisor", list(_at_most(d_model, HEAD_DIVISOR_CHOICES))
            )

        valid_heads = _divisors(d_model, NHEAD_CHOICES)
        if not valid_heads:
            raise optuna.TrialPruned(f"No valid nhead for d_model={d_model}.")
        nhead = trial.suggest_categorical("nhead", list(valid_heads))
        num_layers = trial.suggest_int("num_layers_transformer", 2, 8)
        attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.3)
        use_qk_norm = trial.suggest_categorical("use_qk_norm", [True, False])
        qkv_bias = trial.suggest_categorical("qkv_bias", [True, False])
        ffn_type = trial.suggest_categorical("ffn_type", ["gelu", "swiglu"])

        partial_transformer_block: Dict[str, Any] = {
            "model_type": "transformer",
            "d_model": int(d_model),
            "dropout": float(dropout),
            "output_head_divisor": int(head_divisor),
            "output_head_dropout_factor": float(head_dropout_factor),
            "film_clamp": float(film_clamp),
            "transformer": {
                "nhead": int(nhead),
                "num_layers": int(num_layers),
                "dim_feedforward": 256,  # placeholder, set below
                "attention_dropout": float(attention_dropout),
                "use_qk_norm": bool(use_qk_norm),
                "qkv_bias": bool(qkv_bias),
                "ffn_type": str(ffn_type),
            },
        }

        if param_match_mode == "free":
            dim_feedforward = trial.suggest_categorical(
                "dim_feedforward", list(DIM_FF_CHOICES)
            )
        else:
            dim_feedforward = _solve_transformer_dim_ff(
                base_config, partial_transformer_block, target_params
            )

        partial_transformer_block["transformer"]["dim_feedforward"] = int(dim_feedforward)
        model_block = partial_transformer_block
        use_amp = True
        amp_autocast_dtype = "bfloat16"

    else:  # LSTM
        num_layers = trial.suggest_int("num_layers_lstm", 1, 5)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])

        if param_match_mode == "free":
            d_model = trial.suggest_categorical("d_model", list(DISCRETE_D_MODEL))
            head_divisor = trial.suggest_categorical(
                "output_head_divisor", list(_at_most(d_model, HEAD_DIVISOR_CHOICES))
            )
        else:
            # Same head_divisor=1 convention as the matched transformer path
            # so output heads are identical across paired trials.
            head_divisor = 1

        partial_lstm_block: Dict[str, Any] = {
            "model_type": "lstm",
            "d_model": int(d_model) if param_match_mode == "free" else 64,
            "dropout": float(dropout),
            "output_head_divisor": int(head_divisor),
            "output_head_dropout_factor": float(head_dropout_factor),
            "film_clamp": float(film_clamp),
            "lstm": {
                "num_layers": int(num_layers),
                "bidirectional": bool(bidirectional),
            },
        }

        if param_match_mode == "matched":
            d_model = _solve_lstm_d_model(
                base_config, partial_lstm_block, target_params
            )
            partial_lstm_block["d_model"] = int(d_model)

        model_block = partial_lstm_block
        # use_amp=False requires amp_autocast_dtype="none" (validate_config invariant).
        use_amp = False
        amp_autocast_dtype = "none"

    if param_match_mode == "matched":
        actual_n = _count_model_params(base_config, model_block)
        trial.set_user_attr("matched_target_params", int(target_params))
        trial.set_user_attr("actual_param_count", int(actual_n))
    trial.set_user_attr("param_match_mode", str(param_match_mode))

    return {
        "model_hyperparameters": model_block,
        "training_hyperparameters": {
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "gradient_clip_val": float(gradient_clip_val),
            "scheduler_type": str(scheduler_type),
            "warmup_start_factor": float(warmup_start_factor),
            "warmup_unit": str(warmup_unit),
            "adam_beta1": 0.9,
            "adam_beta2": float(adam_beta2),
            "ema_decay": float(ema_decay),
            "use_amp": bool(use_amp),
        },
        "precision": {
            "amp_autocast_dtype": amp_autocast_dtype,
        },
    }


def _forced_overrides(args: argparse.Namespace, trial_number: int, study_name: str) -> Dict[str, Any]:
    """Tuner-mandated config keys that are NOT sampled."""
    return {
        "training_hyperparameters": {
            "epochs": int(args.epochs),
            "dataset_fraction_to_use": float(args.data_fraction),
            "early_stopping_patience": int(args.patience),
            "warmup_epochs": int(WARMUP_EPOCHS_DEFAULT),
        },
        "miscellaneous_settings": {
            "torch_compile": False,
            "num_workers": 1,
            "rebuild_processed_data": False,
            "execution_mode": "train",
            # Fixed seed across trials => identical 10% slice for every trial
            # (apples-to-apples architecture comparison).
            "random_seed": int(args.sampler_seed),
        },
        "output_paths_config": {
            # Per-trial unique foldername satisfies validate_config; the actual
            # save_dir is passed explicitly to ModelTrainer.
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
        return "out of memory" in msg or "cuda" in msg and "memory" in msg
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
        suggested = _suggest_search_space(trial, base_config)
        forced = _forced_overrides(args, trial.number, study_name)

        merged = _deep_merge(base_config, suggested)
        merged = _deep_merge(merged, forced)
        validate_config(merged)

        save_dir = _trial_dir(models_root, study_name, trial.number)
        if not ensure_dirs(save_dir):
            raise RuntimeError(f"Failed to create trial dir: {save_dir}")

        if not save_json(merged, save_dir / "trial_config.json"):
            raise RuntimeError(f"Failed to save trial_config.json in {save_dir}")

        # Stash a compact summary on the trial for the leaderboard.
        trial.set_user_attr("model_type", merged["model_hyperparameters"]["model_type"])
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
            best_val = trainer.train()
            return float(best_val)
        except optuna.TrialPruned:
            raise
        except BaseException as exc:
            if _is_oom_error(exc):
                logger.warning(
                    "Trial %d marked pruned due to GPU memory pressure: %s",
                    trial.number, exc,
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
    models_root: Path, study_name: str, top_k: int
) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    """After every N trials, delete checkpoints for trials outside top-K."""

    def _callback(study: optuna.Study, _frozen: optuna.trial.FrozenTrial) -> None:
        completed = [
            t for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
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
                deleted, top_k, sorted(keep_numbers),
            )

    return _callback


def _emit_leaderboard(study: optuna.Study, output_dir: Path) -> None:
    """Write leaderboard.json (top-K by value asc) and all_trials.json."""
    trials = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
    )

    def _serialize(t: optuna.trial.FrozenTrial) -> Dict[str, Any]:
        duration_s = None
        if t.datetime_start is not None and t.datetime_complete is not None:
            duration_s = (t.datetime_complete - t.datetime_start).total_seconds()
        return {
            "number": t.number,
            "state": t.state.name,
            "value": None if t.value is None or not math.isfinite(t.value) else float(t.value),
            "params": dict(t.params),
            "user_attrs": dict(t.user_attrs),
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_s": duration_s,
        }

    serialized = [_serialize(t) for t in trials]
    completed = [s for s in serialized if s["state"] == "COMPLETE" and s["value"] is not None]
    completed.sort(key=lambda s: s["value"])

    leaderboard = {
        "study_name": study.study_name,
        "direction": "minimize",
        "n_trials_total": len(trials),
        "n_trials_complete": len(completed),
        "best_trial_number": completed[0]["number"] if completed else None,
        "best_value": completed[0]["value"] if completed else None,
        "best_params": completed[0]["params"] if completed else None,
        "top_k": completed[:LEADERBOARD_TOP_K],
        "note": (
            "Tuned on a small data fraction with short epochs. To deploy, copy "
            "best_params into a fresh .jsonc config and run train.sh on full data."
        ),
    }

    if not save_json(leaderboard, output_dir / "leaderboard.json"):
        raise RuntimeError("Failed to write leaderboard.json")
    if not save_json({"trials": serialized}, output_dir / "all_trials.json"):
        raise RuntimeError("Failed to write all_trials.json")
    logger.info("Wrote leaderboard with %d completed trials.", len(completed))


def _build_study(args: argparse.Namespace) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=TPE_N_STARTUP_TRIALS,
        n_ei_candidates=TPE_N_EI_CANDIDATES,
        multivariate=True,
        group=True,
        constant_liar=True,
        seed=int(args.sampler_seed),
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=PRUNE_MIN_RESOURCE_EPOCHS,
        max_resource=int(args.epochs),
        reduction_factor=HYPERBAND_REDUCTION_FACTOR,
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
        description="Optuna hyperparameter search for the Problemulator emulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "transformer_v2.jsonc",
        help="Base config file (full schema). Trial overrides merge on top.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data dir (contains raw/ and processed/).",
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
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT_SECONDS,
                        help="Walltime budget in seconds.")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--data-fraction", type=float, default=DATA_FRACTION_DEFAULT)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE_DEFAULT)
    parser.add_argument("--sampler-seed", type=int, default=SAMPLER_SEED_DEFAULT)
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

        # Per-study log file alongside study.db so resumes accumulate context.
        setup_logging(log_file=study_dir / "tune_run.log", force=True)
        logger.info("Tune CLI args: %s", json.dumps(vars(args), default=str))

        # Validation ranges: epochs must be > warmup_epochs; pruner needs epochs > min_resource.
        if args.epochs <= WARMUP_EPOCHS_DEFAULT:
            raise ValueError(
                f"--epochs ({args.epochs}) must be > WARMUP_EPOCHS_DEFAULT ({WARMUP_EPOCHS_DEFAULT})."
            )
        if args.epochs <= PRUNE_MIN_RESOURCE_EPOCHS:
            raise ValueError(
                f"--epochs ({args.epochs}) must be > PRUNE_MIN_RESOURCE_EPOCHS "
                f"({PRUNE_MIN_RESOURCE_EPOCHS}); the pruner needs room to compare."
            )
        if not (0.0 < args.data_fraction <= 1.0):
            raise ValueError("--data-fraction must be in (0, 1].")

        base_config = load_config(args.base_config)

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

        # Build/validate processed artifacts ONCE, shared across all trials.
        # Force the base config into a known-safe shape for normalize:
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
            1 for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        logger.info(
            "Optuna study '%s' loaded (existing completed trials=%d). Storage: %s",
            args.study_name, existing_complete, args.storage,
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
            args.models_dir, args.study_name, LEADERBOARD_TOP_K
        )

        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            callbacks=[gc_callback],
            catch=(),
        )

        _emit_leaderboard(study, study_dir)
        logger.info("Tuning complete. Study dir: %s", study_dir.resolve())
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
