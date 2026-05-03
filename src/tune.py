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
from typing import Any, Callable, Dict, List, Tuple

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

# -----------------------------------------------------------------------------
# Search-loop constants.
# -----------------------------------------------------------------------------

# Median pruning is simpler than Hyperband for this compact comparison. It avoids
# the earlier failure mode where many transformer trials were killed too early.
PRUNE_WARMUP_EPOCHS = 10

TPE_N_STARTUP_TRIALS = 12
TPE_N_EI_CANDIDATES = 32

LEADERBOARD_TOP_K = 5
GC_CALLBACK_EVERY_N_TRIALS = 5

DATA_FRACTION_DEFAULT = 0.1
EPOCHS_DEFAULT = 25
EARLY_STOPPING_PATIENCE_DEFAULT = 8
WARMUP_EPOCHS_DEFAULT = 2
SAMPLER_SEED_DEFAULT = 42
N_TRIALS_DEFAULT = 200
TIMEOUT_DEFAULT_SECONDS = 248_400  # ~69 h; timeout still acts as the real cap.

# -----------------------------------------------------------------------------
# Compact search space.
# -----------------------------------------------------------------------------

# Shared architecture choices.
# d_model=128 was consistently uncompetitive; removed.
D_MODEL_CHOICES: Tuple[int, ...] = (256, 512)
# Dropout=0 was the strongest value in the prior search, but the reviewer
# specifically questions whether zero-dropout is robust. Three light values
# are kept so the new search can confirm or refute that empirically.
DROPOUT_CHOICES: Tuple[float, ...] = (0.0, 0.025, 0.05)
OUTPUT_HEAD_DIVISOR_CHOICES: Tuple[int, ...] = (1, 2, 4)
# output_head_dropout_factor=0.5 was never competitive; removed.
OUTPUT_HEAD_DROPOUT_FACTOR_CHOICES: Tuple[float, ...] = (0.0, 0.25)
# film_clamp=2.0 was consistently the weakest value; removed.
FILM_CLAMP_CHOICES: Tuple[float, ...] = (5.0, 10.0, 50.0)

# Transformer-only choices.
# ffn_type: swiglu mean was 33% worse than gelu; fixed to gelu (TRANSFORMER_FFN_TYPE_FIXED).
# use_qk_norm: False trials had best=2.73e-4 vs True best=2.14e-4; fixed to True.
TRANSFORMER_FFN_TYPE_FIXED: str = "gelu"
TRANSFORMER_QK_NORM_FIXED: bool = True
TRANSFORMER_LAYER_CHOICES: Tuple[int, ...] = (2, 3, 4)
TRANSFORMER_NHEAD_CHOICES: Tuple[int, ...] = (2, 4, 8)
TRANSFORMER_FFN_MULT_CHOICES: Tuple[int, ...] = (2, 4, 8)
# attention_dropout=0.05 had only 1 trial with the worst mean; removed.
TRANSFORMER_ATTENTION_DROPOUT_CHOICES: Tuple[float, ...] = (0.0, 0.025, 0.1)

# LSTM-only choices.
# bidirectional=False was 4–5× worse than True; fixed to True.
# num_layers=1 was consistently weaker; removed.
LSTM_BIDIRECTIONAL_FIXED: bool = True
LSTM_LAYER_CHOICES: Tuple[int, ...] = (2, 3)


def _valid_heads(d_model: int) -> Tuple[int, ...]:
    """Return legal attention-head counts for ``d_model``."""
    return tuple(h for h in TRANSFORMER_NHEAD_CHOICES if d_model % h == 0)


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


def _choose_model_type(trial: optuna.Trial, requested_family: str) -> str:
    """Select model family without letting TPE abandon one branch.

    ``both`` alternates by trial number:
    - even trials: transformer
    - odd trials: LSTM

    This makes early comparisons balanced and deterministic.
    """
    family = str(requested_family).lower()
    if family == "both":
        return "transformer" if trial.number % 2 == 0 else "lstm"
    if family in {"transformer", "lstm"}:
        return family
    raise ValueError(f"Unknown model family '{requested_family}'.")


def _suggest_common_model_block(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample shared architecture settings."""
    d_model = int(trial.suggest_categorical("d_model", list(D_MODEL_CHOICES)))
    dropout = float(trial.suggest_categorical("dropout", list(DROPOUT_CHOICES)))
    output_head_divisor = int(
        trial.suggest_categorical("output_head_divisor", list(OUTPUT_HEAD_DIVISOR_CHOICES))
    )
    output_head_dropout_factor = float(
        trial.suggest_categorical(
            "output_head_dropout_factor", list(OUTPUT_HEAD_DROPOUT_FACTOR_CHOICES)
        )
    )
    film_clamp = float(trial.suggest_categorical("film_clamp", list(FILM_CLAMP_CHOICES)))

    # Guard against invalid output-head widths if d_model is small.
    if output_head_divisor > d_model:
        raise optuna.TrialPruned(
            f"output_head_divisor={output_head_divisor} invalid for d_model={d_model}."
        )

    return {
        "d_model": d_model,
        "dropout": dropout,
        "output_head_divisor": output_head_divisor,
        "output_head_dropout_factor": output_head_dropout_factor,
        "film_clamp": film_clamp,
    }


def _suggest_search_space(
    trial: optuna.Trial,
    base_config: Dict[str, Any],
    requested_family: str,
) -> Dict[str, Any]:
    """Sample a compact per-trial config delta.

    This intentionally does NOT sample:
    - learning_rate
    - weight_decay
    - batch_size
    - Adam betas
    - EMA decay
    - scheduler type

    Those stay fixed from the base config, which makes the architecture
    comparison much faster and easier to interpret.
    """
    del base_config  # Kept in signature for future extension and objective symmetry.

    model_type = _choose_model_type(trial, requested_family)
    trial.set_user_attr("model_type", model_type)

    common = _suggest_common_model_block(trial)

    if model_type == "transformer":
        d_model = int(common["d_model"])
        valid_heads = _valid_heads(d_model)
        if not valid_heads:
            raise optuna.TrialPruned(f"No valid nhead for d_model={d_model}.")

        nhead = int(trial.suggest_categorical("nhead", list(valid_heads)))
        num_layers = int(
            trial.suggest_categorical("num_layers_transformer", list(TRANSFORMER_LAYER_CHOICES))
        )
        ffn_mult = int(
            trial.suggest_categorical("ffn_mult", list(TRANSFORMER_FFN_MULT_CHOICES))
        )
        attention_dropout = float(
            trial.suggest_categorical(
                "attention_dropout", list(TRANSFORMER_ATTENTION_DROPOUT_CHOICES)
            )
        )

        model_block = {
            "model_type": "transformer",
            **common,
            "transformer": {
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": int(d_model * ffn_mult),
                "attention_dropout": attention_dropout,
                "use_qk_norm": TRANSFORMER_QK_NORM_FIXED,
                "qkv_bias": True,
                "ffn_type": TRANSFORMER_FFN_TYPE_FIXED,
            },
        }

    else:
        num_layers = int(
            trial.suggest_categorical("num_layers_lstm", list(LSTM_LAYER_CHOICES))
        )

        model_block = {
            "model_type": "lstm",
            **common,
            "lstm": {
                "num_layers": num_layers,
                "bidirectional": LSTM_BIDIRECTIONAL_FIXED,
            },
        }

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
        suggested = _suggest_search_space(trial, base_config, args.model_family)
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

            best_val = trainer.train()
            return float(best_val)

        except optuna.TrialPruned:
            raise
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
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=TPE_N_STARTUP_TRIALS,
        n_ei_candidates=TPE_N_EI_CANDIDATES,
        multivariate=True,
        group=True,
        constant_liar=True,
        seed=int(args.sampler_seed),
    )

    # When comparing both architectures, MedianPruner is unfair: transformer
    # trials converge faster and dominate the shared median, causing LSTM trials
    # to be pruned before they reach their asymptotic loss. Use NopPruner so
    # every trial runs to early-stopping completion on equal footing.
    if args.model_family == "both":
        pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()
        logger.info(
            "model_family='both': using NopPruner so both architectures run "
            "to early-stopping completion without cross-architecture bias."
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=TPE_N_STARTUP_TRIALS,
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
        default=PROJECT_ROOT / "config" / "transformer_v2.jsonc",
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
        choices=("both", "transformer", "lstm"),
        default="both",
        help=(
            "'both' alternates transformer/LSTM trials. Use 'transformer' for a "
            "transformer-only search."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
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

        if study.best_trial is not None:
            logger.info(
                "Best trial: number=%d value=%.6e params=%s user_attrs=%s",
                study.best_trial.number,
                float(study.best_trial.value),
                json.dumps(study.best_trial.params, sort_keys=True),
                json.dumps(study.best_trial.user_attrs, sort_keys=True, default=str),
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
