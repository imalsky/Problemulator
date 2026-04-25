#!/usr/bin/env python3
"""Training loop orchestration for the atmospheric-profile emulator."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from dataset import create_dataset
from hardware import should_pin_memory
from model import create_prediction_model
from utils import get_precision_config, save_json, seed_everything

logger = logging.getLogger(__name__)


def _assert_finite_tensor(tensor: torch.Tensor, *, label: str, mode: str, batch_idx: int) -> None:
    """Raise a descriptive error when a training/eval tensor contains NaN or Inf."""
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all()):
        return

    bad_count = int((~finite_mask).sum().item())
    raise RuntimeError(
        f"Non-finite values detected in {label} during {mode} batch {batch_idx} "
        f"({bad_count} values)."
    )


class WarmupScheduler:
    """
    Apply linear warmup before handing control to a downstream scheduler.

    The wrapped scheduler can be either step-based (for example cosine) or
    metric-based (for example ReduceLROnPlateau). Warmup updates happen after
    each epoch so the logged learning rate always reflects the value used
    during that epoch.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        warmup_start_factor: float,
        after_scheduler: torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau,
        after_scheduler_requires_metric: bool,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_factor = float(warmup_start_factor)
        self.after_scheduler = after_scheduler
        self.after_scheduler_requires_metric = after_scheduler_requires_metric
        self.completed_epochs = 0
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]

        if self.warmup_epochs > 0:
            self._set_warmup_lr(0)

    def _set_warmup_lr(self, completed_warmup_epochs: int) -> None:
        """Set optimizer learning rates for the current warmup progress."""
        progress = completed_warmup_epochs / self.warmup_epochs
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self, metric: Optional[float] = None) -> None:
        """Advance warmup or the wrapped scheduler by one epoch."""
        if self.completed_epochs < self.warmup_epochs:
            self.completed_epochs += 1
            self._set_warmup_lr(self.completed_epochs)
            return

        if self.after_scheduler_requires_metric:
            if metric is None:
                raise ValueError("WarmupScheduler requires a metric for plateau stepping.")
            self.after_scheduler.step(metric)
        else:
            self.after_scheduler.step()

        self.completed_epochs += 1

class DevicePrefetchLoader:
    """
    Wrap a DataLoader and move batches onto the target device ahead of use.

    On CUDA, this overlaps host-to-device transfer with model execution on a
    side stream. The nested batch containers are updated in place to avoid
    rebuilding small dictionaries for every batch.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        forward_dtype: torch.dtype,
        loss_dtype: torch.dtype,
    ) -> None:
        """
        Initialize prefetch loader.
        
        Args:
            loader: Base DataLoader
            device: Target device for data
            forward_dtype: Dtype for model inputs
            loss_dtype: Dtype for targets/loss
        """
        self.loader = loader
        self.device = device
        self.forward_dtype = forward_dtype
        self.loss_dtype = loss_dtype
        self.is_cuda = self.device.type == 'cuda'
    
    def __iter__(self):
        """
        Yield batches already moved to ``self.device``.

        Each yielded batch keeps the training contract used throughout the codebase:
        ``(inputs, masks, targets, target_masks)`` where
        ``inputs["sequence"]`` is ``[batch, seq_len, input_dim]`` and
        ``targets`` is ``[batch, seq_len, target_dim]``.
        """
        if self.is_cuda:
            stream = torch.cuda.Stream()
            first = True
            current_batch = None
            
            for next_batch in self.loader:
                # Asynchronously transfer next batch
                with torch.cuda.stream(stream):
                    next_batch = self._to_device(next_batch)
                
                # Yield previous batch while next is transferring
                if not first:
                    yield current_batch
                else:
                    first = False
                
                # Wait for transfer to complete
                torch.cuda.current_stream().wait_stream(stream)
                current_batch = next_batch
            
            # Yield final batch
            if current_batch is not None:
                yield current_batch
        else:
            # No prefetching for non-CUDA devices
            for batch in self.loader:
                yield self._to_device(batch)
    
    def _to_device(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Move one collated batch to the target device.

        Args:
            batch: ``(inputs, masks, targets, target_masks)`` from ``pad_collate``.
                ``inputs["sequence"]`` has shape ``[batch, seq_len, input_dim]``.
                ``targets`` has shape ``[batch, seq_len, target_dim]``.

        Returns:
            The same logical batch on ``self.device``. The input and mask
            dictionaries are updated in place; masks remain boolean with
            ``True`` meaning padding so they can be consumed directly by
            attention and loss code.
        """
        inputs, masks, targets, tgt_masks = batch
        non_blocking = self.is_cuda

        inputs["sequence"] = inputs["sequence"].to(
            self.device, dtype=self.forward_dtype, non_blocking=non_blocking
        )
        if "global_features" in inputs:
            inputs["global_features"] = inputs["global_features"].to(
                self.device, dtype=self.forward_dtype, non_blocking=non_blocking
            )

        masks["sequence"] = masks["sequence"].to(
            self.device, non_blocking=non_blocking
        )

        targets = targets.to(self.device, dtype=self.loss_dtype, non_blocking=non_blocking)
        tgt_masks = tgt_masks.to(self.device, non_blocking=non_blocking)

        return inputs, masks, targets, tgt_masks
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


class ModelTrainer:
    """
    Orchestrates model training, validation, and testing.
    
    Handles all aspects of the training pipeline including:
    - Dataset creation and DataLoader setup
    - Model initialization and optimization
    - Training loop with validation
    - Checkpointing and early stopping
    - Correct padding mask handling throughout
    """

    def __init__(
            self,
            config: Dict[str, Any],
            device: torch.device,
            save_dir: Path,
            processed_dir: Path,
            splits: Dict[str, List[Tuple[str, int]]],
            collate_fn: Callable,
    ) -> None:
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
            device: Compute device
            save_dir: Directory for saving checkpoints
            processed_dir: Directory with preprocessed data
            splits: Train/val/test splits
            collate_fn: Collation function for DataLoader
        """
        self.cfg = config
        self.device = device
        self.save_dir = save_dir
        self.model = None

        # Extract configuration
        misc_cfg = self.cfg["miscellaneous_settings"]
        train_params = self.cfg["training_hyperparameters"]
        self.precision = get_precision_config(self.cfg)
        self.forward_dtype = self.precision["forward_dtype"]
        self.loss_dtype = self.precision["loss_dtype"]
        self._using_prefetch_loader = False

        if self.device.type == "cuda":
            logger.info("Applying GPU performance optimizations")

            # Respect config: do not auto-enable AMP
            if bool(train_params["use_amp"]):
                logger.info("AMP enabled per config")

        # Enable anomaly detection if requested
        if bool(misc_cfg["detect_anomaly"]):
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - training will be slower.")

        # Import compress_splits from utils
        from utils import compress_splits

        # Save splits for reproducibility in COMPACT format
        compressed_splits = compress_splits(splits)
        if not save_json(compressed_splits, self.save_dir / "dataset_splits.json", compact=True):
            raise RuntimeError("Failed to save dataset_splits.json.")

        # Initialize components
        self._setup_datasets(processed_dir, splits)
        self._build_dataloaders(collate_fn, misc_cfg, train_params)
        self._build_model()
        self._build_optimizer(train_params)
        self._build_scheduler(train_params)
        self._setup_training_params(train_params)
        self._setup_logging()
        self._save_metadata()

        # Check if validation set exists
        self.has_val = len(self.val_loader) > 0
        if not self.has_val:
            raise RuntimeError("Validation DataLoader is empty.")
        if len(self.test_loader) == 0:
            raise RuntimeError("Test DataLoader is empty.")

        # Clean up memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Trainer initialized. Padding convention: True = padding position")
    
    def _setup_datasets(
        self, processed_dir: Path, splits: Dict[str, List[Tuple[str, int]]]
    ) -> None:
        """Create datasets from preprocessed data."""
        # Check if using subset
        fraction = float(self.cfg["training_hyperparameters"]["dataset_fraction_to_use"])
        
        if 0.0 < fraction < 1.0:
            logger.warning(f"Using only {fraction:.0%} of the dataset.")
            random_seed = int(self.cfg["miscellaneous_settings"]["random_seed"])
            seed_everything(random_seed)
        
        # Dataset directories
        train_dir = processed_dir / "train"
        val_dir = processed_dir / "val"
        test_dir = processed_dir / "test"

        missing_dirs = [d for d in (train_dir, val_dir, test_dir) if not d.exists()]
        if missing_dirs:
            raise RuntimeError(
                f"Missing required processed split directories: {[str(p) for p in missing_dirs]}"
            )
        
        def get_indices(split_data: List[Tuple[str, int]]) -> List[int] | None:
            """Get indices, optionally sampling a fraction."""
            num_samples = len(split_data)
            if num_samples == 0:
                return []
            
            if 0.0 < fraction < 1.0:
                import random
                k = max(1, int(num_samples * fraction))
                return random.sample(range(num_samples), k)
            
            # Full split requested: keep identity indexing to avoid large list materialization.
            return None
        
        # Create datasets
        train_idx = get_indices(splits["train"])
        val_idx = get_indices(splits["validation"])
        test_idx = get_indices(splits["test"])
        
        self.train_ds = create_dataset(train_dir, self.cfg, train_idx)
        self.val_ds = create_dataset(val_dir, self.cfg, val_idx)
        self.test_ds = create_dataset(test_dir, self.cfg, test_idx)
        
        logger.info(
            f"Datasets ready - train:{len(self.train_ds):,}  "
            f"val:{len(self.val_ds):,}  test:{len(self.test_ds):,}"
        )
    
    def _build_dataloaders(
        self, collate_fn: Callable, misc_cfg: Dict, train_cfg: Dict
    ) -> None:
        """Create DataLoaders with optimized settings."""
        pin_memory = should_pin_memory(self.device)
        num_workers = int(misc_cfg["num_workers"])

        # Common DataLoader arguments
        dl_common = dict(
            batch_size=int(train_cfg["batch_size"]),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        if pin_memory and self.device.type == "cuda":
            dl_common["pin_memory_device"] = "cuda"
        if num_workers > 0:
            dl_common["persistent_workers"] = True
            dl_common["prefetch_factor"] = 2

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=False, **dl_common
        )

        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, drop_last=False, **dl_common
        )

        self.test_loader = DataLoader(
            self.test_ds, shuffle=False, drop_last=False, **dl_common
        )
        
        # Wrap with device prefetching for async CUDA transfers
        if self.device.type == "cuda":
            logger.info("Using DevicePrefetchLoader for CUDA transfers")
            self.train_loader = DevicePrefetchLoader(
                self.train_loader, self.device, self.forward_dtype, self.loss_dtype
            )
            self.val_loader = DevicePrefetchLoader(
                self.val_loader, self.device, self.forward_dtype, self.loss_dtype
            )
            self.test_loader = DevicePrefetchLoader(
                self.test_loader, self.device, self.forward_dtype, self.loss_dtype
            )
            self._using_prefetch_loader = True
    
    def _build_model(self) -> None:
        """Create and initialize the model."""
        self.model = create_prediction_model(
            self.cfg, device=self.device, compile_model=True
        )
    
    def _build_optimizer(self, tp: Dict) -> None:
        """Create optimizer with weight decay groups."""
        opt_name = str(tp["optimizer"]).lower()
        lr = float(tp["learning_rate"])
        wd = float(tp["weight_decay"])
        
        # Separate parameters for weight decay
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Don't apply weight decay to biases and normalization
            if p.dim() == 1 or "bias" in n or "norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        
        # Parameter groups with different weight decay
        groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        if opt_name != "adamw":
            raise ValueError(f"Unsupported optimizer '{opt_name}'. Only 'adamw' is allowed.")
        self.optimizer = optim.AdamW(groups, lr=lr, fused=(self.device.type == "cuda"))
        
        logger.info(f"Optimizer: {opt_name}  lr={lr:.2e}  wd={wd:.2e}")
    
    def _build_scheduler(self, tp: Dict) -> None:
        """Create learning rate scheduler."""
        scheduler_type = str(tp["scheduler_type"]).lower()
        epochs = int(tp["epochs"])
        warmup_epochs = int(tp["warmup_epochs"])
        if warmup_epochs >= epochs:
            raise ValueError("warmup_epochs must be less than total epochs.")
        warmup_start_factor = float(tp["warmup_start_factor"])
        min_lr = float(tp["min_lr"])

        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=min_lr,
            )
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
                after_scheduler=main_scheduler,
                after_scheduler_requires_metric=False,
            )
            logger.info("Cosine scheduler with %d warmup epochs.", warmup_epochs)
            return

        if scheduler_type == "plateau":
            main_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(tp["plateau_factor"]),
                patience=int(tp["plateau_patience"]),
                threshold=float(tp["plateau_threshold"]),
                threshold_mode=str(tp["plateau_threshold_mode"]).lower(),
                min_lr=min_lr,
            )
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
                after_scheduler=main_scheduler,
                after_scheduler_requires_metric=True,
            )
            logger.info(
                "Plateau scheduler with %d warmup epochs, patience=%d, factor=%.3f, threshold=%.2e.",
                warmup_epochs,
                int(tp["plateau_patience"]),
                float(tp["plateau_factor"]),
                float(tp["plateau_threshold"]),
            )
            return

        raise ValueError(f"Unsupported scheduler_type '{scheduler_type}'.")
    
    def _setup_training_params(self, tp: Dict) -> None:
        """Setup training parameters and loss function."""
        # Loss function (MSE with no reduction for masking)
        self.criterion = nn.MSELoss(reduction="none")
        
        # Gradient clipping
        self.max_grad_norm = float(tp["gradient_clip_val"])
        self.early_stopping_min_delta = float(tp["min_delta"])
        
        # Mixed precision training
        requested_amp = self.precision["use_amp"]
        if requested_amp and self.device.type != "cuda":
            raise RuntimeError("AMP is enabled in config but CUDA is not available.")
        self.use_amp = requested_amp and self.device.type == "cuda"
        self.amp_dtype = self.precision["amp_dtype"] if self.use_amp else None
        # GradScaler only for fp16; bf16 does not need scaling
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)

        if self.use_amp:
            logger.info(f"AMP (Automatic Mixed Precision) enabled with dtype={self.amp_dtype}.")
        logger.info(
            "Precision: input=%s stats=%s model=%s forward=%s loss=%s optimizer_state=%s amp=%s",
            self.precision["input_dtype_name"],
            self.precision["stats_dtype_name"],
            self.precision["model_dtype_name"],
            self.precision["forward_dtype_name"],
            self.precision["loss_dtype_name"],
            self.precision["optimizer_state_dtype_name"],
            self.precision["amp_dtype_name"],
        )

    
    def _setup_logging(self) -> None:
        """Setup training log file."""
        self.log_path = self.save_dir / "training_log.csv"
        self.log_path.write_text("epoch,train_loss,val_loss,lr,time_s,improvement\n")
        
        self.best_val_loss = float("inf")
        self.best_early_stopping_val = float("inf")
        self.best_epoch = -1

    def _save_metadata(self) -> None:
        """Save training metadata."""
        metadata = {
            "device": str(self.device),
            "use_amp": self.use_amp,
            "input_dtype": self.precision["input_dtype_name"],
            "stats_dtype": self.precision["stats_dtype_name"],
            "model_dtype": self.precision["model_dtype_name"],
            "forward_dtype": self.precision["forward_dtype_name"],
            "loss_dtype": self.precision["loss_dtype_name"],
            "optimizer_state_dtype": self.precision["optimizer_state_dtype_name"],
            "amp_autocast_dtype": self.precision["amp_dtype_name"],
            "effective_batch_size": int(self.cfg["training_hyperparameters"]["batch_size"]),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "padding_convention": "True = padding position (PyTorch standard)",
        }
        # Use compact format for metadata
        if not save_json(metadata, self.save_dir / "training_metadata.json", compact=True):
            raise RuntimeError("Failed to save training_metadata.json.")
    
    def train(self) -> float:
        """
        Run the training loop.
        
        Returns:
            Best validation loss achieved
        """
        tp = self.cfg["training_hyperparameters"]
        epochs = int(tp["epochs"])
        patience = int(tp["early_stopping_patience"])
        
        epochs_without_improvement = 0
        
        logger.info(f"Starting training for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            
            # Training epoch
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            
            # Validation epoch
            val_loss = self._run_epoch(self.val_loader, is_train=False)

            previous_best_val = self.best_val_loss
            improvement: Optional[float] = None
            if previous_best_val != float("inf"):
                improvement = previous_best_val - val_loss
            self._log_epoch_results(
                epoch,
                train_loss,
                val_loss,
                time.time() - epoch_start,
                improvement,
                epoch_lr,
            )
            
            # Strict validation decreases always update the best checkpoint.
            if val_loss < previous_best_val:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_best_model()

            # Early stopping uses its own minimum-improvement threshold so the
            # training loop can continue through tiny strict improvements while
            # still stopping once meaningful validation progress has stalled.
            if val_loss < self.best_early_stopping_val - self.early_stopping_min_delta:
                self.best_early_stopping_val = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.scheduler.step(val_loss)

            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs "
                    f"without improvement >= {self.early_stopping_min_delta:.1e}."
                )
                break
        
        logger.info(
            f"Training complete. Best val_loss={self.best_val_loss:.4e} "
            f"at epoch {self.best_epoch}."
        )
        
        return self.best_val_loss
    
    def test(self) -> Dict[str, float]:
        """
        Run testing on the test set.
        
        Returns:
            Dictionary with test metrics
        """
        if len(self.test_loader) == 0:
            raise RuntimeError("Test DataLoader is empty.")
        
        # Load best model checkpoint
        checkpoint_path = self.save_dir / "best_model.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = state["state_dict"]
            
            # Handle compiled model state dict
            if hasattr(self.model, '_orig_mod'):
                # Current model is compiled, but saved state might not be
                if not any(k.startswith("_orig_mod.") for k in state_dict):
                    # Add _orig_mod. prefix to match compiled model
                    state_dict = {
                        f"_orig_mod.{k}": v
                        for k, v in state_dict.items()
                    }
            else:
                # Current model is not compiled, but saved state might be
                if any(k.startswith("_orig_mod.") for k in state_dict):
                    # Remove _orig_mod. prefix
                    state_dict = {
                        k.replace("_orig_mod.", ""): v
                        for k, v in state_dict.items()
                    }

            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded best model from epoch {state['epoch']}")
        
        # Run test evaluation
        test_loss = self._run_epoch(self.test_loader, is_train=False)
        
        # Save test metrics
        metrics = {
            "test_loss": test_loss,
            "best_epoch": self.best_epoch,
        }
        if not save_json(metrics, self.save_dir / "test_metrics.json"):
            raise RuntimeError("Failed to save test_metrics.json.")
        
        logger.info(f"Test loss: {test_loss:.4e}")
        
        return metrics
    
    def _run_epoch(self, loader: DataLoader, is_train: bool) -> float:
        """
        Run one epoch of training or validation.

        Args:
            loader: DataLoader whose batches follow the collate contract
                ``(inputs, masks, targets, target_masks)`` with
                ``inputs["sequence"]`` shaped ``[batch, seq_len, input_dim]`` and
                ``targets`` shaped ``[batch, seq_len, target_dim]``
            is_train: Whether this is a training epoch
            
        Returns:
            Mean masked MSE over all valid target elements in the epoch
        """
        mode = "training" if is_train else "validation"
        if len(loader) == 0:
            raise RuntimeError(f"DataLoader for {mode} is empty.")
        
        # Set model mode
        self.model.train(is_train)
        
        # Track statistics on-device to avoid per-batch host sync.
        total_masked_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_elements = torch.zeros((), device=self.device, dtype=torch.int64)
        zero_valid_batches = torch.zeros((), device=self.device, dtype=torch.int64)
        device_type = str(self.device.type)

        grad_context = torch.enable_grad() if is_train else torch.inference_mode()

        with grad_context:
            for batch_idx, batch in enumerate(loader, start=1):
                # Data already sits on the destination device when the CUDA prefetcher is active.
                inputs, masks, targets, target_masks = batch

                if self._using_prefetch_loader:
                    sequence = inputs["sequence"]
                    global_features = inputs.get("global_features")
                    sequence_mask = masks["sequence"]
                else:
                    sequence = inputs["sequence"].to(
                        device=self.device, dtype=self.forward_dtype
                    )
                    global_features = inputs.get("global_features")
                    if global_features is not None:
                        global_features = global_features.to(
                            device=self.device, dtype=self.forward_dtype
                        )
                    sequence_mask = masks["sequence"].to(device=self.device)
                    targets = targets.to(device=self.device, dtype=self.loss_dtype)
                    target_masks = target_masks.to(device=self.device)

                _assert_finite_tensor(
                    sequence,
                    label="sequence inputs",
                    mode=mode,
                    batch_idx=batch_idx,
                )
                if global_features is not None:
                    _assert_finite_tensor(
                        global_features,
                        label="global features",
                        mode=mode,
                        batch_idx=batch_idx,
                    )
                _assert_finite_tensor(
                    targets,
                    label="targets",
                    mode=mode,
                    batch_idx=batch_idx,
                )

                # Forward pass with autocast
                with autocast(device_type=device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                    predictions = self.model(sequence, global_features, sequence_mask)
                    if predictions.dtype != self.loss_dtype:
                        predictions = predictions.to(dtype=self.loss_dtype)
                    _assert_finite_tensor(
                        predictions,
                        label="model predictions",
                        mode=mode,
                        batch_idx=batch_idx,
                    )

                    # Loss is computed per element, then reduced only across valid
                    # non-padding target elements. This keeps right-padding completely
                    # out of the training objective.
                    unreduced_loss = self.criterion(predictions, targets)
                    _assert_finite_tensor(
                        unreduced_loss,
                        label="unreduced loss",
                        mode=mode,
                        batch_idx=batch_idx,
                    )
                    valid_steps = ~target_masks
                    valid_step_mask = valid_steps.unsqueeze(-1)
                    masked_loss_sum = unreduced_loss.masked_fill(~valid_step_mask, 0).sum()
                    num_valid_elements = valid_steps.sum(dtype=torch.int64) * unreduced_loss.shape[-1]
                    zero_valid_batches += (num_valid_elements == 0).to(dtype=torch.int64)
                    loss = masked_loss_sum / num_valid_elements.clamp_min(1).to(dtype=self.loss_dtype)
                    _assert_finite_tensor(
                        loss.reshape(1),
                        label="masked loss",
                        mode=mode,
                        batch_idx=batch_idx,
                    )

                if is_train:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                    else:
                        loss.backward()

                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                total_masked_loss += masked_loss_sum.detach().to(dtype=torch.float32)
                total_elements += num_valid_elements.detach()

        zero_valid_count = int(zero_valid_batches.item())
        if zero_valid_count > 0:
            raise RuntimeError(
                f"Encountered {zero_valid_count} all-padding batches."
            )

        if int(total_elements.item()) == 0:
            raise RuntimeError("No valid (non-padding) elements were processed in this epoch.")

        avg_loss = total_masked_loss / total_elements.to(dtype=torch.float32)
        return float(avg_loss.item())
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        elapsed_time: float,
        improvement: Optional[float],
        lr: float,
    ) -> None:
        """Log epoch results to console and file."""
        # Console log
        msg = (
            f"Epoch {epoch:03d}  "
            f"train:{train_loss:.3e}  "
            f"val:{val_loss:.3e}  "
            f"lr:{lr:.2e}  "
            f"time:{elapsed_time:.1f}s"
        )
        
        if improvement is not None and improvement > 0:
            msg += f"  ↓{improvement:.3e}"
        
        logger.info(msg)
        
        # File log
        improvement_str = "" if improvement is None else f"{improvement:.6e}"
        with self.log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6e},{val_loss:.6e},"
                f"{lr:.6e},{elapsed_time:.1f},{improvement_str}\n"
            )
    
    def _save_best_model(self) -> None:
        """Save the best model checkpoint."""
        # Prepare checkpoint
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "epoch": self.best_epoch,
            "val_loss": self.best_val_loss,
            "config": self.cfg,
            "padding_info": {
                "convention": "True = padding position",
                "follows_pytorch_standard": True,
                "loss_excludes_padding": True,
                "model_output_not_masked": True,
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
    
__all__ = ["ModelTrainer", "DevicePrefetchLoader"]
