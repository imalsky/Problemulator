#!/usr/bin/env python3
"""Device detection and DataLoader optimization helpers."""

import logging
import torch

logger = logging.getLogger(__name__)


def _mps_is_available() -> bool:
    """Return whether the current PyTorch build can execute on Apple MPS."""
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def setup_device(requested_backend: str) -> torch.device:
    """Select the configured compute backend and hard-fail on unavailability."""
    backend = requested_backend.strip().lower()
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Configured backend 'cuda' is not available.")
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"Using CUDA: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1024**3:.1f}GB)"
        )
        return device

    if backend == "mps":
        if not _mps_is_available():
            raise RuntimeError("Configured backend 'mps' is not available.")
        logger.info("Using Apple MPS device.")
        return torch.device("mps")

    if backend == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")

    raise ValueError(f"Unsupported device backend '{requested_backend}'.")


def should_pin_memory(device: torch.device) -> bool:
    """Determine if memory pinning should be enabled."""
    return device.type == "cuda"


__all__ = ["setup_device", "should_pin_memory", "_mps_is_available"]
