"""Configuration loading, seeding, checkpointing, and helper utilities."""

import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: dict | None = None) -> dict:
    """Load YAML config and optionally merge CLI overrides.

    Args:
        path: Path to the YAML config file.
        overrides: Dict of dotted-key overrides, e.g. {"training.phase1_lr": 0.001}.

    Returns:
        Nested config dict.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return config


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across torch, numpy, random, and CUDA.

    Args:
        seed: Random seed.
        deterministic: If True, force cuDNN deterministic mode (slower).
            On Hopper/GH200, benchmark=True is preferred for performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """Save model checkpoint with optimizer state and metrics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint. Returns the saved metadata dict.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optionally restore optimizer state.

    Returns:
        Dict with keys: epoch, metrics.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {"epoch": checkpoint["epoch"], "metrics": checkpoint["metrics"]}


def parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI override strings like 'training.phase1_lr=0.01' into a dict.

    Values are automatically cast to int, float, or bool where possible.
    """
    overrides = {}
    for item in override_list:
        key, value = item.split("=", 1)
        # Try type casting
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string
        overrides[key] = value
    return overrides
