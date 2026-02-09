"""Two-phase training loop for the MILK10k Concept Bottleneck Model.

Phase 1: Frozen backbone, train heads only (warm-up).
Phase 2: Full fine-tuning with differential learning rates.

Usage:
    python -m src.train --config configs/default.yaml [--overrides key=value ...]
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.dataset import DIAGNOSIS_COLUMNS, get_dataloaders
from src.model import get_model
from src.utils import (
    get_device,
    load_config,
    parse_overrides,
    save_checkpoint,
    set_seed,
)


def _resolve_amp(config: dict, device: torch.device):
    """Resolve AMP settings for the current device.

    Returns (use_amp, amp_dtype, scaler).
    - bf16 on Hopper/Ampere+: no GradScaler needed (same exponent range as fp32).
    - fp16 on older GPUs: GradScaler required to avoid underflow.
    - Non-CUDA devices: AMP disabled.
    """
    use_amp = config["training"]["use_amp"] and device.type == "cuda"
    amp_dtype_str = config["training"].get("amp_dtype", "bfloat16")

    if not use_amp:
        return False, torch.float32, None

    if amp_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None  # bf16 needs no scaler
    else:
        # Fall back to fp16 + GradScaler for older GPUs
        return True, torch.float16, torch.amp.GradScaler("cuda")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    concept_criterion: nn.Module,
    cls_criterion: nn.Module,
    device: torch.device,
    config: dict,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float32,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Run one training epoch. Returns dict of average losses."""
    model.train()
    total_loss = 0.0
    total_concept_loss = 0.0
    total_cls_loss = 0.0
    n_batches = 0

    alpha = config["training"]["concept_loss_weight"]
    beta = config["training"]["classification_loss_weight"]
    grad_clip = config["training"]["grad_clip_norm"]
    use_amp = amp_dtype != torch.float32

    for batch in tqdm(loader, desc="Train", leave=False):
        clinical = batch["clinical_image"].to(device, non_blocking=True)
        dermoscopic = batch["dermoscopic_image"].to(device, non_blocking=True)
        concept_targets = batch["concepts"].to(device, non_blocking=True)
        diagnosis_targets = batch["diagnosis"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(clinical, dermoscopic)
            logits = outputs["logits"]

            # Classification loss
            cls_labels = diagnosis_targets.argmax(dim=1)
            loss_cls = cls_criterion(logits, cls_labels)

            # Concept loss (only for CBM variants)
            if outputs["concepts"] is not None:
                loss_concept = concept_criterion(outputs["concepts"], concept_targets)
                loss = alpha * loss_concept + beta * loss_cls
            else:
                loss_concept = torch.tensor(0.0, device=device)
                loss = beta * loss_cls

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Step scheduler per batch (not per epoch) for correct warmup + cosine
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_concept_loss += loss_concept.item()
        total_cls_loss += loss_cls.item()
        n_batches += 1

    return {
        "train/loss": total_loss / n_batches,
        "train/concept_loss": total_concept_loss / n_batches,
        "train/cls_loss": total_cls_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    concept_criterion: nn.Module,
    cls_criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Run validation. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    total_concept_loss = 0.0
    total_cls_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    alpha = config["training"]["concept_loss_weight"]
    beta = config["training"]["classification_loss_weight"]

    for batch in tqdm(loader, desc="Val", leave=False):
        clinical = batch["clinical_image"].to(device)
        dermoscopic = batch["dermoscopic_image"].to(device)
        concept_targets = batch["concepts"].to(device)
        diagnosis_targets = batch["diagnosis"].to(device)

        outputs = model(clinical, dermoscopic)
        logits = outputs["logits"]

        cls_labels = diagnosis_targets.argmax(dim=1)
        loss_cls = cls_criterion(logits, cls_labels)

        if outputs["concepts"] is not None:
            loss_concept = concept_criterion(outputs["concepts"], concept_targets)
            loss = alpha * loss_concept + beta * loss_cls
        else:
            loss_concept = torch.tensor(0.0, device=device)
            loss = beta * loss_cls

        total_loss += loss.item()
        total_concept_loss += loss_concept.item()
        total_cls_loss += loss_cls.item()
        n_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        labels = cls_labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=range(len(DIAGNOSIS_COLUMNS)),
        zero_division=0,
    )

    metrics = {
        "val/loss": total_loss / n_batches,
        "val/concept_loss": total_concept_loss / n_batches,
        "val/cls_loss": total_cls_loss / n_batches,
        "val/macro_f1": macro_f1,
    }
    for i, name in enumerate(DIAGNOSIS_COLUMNS):
        metrics[f"val/f1_{name}"] = per_class_f1[i]

    return metrics


def build_optimizer(model: nn.Module, config: dict, phase: int) -> torch.optim.Optimizer:
    """Build AdamW optimizer with phase-appropriate learning rates."""
    train_cfg = config["training"]

    if phase == 1:
        # Only trainable params (heads), backbone is frozen
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params, lr=train_cfg["phase1_lr"], weight_decay=train_cfg["weight_decay"]
        )
    else:
        # Differential LR: backbone vs heads
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": train_cfg["phase2_backbone_lr"]},
                {"params": head_params, "lr": train_cfg["phase2_head_lr"]},
            ],
            weight_decay=train_cfg["weight_decay"],
        )


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int):
    """Cosine annealing with linear warmup (5% of steps)."""
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _setup_logging(save_dir: str) -> None:
    """Configure logging to console and file."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger.info(f"Logging to {log_file}")


def train(config: dict) -> None:
    """Full two-phase training pipeline."""
    save_dir = config["logging"]["save_dir"]
    _setup_logging(save_dir)

    logger.info(f"Config: {config}")
    set_seed(config["data"]["seed"])
    device = get_device()

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Data
    train_loader, val_loader, full_df = get_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = get_model(config)
    model = model.to(device)
    logger.info(f"Model variant: {config['model']['variant']}")

    # Loss functions — inverse-frequency class weights for rare class support
    concept_criterion = nn.MSELoss()
    label_smoothing = config["training"].get("label_smoothing", 0.0)

    class_counts = full_df[DIAGNOSIS_COLUMNS].sum().values
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights: {dict(zip(DIAGNOSIS_COLUMNS, class_weights.tolist()))}")

    cls_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )

    # AMP setup — bf16 on Hopper needs no GradScaler, fp16 on older GPUs does
    use_amp, amp_dtype, scaler = _resolve_amp(config, device)
    logger.info(f"AMP: {use_amp}, dtype: {amp_dtype}, GradScaler: {scaler is not None}")

    # W&B init
    if HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
        )

    best_f1 = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    # ---- Phase 1: Frozen backbone ----
    logger.info("=== Phase 1: Frozen backbone ===")
    model.freeze_backbone()
    optimizer = build_optimizer(model, config, phase=1)

    for epoch in range(1, config["training"]["phase1_epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, concept_criterion, cls_criterion,
            device, config, scaler, amp_dtype,
        )
        val_metrics = validate(
            model, val_loader, concept_criterion, cls_criterion, device, config
        )

        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "phase": 1}
        _log_epoch(epoch, metrics, optimizer)

        if val_metrics["val/macro_f1"] > best_f1:
            best_f1 = val_metrics["val/macro_f1"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(save_dir, "best_model.pt"),
            )
            logger.info(f"  New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ---- Phase 2: Full fine-tuning ----
    logger.info("=== Phase 2: Full fine-tuning ===")
    model.unfreeze_backbone()
    optimizer = build_optimizer(model, config, phase=2)
    total_steps = config["training"]["phase2_epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps)
    # Re-resolve scaler for phase 2 (fresh state)
    _, _, scaler = _resolve_amp(config, device)
    patience_counter = 0

    for epoch in range(1, config["training"]["phase2_epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, concept_criterion, cls_criterion,
            device, config, scaler, amp_dtype, scheduler,
        )

        val_metrics = validate(
            model, val_loader, concept_criterion, cls_criterion, device, config
        )

        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "phase": 2}
        _log_epoch(epoch, metrics, optimizer)

        if val_metrics["val/macro_f1"] > best_f1:
            best_f1 = val_metrics["val/macro_f1"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(save_dir, "best_model.pt"),
            )
            logger.info(f"  New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Training complete. Best macro F1: {best_f1:.4f}")

    if HAS_WANDB:
        wandb.finish()


def _log_epoch(epoch: int, metrics: dict, optimizer: torch.optim.Optimizer) -> None:
    """Log metrics to logger and optionally to W&B."""
    phase = metrics.get("phase", "?")
    lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"Phase {phase} | Epoch {epoch:3d} | "
        f"Loss: {metrics['train/loss']:.4f} | "
        f"Concept: {metrics['train/concept_loss']:.4f} | "
        f"Cls: {metrics['train/cls_loss']:.4f} | "
        f"Val Loss: {metrics['val/loss']:.4f} | "
        f"Val F1: {metrics['val/macro_f1']:.4f} | "
        f"LR: {lr:.2e}"
    )
    # Log per-class F1 every 5 epochs
    if epoch % 5 == 0:
        per_class = {k: v for k, v in metrics.items() if k.startswith("val/f1_")}
        if per_class:
            parts = [f"{k.replace('val/f1_', '')}: {v:.3f}" for k, v in per_class.items()]
            logger.info(f"  Per-class F1: {' | '.join(parts)}")
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics)


def main():
    parser = argparse.ArgumentParser(description="Train MILK10k CBM")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overrides as key=value pairs",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides) if args.overrides else None
    config = load_config(args.config, overrides)
    train(config)


if __name__ == "__main__":
    main()
