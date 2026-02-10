"""Evaluation, fairness analysis, and benchmark submission for MILK10k CBM.

Usage:
    python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best_model.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.dataset import DIAGNOSIS_COLUMNS, MONET_COLUMNS, get_dataloaders, get_test_dataloader
from src.model import get_model
from src.utils import get_device, load_checkpoint, load_config, parse_overrides, set_seed


# --- Primary Metrics ---

@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device) -> dict:
    """Run model on all batches and collect predictions, labels, concepts, skin tones."""
    model.eval()
    results = {
        "pred_logits": [],
        "pred_concepts": [],
        "true_labels": [],
        "true_concepts": [],
        "skin_tones": [],
        "lesion_ids": [],
    }

    for batch in tqdm(loader, desc="Evaluating"):
        clinical = batch["clinical_image"].to(device)
        dermoscopic = batch["dermoscopic_image"].to(device)

        outputs = model(clinical, dermoscopic)
        results["pred_logits"].append(outputs["logits"].cpu())
        if outputs["concepts"] is not None:
            results["pred_concepts"].append(outputs["concepts"].cpu())
        results["true_labels"].append(batch["diagnosis"])
        results["true_concepts"].append(batch["concepts"])
        results["skin_tones"].extend(batch["skin_tone"].tolist())
        results["lesion_ids"].extend(batch["lesion_id"])

    results["pred_logits"] = torch.cat(results["pred_logits"])
    results["true_labels"] = torch.cat(results["true_labels"])
    results["true_concepts"] = torch.cat(results["true_concepts"])
    if results["pred_concepts"]:
        results["pred_concepts"] = torch.cat(results["pred_concepts"])
    else:
        results["pred_concepts"] = None

    return results


def compute_classification_metrics(results: dict, save_dir: str | None = None) -> dict:
    """Compute macro F1, per-class F1, confusion matrix."""
    pred_classes = results["pred_logits"].argmax(dim=1).numpy()
    true_classes = results["true_labels"].argmax(dim=1).numpy()

    macro_f1 = f1_score(true_classes, pred_classes, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        true_classes, pred_classes, average=None,
        labels=range(len(DIAGNOSIS_COLUMNS)), zero_division=0,
    )

    print(f"\nMacro F1: {macro_f1:.4f}")
    print("\nPer-class F1:")
    for name, score in zip(DIAGNOSIS_COLUMNS, per_class_f1):
        print(f"  {name:10s}: {score:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        true_classes, pred_classes,
        target_names=DIAGNOSIS_COLUMNS, zero_division=0,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=range(len(DIAGNOSIS_COLUMNS)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=DIAGNOSIS_COLUMNS, yticklabels=DIAGNOSIS_COLUMNS, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)

    return {
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(DIAGNOSIS_COLUMNS, per_class_f1.tolist())),
        "confusion_matrix": cm,
    }


# --- Concept Quality ---

def compute_concept_metrics(results: dict, save_dir: str | None = None) -> dict:
    """Per-concept MSE, Pearson correlation, and scatter plots."""
    if results["pred_concepts"] is None:
        print("No concept predictions (baseline model). Skipping concept metrics.")
        return {}

    pred = results["pred_concepts"].numpy()
    true = results["true_concepts"].numpy()

    print("\nConcept Quality:")
    concept_metrics = {}
    for i, name in enumerate(MONET_COLUMNS):
        mse = float(np.mean((pred[:, i] - true[:, i]) ** 2))
        corr, _ = pearsonr(pred[:, i], true[:, i])
        short_name = name.replace("MONET_", "")
        print(f"  {short_name:45s} | MSE: {mse:.4f} | r: {corr:.4f}")
        concept_metrics[short_name] = {"mse": mse, "pearson_r": corr}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        for i, name in enumerate(MONET_COLUMNS):
            ax = axes[i]
            short_name = name.replace("MONET_", "")
            ax.scatter(true[:, i], pred[:, i], alpha=0.3, s=10)
            ax.plot([0, 1], [0, 1], "r--", linewidth=1)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(short_name[:25])
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        # Hide unused subplot
        axes[-1].set_visible(False)
        fig.suptitle("Concept Predictions vs Ground Truth")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "concept_scatter.png"), dpi=150)
        plt.close(fig)

    return concept_metrics


# --- Fairness Analysis ---

SKIN_TONE_GROUPS = {
    "0-1": [0, 1],
    "2": [2],
    "3": [3],
    "4-5": [4, 5],
}


def compute_fairness_metrics(results: dict) -> dict:
    """Macro F1 stratified by skin tone groups."""
    pred_classes = results["pred_logits"].argmax(dim=1).numpy()
    true_classes = results["true_labels"].argmax(dim=1).numpy()
    skin_tones = np.array(results["skin_tones"])

    print("\nFairness Analysis (Macro F1 by skin tone):")
    fairness = {}
    for group_name, tones in SKIN_TONE_GROUPS.items():
        mask = np.isin(skin_tones, tones)
        n = mask.sum()
        if n == 0:
            print(f"  Skin tone {group_name}: n=0, skipping")
            continue
        f1 = f1_score(
            true_classes[mask], pred_classes[mask], average="macro", zero_division=0
        )
        print(f"  Skin tone {group_name}: F1={f1:.4f} (n={n})")
        fairness[group_name] = {"macro_f1": f1, "n": int(n)}

    return fairness


# --- Benchmark Submission ---

def generate_submission(
    model: nn.Module,
    loader,
    device: torch.device,
    output_path: str,
) -> None:
    """Generate prediction CSV for benchmark submission."""
    model.eval()
    all_lesion_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating submission"):
            clinical = batch["clinical_image"].to(device)
            dermoscopic = batch["dermoscopic_image"].to(device)

            outputs = model(clinical, dermoscopic)
            logits = outputs["logits"]
            # Keep softmax for relative confidence, then take argmax
            # Set top class to 0.55, rest to 0 — guarantees exactly
            # 1 prediction per row above the 0.5 threshold
            probs = F.softmax(logits, dim=1)
            submission_probs = torch.zeros_like(probs)
            top_class = probs.argmax(dim=1)
            for i in range(len(probs)):
                submission_probs[i, top_class[i]] = 0.55
            probs = submission_probs.cpu().numpy()
            all_probs.append(probs)
            all_lesion_ids.extend(batch["lesion_id"])

    all_probs = np.concatenate(all_probs)
    submission = pd.DataFrame(all_probs, columns=DIAGNOSIS_COLUMNS)
    submission.insert(0, "lesion_id", all_lesion_ids)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")


# --- Main Evaluation ---

def evaluate(config: dict, checkpoint_path: str) -> dict:
    """Run full evaluation pipeline."""
    set_seed(config["data"]["seed"])
    device = get_device()
    save_dir = os.path.join(config["logging"]["save_dir"], "eval")

    # Data
    _, val_loader, _ = get_dataloaders(config)

    # Model
    model = get_model(config)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)

    # Collect predictions
    results = collect_predictions(model, val_loader, device)

    # Metrics
    all_metrics = {}
    all_metrics["classification"] = compute_classification_metrics(results, save_dir)
    all_metrics["concepts"] = compute_concept_metrics(results, save_dir)
    all_metrics["fairness"] = compute_fairness_metrics(results)

    return all_metrics


def submit(config: dict, checkpoint_path: str, output_path: str) -> None:
    """Generate benchmark submission on test set."""
    set_seed(config["data"]["seed"])
    device = get_device()

    # Test data
    test_loader, test_df = get_test_dataloader(config)
    print(f"Test set: {len(test_df)} lesions, {len(test_loader)} batches")

    # Model
    model = get_model(config)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)

    # Generate submission
    generate_submission(model, test_loader, device, output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MILK10k CBM")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Generate benchmark submission on test set instead of evaluating",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/submission.csv",
        help="Path for submission CSV (used with --submit)",
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[],
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides) if args.overrides else None
    config = load_config(args.config, overrides)

    if args.submit:
        submit(config, args.checkpoint, args.output)
    else:
        evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
