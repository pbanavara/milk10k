"""Data pipeline for the MILK10k Concept Bottleneck Model.

Handles CSV joining, Dataset class, patient-level stratified splits,
image transforms, and DataLoader construction.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MONET_COLUMNS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

DIAGNOSIS_COLUMNS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_lesion_dataframe(metadata_csv: str, groundtruth_csv: str) -> pd.DataFrame:
    """Join metadata and ground-truth CSVs into one row per lesion.

    Pivots the metadata by image_type so each lesion row has columns for
    both clinical and dermoscopic isic_ids plus dermoscopic MONET scores.

    Returns:
        DataFrame with columns: lesion_id, clinical_isic_id, dermoscopic_isic_id,
        7 MONET concept columns, 11 diagnosis columns, skin_tone_class,
        age_approx, sex, site.
    """
    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(groundtruth_csv)

    # Split by image type
    clinical = meta[meta["image_type"] == "clinical: close-up"].copy()
    dermoscopic = meta[meta["image_type"] == "dermoscopic"].copy()

    # Keep only needed columns from each
    clinical_cols = clinical[["lesion_id", "isic_id"]].rename(
        columns={"isic_id": "clinical_isic_id"}
    )
    derm_cols = dermoscopic[
        ["lesion_id", "isic_id", "age_approx", "sex", "skin_tone_class", "site"]
        + MONET_COLUMNS
    ].rename(columns={"isic_id": "dermoscopic_isic_id"})

    # Merge clinical + dermoscopic on lesion_id
    merged = pd.merge(clinical_cols, derm_cols, on="lesion_id", how="inner")

    # Merge with ground truth
    merged = pd.merge(merged, gt, on="lesion_id", how="inner")

    return merged


def get_split_indices(
    df: pd.DataFrame, val_split: float, seed: int, save_path: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Patient-level stratified train/val split.

    Uses lesion_id as patient proxy. Stratifies by diagnosis class.

    Args:
        df: Lesion-level DataFrame with diagnosis columns.
        val_split: Fraction for validation (e.g. 0.2).
        seed: Random seed.
        save_path: If provided, save split indices to JSON.

    Returns:
        (train_indices, val_indices) as numpy arrays.
    """
    # Get the diagnosis class label for stratification (argmax of one-hot)
    labels = df[DIAGNOSIS_COLUMNS].values.argmax(axis=1)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(df, labels))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {"train": train_idx.tolist(), "val": val_idx.tolist()}, f
            )

    return train_idx, val_idx


def get_transforms(image_size: int, is_training: bool) -> transforms.Compose:
    """Build image transforms for training or validation."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class MILK10kDataset(Dataset):
    """Dataset for the MILK10k skin lesion CBM.

    Each sample represents one lesion and returns both clinical and
    dermoscopic images, MONET concept targets, and diagnosis labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            df: Lesion-level DataFrame from build_lesion_dataframe.
            images_dir: Path to image directory.
            transform: Torchvision transforms applied to both images.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load images
        clinical_path = os.path.join(
            self.images_dir, f"{row['clinical_isic_id']}.jpg"
        )
        dermoscopic_path = os.path.join(
            self.images_dir, f"{row['dermoscopic_isic_id']}.jpg"
        )

        clinical_img = Image.open(clinical_path).convert("RGB")
        dermoscopic_img = Image.open(dermoscopic_path).convert("RGB")

        if self.transform:
            clinical_img = self.transform(clinical_img)
            dermoscopic_img = self.transform(dermoscopic_img)

        # MONET concept targets (from dermoscopic image)
        concepts = torch.tensor(
            row[MONET_COLUMNS].values.astype(np.float32), dtype=torch.float32
        )

        # Diagnosis labels (one-hot)
        diagnosis = torch.tensor(
            row[DIAGNOSIS_COLUMNS].values.astype(np.float32), dtype=torch.float32
        )

        # Skin tone for fairness analysis
        skin_tone = int(row["skin_tone_class"]) if pd.notna(row["skin_tone_class"]) else -1

        return {
            "lesion_id": row["lesion_id"],
            "clinical_image": clinical_img,
            "dermoscopic_image": dermoscopic_img,
            "concepts": concepts,
            "diagnosis": diagnosis,
            "skin_tone": skin_tone,
        }


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    """Build train and validation DataLoaders from config.

    Args:
        config: Full config dict (from load_config).

    Returns:
        (train_loader, val_loader, full_df) tuple.
    """
    data_cfg = config["data"]

    # Build merged dataframe
    df = build_lesion_dataframe(
        data_cfg["metadata_csv"], data_cfg["groundtruth_csv"]
    )

    # Split
    split_path = os.path.join(config["logging"]["save_dir"], "split_indices.json")
    train_idx, val_idx = get_split_indices(
        df, data_cfg["val_split"], data_cfg["seed"], save_path=split_path
    )

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Transforms
    train_transform = get_transforms(data_cfg["image_size"], is_training=True)
    val_transform = get_transforms(data_cfg["image_size"], is_training=False)

    # Datasets
    train_dataset = MILK10kDataset(train_df, data_cfg["images_dir"], train_transform)
    val_dataset = MILK10kDataset(val_df, data_cfg["images_dir"], val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, df
