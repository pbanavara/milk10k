# MILK10k Concept Bottleneck Model (CBM) — Design Document

## Objective

Build a Concept Bottleneck Model for skin lesion classification on the MILK10k dataset. The model takes paired clinical close-up + dermoscopic images per lesion, predicts 7 MONET dermoscopic concepts as an intermediate bottleneck layer, then predicts 11 diagnostic categories from those concepts. This architecture provides inherently interpretable predictions — every diagnosis is traceable to clinically meaningful concepts.

## Project Structure

```
milk10k-cbm/
├── data/
│   ├── images_train/          # 10,480 JPEG images (pre-downloaded)
│   ├── MILK10k_Training_Metadata.csv
│   ├── MILK10k_Training_GroundTruth.csv
│   └── MILK10k_Training_Supplement.csv
├── src/
│   ├── dataset.py             # Dataset class and data loading
│   ├── model.py               # CBM architecture
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation and fairness analysis
│   └── utils.py               # Helpers, logging, config
├── configs/
│   └── default.yaml           # Hyperparameters
├── outputs/                   # Checkpoints, logs, predictions
├── requirements.txt
└── README.md
```

## Data Pipeline (`dataset.py`)

### Data Sources

Three CSVs must be joined:

1. **MILK10k_Training_Metadata.csv** (10,480 rows — one per image)
   - Key columns: `lesion_id`, `isic_id`, `image_type`, `age_approx`, `sex`, `skin_tone_class`, `site`
   - MONET concept columns (7 floats, 0-1): `MONET_ulceration_crust`, `MONET_hair`, `MONET_vasculature_vessels`, `MONET_erythema`, `MONET_pigmented`, `MONET_gel_water_drop_fluid_dermoscopy_liquid`, `MONET_skin_markings_pen_ink_purple_pen`
   - `image_type` is either `"clinical: close-up"` or `"dermoscopic"`

2. **MILK10k_Training_GroundTruth.csv** (5,240 rows — one per lesion)
   - Key columns: `lesion_id` + 11 binary one-hot diagnosis columns: `AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`

3. **MILK10k_Training_Supplement.csv** (10,480 rows)
   - Key columns: `isic_id`, `diagnosis_full`, `diagnosis_confirm_type`, `invasion_thickness_interval`
   - Used for analysis, not directly in training

### Dataset Class: `MILK10kDataset`

Each sample represents one **lesion** (not one image). Each sample returns:

```python
{
    "lesion_id": str,
    "clinical_image": Tensor,       # [3, H, W] — clinical close-up photo
    "dermoscopic_image": Tensor,    # [3, H, W] — dermoscopic photo
    "concepts": Tensor,             # [7] — MONET soft labels (float32, 0-1)
    "diagnosis": Tensor,            # [11] — one-hot diagnosis labels (float32)
    "skin_tone": int,               # 0-5 for fairness stratification
    "metadata": {                   # optional, for analysis
        "age_approx": int,
        "sex": str,
        "site": str,
    }
}
```

### Data Join Logic

1. Pivot metadata CSV by `image_type` on `lesion_id` to get one row per lesion with columns for both clinical and dermoscopic `isic_id`
2. MONET concepts exist per-image (10,480 rows). For the concept bottleneck, use the **dermoscopic** image's MONET scores as supervision targets (MONET was designed for dermoscopic analysis). Store clinical MONET scores separately for ablation.
3. Join ground truth on `lesion_id`
4. Image filenames follow the pattern: `{isic_id}.jpg` in the images directory

### Data Split

- Use **patient-level stratified split** (80/20 train/val) to prevent data leakage
- Since we don't have explicit patient IDs, use `lesion_id` as proxy (each lesion = unique patient encounter)
- Stratify by diagnosis to maintain class distribution
- Use `sklearn.model_selection.StratifiedGroupKFold` or manual stratified split
- Store split indices in a JSON for reproducibility

### Transforms

- Training: Resize(224) → RandomHorizontalFlip → RandomVerticalFlip → RandomRotation(20) → ColorJitter(brightness=0.2, contrast=0.2) → Normalize(ImageNet stats)
- Validation: Resize(224) → CenterCrop(224) → Normalize(ImageNet stats)
- Use same transform for both clinical and dermoscopic images independently

### DataLoader

- Batch size: 32 (adjust based on GPU memory)
- Num workers: 4
- Pin memory: True
- Drop last: True for training

## Model Architecture (`model.py`)

### Overview

```
Clinical Image ──→ DINOv2 Encoder ──→ [CLS] token ──┐
                                                      ├──→ Concat [2048] ──→ Concept Head [7] ──→ Classification Head [11]
Dermoscopic Image → DINOv2 Encoder ──→ [CLS] token ──┘
```

Both images share the **same** DINOv2 encoder (weight sharing). This is important — it halves parameters and the encoder learns modality-invariant features.

### Components

#### 1. Backbone: DINOv2 ViT-L/14

```python
# Use torch.hub or timm
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# Output: [batch, 1024] CLS token
```

- Freeze initially, unfreeze later for fine-tuning
- Output dimension: 1024 (ViT-L)

#### 2. Fusion Layer

- Concatenate CLS tokens from clinical and dermoscopic passes: [batch, 2048]
- Single hidden layer MLP: Linear(2048, 512) → ReLU → Dropout(0.3) → output [batch, 512]

#### 3. Concept Head (the bottleneck)

```python
concept_head = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 7),
    nn.Sigmoid()  # Output in [0, 1] to match MONET probability targets
)
```

- Output: [batch, 7] — predicted concept probabilities
- Supervised with MONET soft labels

#### 4. Classification Head

```python
classification_head = nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 11)
)
```

- Input: **only** the 7 predicted concepts (this is the bottleneck constraint)
- Output: [batch, 11] — raw logits for 11 diagnosis categories
- No sigmoid here — use BCEWithLogitsLoss or CrossEntropyLoss

### Model Variants to Implement

1. **Strict CBM**: Classification sees only 7 concepts (fully interpretable)
2. **Hybrid CBM**: Classification sees 7 concepts + a small residual vector from the fusion layer (higher accuracy, partially interpretable). Add a residual path: Linear(512, 16) that bypasses the bottleneck and is concatenated with concept predictions before classification.
3. **Image-only baseline**: Standard DINOv2 → fusion → 11-class head (no concept bottleneck). For comparison.

Use a config flag to switch between variants.

## Training (`train.py`)

### Loss Function

Joint loss with tunable weighting:

```python
L_total = alpha * L_concept + beta * L_classification

# Concept loss: MSE (soft regression targets, continuous 0-1)
L_concept = MSE(predicted_concepts, monet_targets)

# Classification loss: Cross-entropy
# Convert one-hot ground truth to class indices for CE, or use BCEWithLogitsLoss for multi-label
L_classification = CrossEntropyLoss(logits, class_indices)
```

- Start with `alpha=1.0, beta=1.0`
- The ground truth CSV has one-hot labels. Check if any lesion has multiple 1s (multi-label) — if all are single-label, use CrossEntropyLoss with argmax indices. If multi-label, use BCEWithLogitsLoss.

### Training Schedule

**Phase 1 — Frozen backbone (5-10 epochs)**
- Freeze DINOv2 entirely
- Train only: fusion layer, concept head, classification head
- Learning rate: 1e-3 with AdamW
- Purpose: warm up the heads before fine-tuning

**Phase 2 — Full fine-tuning (20-30 epochs)**
- Unfreeze DINOv2 backbone
- Differential learning rates:
  - Backbone: 1e-5
  - Fusion + heads: 1e-4
- Cosine annealing scheduler with warmup (5% of steps)
- Weight decay: 0.05

### Training Details

- Mixed precision (torch.cuda.amp) — bf16 on A100
- Gradient clipping: max_norm=1.0
- Early stopping: patience=7 on validation macro F1
- Save best checkpoint by validation macro F1

### Logging

- Use wandb or tensorboard
- Log per-epoch: train/val loss (total, concept, classification separately), macro F1, per-class F1, concept MSE per concept
- Log per-checkpoint: concept correlation plots (predicted vs actual for each MONET concept)

## Evaluation (`evaluate.py`)

### Primary Metrics (match MILK10k benchmark)

- **Macro F1 Score** at threshold 0.5 — this is the leaderboard metric
- Per-class F1 scores for all 11 categories
- Confusion matrix

### Concept Quality Metrics

- Per-concept MSE and Pearson correlation (predicted vs MONET ground truth)
- Scatter plots: predicted vs actual for each of the 7 concepts
- This validates the bottleneck is actually learning meaningful concepts

### Fairness Analysis (by `skin_tone_class`)

- Macro F1 stratified by skin tone groups: {0-1, 2, 3, 4-5} (bin 0-1 together due to n=12 for tone 0)
- Per-class performance gaps across skin tone groups
- Concept prediction accuracy stratified by skin tone (do concepts degrade on darker skin?)

### Ablation Comparisons

Generate a results table comparing:
1. Strict CBM (concepts only)
2. Hybrid CBM (concepts + residual)
3. Image-only baseline (no concepts)
4. With vs without clinical metadata (age, sex, site as additional features)

### Benchmark Submission

- Generate predictions on MILK10k Benchmark test images (958 images, 479 lesions)
- Output CSV format: `lesion_id, AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC`
- Submit to https://challenge.isic-archive.com/task/57/

## Configuration (`configs/default.yaml`)

```yaml
data:
  images_dir: "./data/images_train"
  metadata_csv: "./data/MILK10k_Training_Metadata.csv"
  groundtruth_csv: "./data/MILK10k_Training_GroundTruth.csv"
  supplement_csv: "./data/MILK10k_Training_Supplement.csv"
  image_size: 224
  batch_size: 32
  num_workers: 4
  val_split: 0.2
  seed: 42

model:
  backbone: "dinov2_vitl14"        # Options: dinov2_vitl14, dinov2_vitb14, dinov2_vits14
  backbone_dim: 1024               # 1024 for ViT-L, 768 for ViT-B, 384 for ViT-S
  fusion_dim: 512
  num_concepts: 7
  num_classes: 11
  variant: "strict"                # Options: "strict", "hybrid", "baseline"
  residual_dim: 16                 # Only used in hybrid variant
  dropout: 0.3

training:
  phase1_epochs: 10
  phase2_epochs: 30
  phase1_lr: 1e-3
  phase2_backbone_lr: 1e-5
  phase2_head_lr: 1e-4
  weight_decay: 0.05
  concept_loss_weight: 1.0        # alpha
  classification_loss_weight: 1.0  # beta
  grad_clip_norm: 1.0
  early_stopping_patience: 7
  use_amp: true

logging:
  project_name: "milk10k-cbm"
  save_dir: "./outputs"
  log_interval: 10                # steps between logging
```

## Dependencies (`requirements.txt`)

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
pyyaml>=6.0
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## Implementation Order

1. **`dataset.py`** — Data joining, Dataset class, transforms, split logic. Test with a simple print loop to verify image loading and label alignment.
2. **`model.py`** — All three model variants. Test with random tensors to verify shapes.
3. **`train.py`** — Phase 1 and Phase 2 training with logging. Run a quick overfit test on 10 samples.
4. **`evaluate.py`** — Metrics, fairness analysis, ablation table generation.
5. **`utils.py`** — Config loading, checkpoint saving/loading, visualization helpers.

## Key Constraints

- Every prediction MUST pass through the 7-concept bottleneck in strict mode (no shortcut connections)
- Use the same DINOv2 encoder for both image types (weight sharing)
- MONET concept targets are soft probabilities — use MSE loss, not cross-entropy
- Patient-level splits only — no lesion appearing in both train and val
- All evaluation must be stratifiable by `skin_tone_class`
- Code should be clean, well-documented, and runnable from CLI with config overrides