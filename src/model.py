"""Concept Bottleneck Model architecture for MILK10k skin lesion classification.

Three variants:
  - strict: Classification head sees only 7 concept predictions (fully interpretable)
  - hybrid: Classification head sees 7 concepts + residual vector (partially interpretable)
  - baseline: Direct image-to-diagnosis, no concept bottleneck (for comparison)
"""

import torch
import torch.nn as nn


class ConceptBottleneckModel(nn.Module):
    """CBM with shared DINOv2 backbone for paired clinical + dermoscopic images."""

    def __init__(
        self,
        backbone_name: str = "dinov2_vitl14",
        backbone_dim: int = 1024,
        fusion_dim: int = 512,
        num_concepts: int = 7,
        num_classes: int = 11,
        variant: str = "strict",
        residual_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.variant = variant
        self.backbone_dim = backbone_dim
        self.num_concepts = num_concepts

        # Shared DINOv2 backbone
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        # Freeze backbone by default
        self.freeze_backbone()

        # Fusion MLP: concat of two CLS tokens → fusion_dim
        self.fusion = nn.Sequential(
            nn.Linear(backbone_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if variant in ("strict", "hybrid"):
            # Concept head (the bottleneck)
            self.concept_head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_concepts),
                nn.Sigmoid(),
            )

            if variant == "hybrid":
                # Residual path bypassing the bottleneck
                self.residual_head = nn.Sequential(
                    nn.Linear(fusion_dim, residual_dim),
                    nn.Dropout(dropout),
                )
                cls_input_dim = num_concepts + residual_dim
            else:
                self.residual_head = None
                cls_input_dim = num_concepts

            # Classification head
            self.classification_head = nn.Sequential(
                nn.Linear(cls_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        elif variant == "baseline":
            # No concept bottleneck — direct classification
            self.concept_head = None
            self.residual_head = None
            self.classification_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        clinical_image: torch.Tensor,
        dermoscopic_image: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            clinical_image: [B, 3, H, W] clinical close-up image.
            dermoscopic_image: [B, 3, H, W] dermoscopic image.

        Returns:
            Dict with keys:
              - "logits": [B, 11] classification logits
              - "concepts": [B, 7] predicted concepts (None for baseline)
        """
        # Shared backbone: extract CLS tokens
        cls_clinical = self.backbone(clinical_image)      # [B, backbone_dim]
        cls_dermoscopic = self.backbone(dermoscopic_image)  # [B, backbone_dim]

        # Fuse
        fused = self.fusion(
            torch.cat([cls_clinical, cls_dermoscopic], dim=1)
        )  # [B, fusion_dim]

        if self.variant == "baseline":
            logits = self.classification_head(fused)
            return {"logits": logits, "concepts": None}

        # Concept prediction
        concepts = self.concept_head(fused)  # [B, 7]

        if self.variant == "hybrid":
            residual = self.residual_head(fused)  # [B, residual_dim]
            cls_input = torch.cat([concepts, residual], dim=1)
        else:
            cls_input = concepts

        logits = self.classification_head(cls_input)  # [B, 11]

        return {"logits": logits, "concepts": concepts}


def get_model(config: dict) -> ConceptBottleneckModel:
    """Factory function to build CBM from config."""
    model_cfg = config["model"]
    return ConceptBottleneckModel(
        backbone_name=model_cfg["backbone"],
        backbone_dim=model_cfg["backbone_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        num_concepts=model_cfg["num_concepts"],
        num_classes=model_cfg["num_classes"],
        variant=model_cfg["variant"],
        residual_dim=model_cfg["residual_dim"],
        dropout=model_cfg["dropout"],
    )
