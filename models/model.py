"""
model.py — CNN classifier built on top of a pretrained EfficientNet-B0 backbone.

Architecture:
    EfficientNet-B0 (ImageNet pretrained)
        → Global Average Pooling  [built into backbone]
        → Dropout(0.3)
        → FC(1280 → 512) + BatchNorm + ReLU
        → Dropout(0.2)
        → FC(512 → num_classes)

We use EfficientNet-B0 for its excellent accuracy/parameter trade-off.
The classifier head is trained from scratch while the backbone is
progressively unfrozen (see train.py).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from typing import Optional


NUM_CLASSES = 5  # harsh, soft, backlit, low_light, mixed


class LightingClassifier(nn.Module):
    """
    Transfer-learning CNN for lighting condition classification.

    Args:
        num_classes:    Number of output classes (default 5).
        dropout:        Dropout rate before first FC layer.
        freeze_backbone: If True, freeze all backbone weights initially.
                         Use unfreeze_backbone() for fine-tuning.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3,
                 freeze_backbone: bool = True):
        super().__init__()

        # ── Backbone (EfficientNet-B0 pretrained on ImageNet) ──────────────
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features          # Conv stem + MBConv blocks
        self.avgpool  = backbone.avgpool           # Adaptive avg pool → 1x1

        backbone_out_dim = backbone.classifier[1].in_features  # 1280

        # ── Custom classifier head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        # Proper weight init for our custom head
        self._init_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-dim feature embeddings (before classifier head)."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters. Train head only."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — training head only.")

    def unfreeze_backbone(self, layers_from_end: Optional[int] = None) -> None:
        """
        Unfreeze backbone for fine-tuning.

        Args:
            layers_from_end: If None, unfreeze all. Otherwise unfreeze only
                             the last N top-level feature blocks.
        """
        if layers_from_end is None:
            for param in self.features.parameters():
                param.requires_grad = True
            print("[Model] Full backbone unfrozen.")
        else:
            # EfficientNet features has indices 0-8 (stem=0, blocks=1-7, head=8)
            total = len(list(self.features.children()))
            for i, block in enumerate(self.features.children()):
                if i >= total - layers_from_end:
                    for param in block.parameters():
                        param.requires_grad = True
            print(f"[Model] Last {layers_from_end} backbone blocks unfrozen.")

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(num_classes: int = NUM_CLASSES,
                freeze_backbone: bool = True,
                device: Optional[str] = None) -> LightingClassifier:
    """Convenience builder that moves the model to the right device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightingClassifier(num_classes=num_classes,
                                freeze_backbone=freeze_backbone)
    model = model.to(device)
    params = model.count_parameters()
    print(f"[Model] Built LightingClassifier on {device}")
    print(f"        Total params:     {params['total']:,}")
    print(f"        Trainable params: {params['trainable']:,}")
    return model
