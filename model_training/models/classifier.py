"""
Lightweight classifier built on timm backbones.
Supports MobileNetV3, EfficientNet, etc. for mobile deployment.
"""
import timm
import torch
import torch.nn as nn


MOBILE_BACKBONES = {
    'mobilenetv3_small': 'mobilenetv3_small_100',
    'mobilenetv3_large': 'mobilenetv3_large_100',
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_lite0': 'efficientnet_lite0',
    'mnasnet': 'mnasnet_100',
}


class SmallSampleClassifier(nn.Module):
    """
    Classification model with:
    - Pretrained backbone (transfer learning)
    - Optional backbone freezing for very small datasets
    - Dropout for regularization
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        backbone_name = MOBILE_BACKBONES.get(backbone, backbone)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def unfreeze_backbone(self):
        """Gradually unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(cfg: dict) -> SmallSampleClassifier:
    """Build model from config dict."""
    model_cfg = cfg['model']
    return SmallSampleClassifier(
        backbone=model_cfg['backbone'],
        num_classes=model_cfg['num_classes'],
        pretrained=model_cfg.get('pretrained', True),
        freeze_backbone=model_cfg.get('freeze_backbone', False),
    )
