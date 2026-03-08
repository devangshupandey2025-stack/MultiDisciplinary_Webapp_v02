"""
Base model architectures for plant disease detection ensemble.
5 diverse architectures for maximum ensemble strength:
1. EfficientNetV2-S  — compound-scaled CNN (accuracy)
2. ResNet50          — residual connections (robustness)
3. ConvNeXt-Tiny     — modernized CNN (accuracy)
4. Swin-Tiny         — shifted window ViT (attention)
5. MobileNetV3-Large — lightweight (edge deployment)
"""
import torch
import torch.nn as nn
import timm
from typing import Optional


class BaseModel(nn.Module):
    """Base wrapper for all models with consistent interface."""

    def __init__(self, model_name: str, num_classes: int = 38,
                 pretrained: bool = True, dropout: float = 0.3,
                 feature_dim: Optional[int] = None):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_rate=dropout
        )

        # Auto-detect feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
            actual_dim = feat.shape[-1]

        if feature_dim is None:
            feature_dim = actual_dim

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head (for stacking)."""
        return self.backbone(x)

    def get_logits_and_features(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features


def create_efficientnet_v2(num_classes: int = 38, pretrained: bool = True, dropout: float = 0.3):
    """EfficientNetV2-S: Compound-scaled CNN with progressive training."""
    return BaseModel("tf_efficientnetv2_s", num_classes, pretrained, dropout)


def create_resnet50(num_classes: int = 38, pretrained: bool = True, dropout: float = 0.3):
    """ResNet50: Classic residual network, robust baseline."""
    return BaseModel("resnet50", num_classes, pretrained, dropout)


def create_convnext_tiny(num_classes: int = 38, pretrained: bool = True, dropout: float = 0.3):
    """ConvNeXt-Tiny: Modernized CNN competing with ViTs."""
    return BaseModel("convnext_tiny", num_classes, pretrained, dropout)


def create_swin_tiny(num_classes: int = 38, pretrained: bool = True, dropout: float = 0.3):
    """Swin-Tiny: Shifted window transformer for hierarchical features."""
    return BaseModel("swin_tiny_patch4_window7_224", num_classes, pretrained, dropout)


def create_mobilenet_v3(num_classes: int = 38, pretrained: bool = True, dropout: float = 0.2):
    """MobileNetV3-Large: Lightweight model for edge deployment."""
    return BaseModel("mobilenetv3_large_100", num_classes, pretrained, dropout)


# Model registry
MODEL_REGISTRY = {
    "efficientnet_v2": create_efficientnet_v2,
    "resnet50": create_resnet50,
    "convnext_tiny": create_convnext_tiny,
    "swin_tiny": create_swin_tiny,
    "mobilenet_v3": create_mobilenet_v3,
}


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create any registered model."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)


def get_model_info(model_type: str) -> dict:
    """Get model metadata."""
    info = {
        "efficientnet_v2": {"input_size": 384, "params_m": 21.5, "flops_g": 8.4, "profile": "server"},
        "resnet50": {"input_size": 224, "params_m": 25.6, "flops_g": 4.1, "profile": "server"},
        "convnext_tiny": {"input_size": 224, "params_m": 28.6, "flops_g": 4.5, "profile": "server"},
        "swin_tiny": {"input_size": 224, "params_m": 28.3, "flops_g": 4.5, "profile": "server"},
        "mobilenet_v3": {"input_size": 224, "params_m": 5.4, "flops_g": 0.22, "profile": "edge"},
    }
    return info.get(model_type, {})
