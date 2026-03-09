"""
ML Inference Engine — MobileNetV3 plant disease classifier.
Returns probabilities with uncertainty estimates and image metadata for LLM validation.
"""
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageStat
from torchvision import transforms
import timm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class FlexibleModel(nn.Module):
    """MobileNetV3 with frozen backbone and trainable classification head."""
    def __init__(self, backbone_name: str, num_classes: int, feat_dim: int = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        if feat_dim is None:
            with torch.no_grad():
                feat_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


class PlantDiseasePredictor:
    """
    Production inference engine using MobileNetV3 for plant disease classification.
    Extracts image metadata for LLM cross-validation.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.transform = None
        self._loaded = False

    def _load_model_from_checkpoint(self, ckpt):
        """Load MobileNetV3 from checkpoint."""
        num_classes = ckpt['num_classes']

        if 'backbone_name' in ckpt:
            model = FlexibleModel(
                ckpt['backbone_name'],
                num_classes,
                feat_dim=ckpt.get('feat_dim')
            )
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model_type' in ckpt:
            try:
                from ml_pipeline.models.architectures import create_model
                model = create_model(
                    ckpt['model_type'],
                    num_classes=num_classes,
                    pretrained=False, dropout=0.0
                )
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                raise ValueError(f"Cannot load model_type={ckpt.get('model_type')}")
        else:
            raise ValueError("Unknown checkpoint format")

        return model

    def load(self, models_dir: str = "checkpoints"):
        """Load MobileNetV3 model from checkpoint directory."""
        # Load class names
        class_names_path = os.path.join(models_dir, "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path) as f:
                self.class_names = json.load(f)
                self.num_classes = len(self.class_names)
                print(f"Loaded {self.num_classes} class names")

        # Load MobileNetV3 checkpoint
        ckpt_path = os.path.join(models_dir, "mobilenet_v3", "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"MobileNetV3 checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = self._load_model_from_checkpoint(ckpt)
        self.model.to(self.device).eval()

        if not self.class_names and 'class_names' in ckpt:
            self.class_names = ckpt['class_names']
            self.num_classes = len(self.class_names)

        if not self.num_classes:
            self.num_classes = ckpt['num_classes']

        # Fallback class names
        if not self.class_names:
            try:
                from ml_pipeline.data.dataset import PLANTVILLAGE_CLASSES
                self.class_names = PLANTVILLAGE_CLASSES
                self.num_classes = len(self.class_names)
            except ImportError:
                self.class_names = [f"class_{i}" for i in range(self.num_classes or 38)]
                self.num_classes = len(self.class_names)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._loaded = True
        val_f1 = ckpt.get('val_f1', 'N/A')
        print(f"MobileNetV3 ready: {self.num_classes} classes on {self.device} (val_f1={val_f1})")

    def extract_image_metadata(self, image: Image.Image) -> dict:
        """Extract image metadata for LLM validation context."""
        rgb = image.convert('RGB')
        stat = ImageStat.Stat(rgb)
        r_mean, g_mean, b_mean = stat.mean
        brightness = sum(stat.mean) / 3.0 / 255.0
        contrast = sum(stat.stddev) / 3.0 / 255.0

        # Dominant color channel
        channels = {'red': r_mean, 'green': g_mean, 'blue': b_mean}
        dominant = max(channels, key=channels.get)

        # Green ratio (useful for plant/leaf detection)
        total = r_mean + g_mean + b_mean
        green_ratio = g_mean / total if total > 0 else 0

        return {
            "width": image.width,
            "height": image.height,
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "dominant_color": dominant,
            "green_ratio": round(green_ratio, 3),
            "r_mean": round(r_mean, 1),
            "g_mean": round(g_mean, 1),
            "b_mean": round(b_mean, 1),
        }

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """
        Run MobileNetV3 prediction on a single image.
        Returns prediction with class, probability, uncertainty, top-k, and image metadata.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tensor = self.preprocess(image)
        logits = self.model(tensor).cpu().numpy()
        probs = self._softmax(logits)[0]

        # Entropy-based uncertainty
        pred_entropy = -(probs * np.log(probs + 1e-10)).sum()
        max_entropy = np.log(len(probs))
        uncertainty = float(pred_entropy / max_entropy)

        # Top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_predictions = []
        for idx in top_k_indices:
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            top_k_predictions.append({
                "class": class_name,
                "probability": float(probs[idx]),
            })

        best_idx = top_k_indices[0]
        image_metadata = self.extract_image_metadata(image)

        result = {
            "class": self.class_names[best_idx] if best_idx < len(self.class_names) else f"class_{best_idx}",
            "probability": float(probs[best_idx]),
            "uncertainty": round(uncertainty, 4),
            "top_k": top_k_predictions,
            "image_metadata": image_metadata,
        }

        return result

    @staticmethod
    def _softmax(logits):
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


# Global predictor instance
_predictor: Optional[PlantDiseasePredictor] = None


def get_predictor() -> PlantDiseasePredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PlantDiseasePredictor()
    return _predictor
