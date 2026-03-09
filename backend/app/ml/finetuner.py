"""
Dynamic fine-tuning engine for PlantGuard AI.
Fine-tunes only the classification head (backbone stays frozen) using user feedback data.
Designed to run on CPU (head has ~500K params).
"""
import os
import io
import json
import time
import logging
import threading
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.getenv("MODELS_DIR", "checkpoints")
FINETUNE_DIR = os.path.join(CHECKPOINT_DIR, "mobilenet_v3", "finetuned")
IMG_SIZE = 224
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 5
FINETUNE_BATCH = 8
MIN_SAMPLES = 5


class FeedbackDataset(Dataset):
    """Dataset from user feedback images + labels."""

    def __init__(self, images: List[Image.Image], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class FineTuner:
    """Handles model fine-tuning from user feedback."""

    def __init__(self):
        self.is_training = False
        self.last_training: Optional[Dict] = None
        self.training_history: List[Dict] = []
        self._lock = threading.Lock()

    def finetune(self, predictor, feedback_data: List[Dict],
                 supabase_service=None) -> Dict:
        """
        Fine-tune the model head using feedback data.

        Args:
            predictor: PlantDiseasePredictor instance with loaded model
            feedback_data: List of dicts with 'image_url', 'actual_class'
            supabase_service: Optional SupabaseService for downloading images

        Returns:
            Training result dict with metrics
        """
        with self._lock:
            if self.is_training:
                return {"status": "error", "message": "Training already in progress"}
            self.is_training = True

        try:
            start_time = time.time()
            class_names = predictor.class_names
            class_to_idx = {name: i for i, name in enumerate(class_names)}

            # Download and prepare images
            images, labels = [], []
            skipped = 0
            for item in feedback_data:
                actual_class = item.get("actual_class", "")
                if actual_class not in class_to_idx:
                    skipped += 1
                    continue

                img = self._load_image(item, supabase_service)
                if img is None:
                    skipped += 1
                    continue

                images.append(img)
                labels.append(class_to_idx[actual_class])

            if len(images) < MIN_SAMPLES:
                return {
                    "status": "error",
                    "message": f"Need at least {MIN_SAMPLES} valid samples, got {len(images)}"
                }

            logger.info(f"Fine-tuning with {len(images)} samples ({skipped} skipped)")

            # Create dataset with augmentation
            train_dataset = FeedbackDataset(images, labels)
            train_loader = DataLoader(
                train_dataset, batch_size=FINETUNE_BATCH,
                shuffle=True, num_workers=0
            )

            # Fine-tune only the head
            model = predictor.model
            model.train()

            # Freeze backbone, unfreeze head
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True

            # Class weights for imbalanced feedback
            label_counts = np.bincount(labels, minlength=len(class_names))
            nonzero = label_counts > 0
            weights = np.zeros(len(class_names), dtype=np.float32)
            weights[nonzero] = len(labels) / (nonzero.sum() * label_counts[nonzero])
            class_weights = torch.FloatTensor(weights).to(predictor.device)

            optimizer = optim.AdamW(model.head.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

            epoch_metrics = []
            for epoch in range(FINETUNE_EPOCHS):
                total_loss, correct, total = 0, 0, 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(predictor.device), yb.to(predictor.device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += len(yb)

                acc = correct / total if total > 0 else 0
                epoch_metrics.append({
                    "epoch": epoch + 1,
                    "loss": round(total_loss / len(train_loader), 4),
                    "accuracy": round(acc, 4)
                })
                logger.info(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}: loss={epoch_metrics[-1]['loss']:.4f}, acc={acc:.2%}")

            # Save fine-tuned model
            model.eval()
            os.makedirs(FINETUNE_DIR, exist_ok=True)

            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(FINETUNE_DIR, f"model_v{version}.pt")
            best_path = os.path.join(CHECKPOINT_DIR, "mobilenet_v3", "best_model.pt")

            # Save versioned copy
            ckpt = {
                "model_state_dict": model.state_dict(),
                "backbone_name": "mobilenetv3_large_100",
                "num_classes": len(class_names),
                "class_names": class_names,
                "feat_dim": model.head[1].in_features,
                "finetuned_from": "user_feedback",
                "num_samples": len(images),
                "version": version,
            }
            torch.save(ckpt, save_path)

            # Overwrite best_model.pt for auto-loading on restart
            torch.save(ckpt, best_path)
            logger.info(f"Saved fine-tuned model: {save_path}")

            elapsed = time.time() - start_time
            result = {
                "status": "success",
                "version": version,
                "samples_used": len(images),
                "samples_skipped": skipped,
                "epochs": FINETUNE_EPOCHS,
                "final_loss": epoch_metrics[-1]["loss"],
                "final_accuracy": epoch_metrics[-1]["accuracy"],
                "training_time_seconds": round(elapsed, 1),
                "epoch_metrics": epoch_metrics,
            }

            self.last_training = result
            self.training_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        finally:
            with self._lock:
                self.is_training = False

    def _load_image(self, item: Dict, supabase_service=None) -> Optional[Image.Image]:
        """Load an image from URL or Supabase storage."""
        image_url = item.get("image_url", "")
        if not image_url:
            return None

        try:
            import httpx
            resp = httpx.get(image_url, timeout=15.0, follow_redirects=True)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_url}: {e}")
            return None


# Global instance
_finetuner: Optional[FineTuner] = None


def get_finetuner() -> FineTuner:
    global _finetuner
    if _finetuner is None:
        _finetuner = FineTuner()
    return _finetuner
