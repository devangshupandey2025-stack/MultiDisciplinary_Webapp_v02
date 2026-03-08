"""
Knowledge Distillation: Train a lightweight student (MobileNetV3) from ensemble teacher.

Usage:
  python ml_pipeline/scripts/distill.py --teacher_dir checkpoints --student_config ml_pipeline/configs/mobilenet_v3.yaml --data_dir data/plantvillage
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.data.dataset import create_dataloaders
from ml_pipeline.models.architectures import create_model


class DistillationLoss(nn.Module):
    """Combined distillation loss: KL divergence + CE loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (KD loss)
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)

        # Hard targets (CE loss)
        ce_loss = self.ce_loss(student_logits, labels)

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


@torch.no_grad()
def get_teacher_logits(teacher_models, images, device, weights=None):
    """Get weighted ensemble teacher logits."""
    all_logits = []
    for model in teacher_models:
        logits = model(images)
        all_logits.append(logits)

    if weights is None:
        weights = [1.0 / len(teacher_models)] * len(teacher_models)

    teacher_logits = sum(w * l for w, l in zip(weights, all_logits))
    return teacher_logits


def distill(cfg: dict, teacher_dir: str, data_dir: str, output_dir: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = cfg['training']['seed']
    torch.manual_seed(seed)

    # Load teacher models
    teacher_models = []
    teacher_weights = []
    for d in sorted(os.listdir(teacher_dir)):
        ckpt_path = os.path.join(teacher_dir, d, "best_model.pt")
        if os.path.exists(ckpt_path) and d != "ensemble":
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = create_model(ckpt['model_type'], num_classes=ckpt['num_classes'],
                                 pretrained=False, dropout=0.0)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            teacher_models.append(model)
            teacher_weights.append(ckpt.get('val_f1', 0.5))
            print(f"Loaded teacher: {d} (F1={ckpt.get('val_f1', 0):.4f})")

    # Normalize weights
    total = sum(teacher_weights)
    teacher_weights = [w / total for w in teacher_weights]
    print(f"Teacher weights: {teacher_weights}")

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, img_size=cfg['data']['img_size'],
        batch_size=cfg['data']['batch_size'],
        augment_level=cfg['data']['augment_level'],
        num_workers=cfg['data']['num_workers'],
        use_balanced_sampling=True
    )

    # Student model
    student = create_model(
        cfg['model']['type'], num_classes=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'], dropout=cfg['model']['dropout']
    ).to(device)

    # Distillation loss
    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    optimizer = optim.AdamW(student.parameters(), lr=cfg['training']['learning_rate'],
                            weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    scaler = GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(output_dir, exist_ok=True)
    best_f1 = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        student.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Distill Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)

            teacher_logits = get_teacher_logits(teacher_models, images, device, teacher_weights)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                student_logits = student(images)
                loss = criterion(student_logits, teacher_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = student(images)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch, "model_state_dict": student.state_dict(),
                "val_f1": val_f1, "val_acc": val_acc,
                "model_type": cfg['model']['type'],
                "num_classes": cfg['model']['num_classes'],
                "config": cfg, "distilled": True,
            }, os.path.join(output_dir, "distilled_student.pt"))

    print(f"\nDistillation complete. Best student F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument("--teacher_dir", type=str, default="checkpoints")
    parser.add_argument("--student_config", type=str,
                        default="ml_pipeline/configs/mobilenet_v3.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/distilled")
    args = parser.parse_args()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)

    distill(cfg, args.teacher_dir, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
