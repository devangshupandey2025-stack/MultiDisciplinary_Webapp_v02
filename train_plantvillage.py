"""
Train MobileNetV3 on real PlantVillage dataset (38 classes).
Uses frozen pretrained backbone + trainable classifier head for fast CPU training.
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import f1_score
from pathlib import Path
from collections import Counter

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config
DATA_DIR = "data/plantvillage dataset/color"
CHECKPOINT_DIR = "checkpoints/mobilenet_v3"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 3e-3
IMG_SIZE = 224
NUM_WORKERS = 0  # Windows compatibility


class PlantVillageDataset(Dataset):
    """Load PlantVillage images from directory structure."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def create_splits(data_dir, val_ratio=0.15, test_ratio=0.1):
    """Create stratified train/val/test splits."""
    classes = sorted(os.listdir(data_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]

    all_samples = []
    for ci, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            all_samples.append((os.path.join(cls_dir, f), ci))

    # Stratified split
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, label in all_samples:
        by_class[label].append((path, label))

    train, val, test = [], [], []
    for label, samples in by_class.items():
        random.shuffle(samples)
        n = len(samples)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.extend(samples[:n_test])
        val.extend(samples[n_test:n_test + n_val])
        train.extend(samples[n_test + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val, test, classes


def main():
    print("=" * 60)
    print("PlantVillage Training — MobileNetV3 (Frozen Backbone)")
    print("=" * 60)

    # Create splits
    print("\n1. Creating data splits...")
    train_samples, val_samples, test_samples, class_names = create_splits(DATA_DIR)
    print(f"   Classes: {len(class_names)}")
    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = PlantVillageDataset(train_samples, train_transform)
    val_ds = PlantVillageDataset(val_samples, val_transform)
    test_ds = PlantVillageDataset(test_samples, val_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    print("\n2. Building model...")
    num_classes = len(class_names)
    backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    with torch.no_grad():
        feat_dim = backbone(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
    print(f"   Backbone: mobilenetv3_large_100, Feature dim: {feat_dim}")

    class PlantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            with torch.no_grad():
                feat = self.backbone(x)
            return self.head(feat)

    model = PlantModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Class weights for imbalanced data
    label_counts = Counter(label for _, label in train_samples)
    weights = [1.0 / label_counts[i] for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes

    # Training setup
    optimizer = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_dl), pct_start=0.1
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_f1 = 0
    history = []

    print(f"\n3. Training for {NUM_EPOCHS} epochs...")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for bi, (xb, yb) in enumerate(train_dl):
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = logits.argmax(1)
            train_correct += (preds == yb).sum().item()
            train_total += len(yb)

            if (bi + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Batch {bi+1}/{len(train_dl)} | "
                      f"Loss: {train_loss/(bi+1):.4f} | Acc: {train_correct/train_total:.3f}")

        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                preds = logits.argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += len(yb)
                all_preds.extend(preds.numpy())
                all_labels.extend(yb.numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        elapsed = time.time() - t0

        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_dl),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_dl),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'time': elapsed
        }
        history.append(record)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            marker = " ★ BEST"
            torch.save({
                'model_state_dict': model.state_dict(),
                'backbone_name': 'mobilenetv3_large_100',
                'num_classes': num_classes,
                'class_names': class_names,
                'feat_dim': feat_dim,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'epoch': epoch + 1,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {record['train_loss']:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {record['val_loss']:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | "
              f"{elapsed:.0f}s{marker}")

    print("-" * 80)
    print(f"\nBest Val F1: {best_f1:.4f}")

    # Save history
    with open(os.path.join(CHECKPOINT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save class names
    with open('checkpoints/class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)

    # Test evaluation
    print("\n4. Evaluating on test set...")
    model_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt'), 
                            map_location='cpu', weights_only=False)
    model.load_state_dict(model_ckpt['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb)
            preds = logits.argmax(1)
            test_correct += (preds == yb).sum().item()
            test_total += len(yb)
            test_preds.extend(preds.numpy())
            test_labels.extend(yb.numpy())

    test_acc = test_correct / test_total
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Macro F1: {test_f1:.4f}")

    # Per-class F1
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    print("\n   Per-class F1:")
    for i, (cls, f1) in enumerate(zip(class_names, per_class_f1)):
        print(f"   {cls:50s} F1={f1:.3f}")

    low_f1 = [(cls, f1) for cls, f1 in zip(class_names, per_class_f1) if f1 < 0.85]
    if low_f1:
        print(f"\n   ⚠ {len(low_f1)} classes with F1 < 0.85:")
        for cls, f1 in low_f1:
            print(f"     {cls}: {f1:.3f}")

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
