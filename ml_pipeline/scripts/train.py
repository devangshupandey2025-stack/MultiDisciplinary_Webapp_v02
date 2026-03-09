"""
Training script for individual base models.
Supports: AdamW, cosine/onecycle LR, label smoothing, focal loss,
MixUp/CutMix, early stopping, checkpoint saving, logit extraction.

Usage:
  python ml_pipeline/scripts/train.py --config ml_pipeline/configs/efficientnet_v2.yaml --data_dir data/plantvillage

  # Multi-GPU
  torchrun --nproc_per_node=2 ml_pipeline/scripts/train.py --config ml_pipeline/configs/efficientnet_v2.yaml --data_dir data/plantvillage
"""
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.data.dataset import create_dataloaders, MixUpCutMix
from ml_pipeline.models.architectures import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, num_classes=38):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        if targets.dim() > 1:  # one-hot (from MixUp/CutMix)
            ce = -targets * torch.log_softmax(logits, dim=1)
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma * ce).sum(dim=1).mean()
            return focal

        ce = nn.functional.cross_entropy(logits, targets, reduction='none',
                                         label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


class EarlyStopping:
    """Early stopping based on validation metric."""
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0

    def step(self, metric) -> bool:
        improved = (metric > self.best) if self.mode == 'max' else (metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, scaler, mixup_fn, device, epoch, cfg):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # MixUp/CutMix
        use_mixup = cfg['training'].get('mixup_alpha', 0) > 0 and mixup_fn is not None
        if use_mixup and np.random.random() < 0.5:
            images, labels_mixed = mixup_fn(images, labels)
            is_mixed = True
        else:
            is_mixed = False

        optimizer.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            if is_mixed:
                loss = criterion(logits, labels_mixed)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if cfg['training'].get('gradient_clip_val', 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_val'])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if not is_mixed:
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
    return avg_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_logits = [], [], []

    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    all_logits = np.concatenate(all_logits, axis=0)

    return avg_loss, f1, acc, all_logits, np.array(all_labels)


def train(cfg: dict, data_dir: str):
    # Setup
    seed = cfg['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {seed}")

    # Data
    train_loader, val_loader, _ = create_dataloaders(
        data_dir,
        img_size=cfg['data']['img_size'],
        batch_size=cfg['data']['batch_size'],
        augment_level=cfg['data']['augment_level'],
        num_workers=cfg['data']['num_workers'],
        use_balanced_sampling=cfg['data']['use_balanced_sampling']
    )

    # Model
    model = create_model(
        cfg['model']['type'],
        num_classes=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained'],
        dropout=cfg['model']['dropout']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {cfg['model']['type']}, Params: {num_params:.1f}M")

    # Loss
    if cfg['loss']['type'] == 'focal':
        criterion = FocalLoss(
            gamma=cfg['loss']['focal_gamma'],
            label_smoothing=cfg['training']['label_smoothing'],
            num_classes=cfg['model']['num_classes']
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg['training']['label_smoothing'])

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    # Scheduler
    if cfg['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs'] - cfg['training']['warmup_epochs']
        )
    else:  # onecycle
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg['training']['learning_rate'],
            epochs=cfg['training']['epochs'],
            steps_per_epoch=len(train_loader)
        )

    # Warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=cfg['training']['warmup_epochs']
    ) if cfg['training']['warmup_epochs'] > 0 else None

    scaler = GradScaler(enabled=torch.cuda.is_available())
    mixup_fn = MixUpCutMix(
        mixup_alpha=cfg['training']['mixup_alpha'],
        cutmix_prob=cfg['training']['cutmix_prob'],
        num_classes=cfg['model']['num_classes']
    )

    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stopping_patience'], mode='max'
    )

    # Checkpoint directory
    save_dir = Path(cfg['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    best_f1 = 0
    history = []

    for epoch in range(1, cfg['training']['epochs'] + 1):
        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, mixup_fn, device, epoch, cfg
        )

        # Validate
        val_loss, val_f1, val_acc, val_logits, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        if warmup_scheduler and epoch <= cfg['training']['warmup_epochs']:
            warmup_scheduler.step()
        elif cfg['training']['scheduler'] == 'cosine':
            scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f} lr={lr:.6f}")

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("f1", {"train": train_f1, "val": val_f1}, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("lr", lr, epoch)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_f1": train_f1,
            "val_loss": val_loss, "val_f1": val_f1, "val_acc": val_acc, "lr": lr
        })

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1, "val_acc": val_acc, "val_loss": val_loss,
                "config": cfg, "num_classes": cfg['model']['num_classes'],
                "model_type": cfg['model']['type'],
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            np.save(save_dir / "best_val_logits.npy", val_logits)
            np.save(save_dir / "best_val_labels.npy", val_labels)
            print(f"  → Saved best model (F1={val_f1:.4f})")

        # Early stopping
        if early_stopping.step(val_f1):
            print(f"Early stopping at epoch {epoch}")
            break

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {save_dir}")
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Train a base model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to split dataset")
    parser.add_argument("--override", nargs="*", help="Override config values (key=value)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if args.override:
        for ov in args.override:
            keys, val = ov.split("=", 1)
            keys = keys.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            # Try to parse as int/float/bool
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() in ('true', 'false'):
                        val = val.lower() == 'true'
            d[keys[-1]] = val

    train(cfg, args.data_dir)


if __name__ == "__main__":
    main()
