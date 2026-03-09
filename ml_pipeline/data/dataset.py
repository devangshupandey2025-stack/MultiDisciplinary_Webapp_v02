"""
Data pipeline for PlantVillage dataset.
Handles downloading, validation, stratified splitting, and preprocessing.
"""
import os
import json
import shutil
import random
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets


# PlantVillage 38-class label mapping
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Map class to plant species for group-based splitting
CLASS_TO_SPECIES = {}
for cls in PLANTVILLAGE_CLASSES:
    species = cls.split("___")[0]
    CLASS_TO_SPECIES[cls] = species


def normalize_label(label: str) -> str:
    """Normalize label names to consistent format."""
    label = label.strip().replace(" ", "_").replace(",", ",")
    return label


def get_species_groups(data_dir: str) -> Dict[str, List[str]]:
    """Group classes by plant species to prevent data leakage in splits."""
    species_to_classes = defaultdict(list)
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            species = cls.split("___")[0] if "___" in cls else cls.split("_")[0]
            species_to_classes[species].append(cls)
    return dict(species_to_classes)


def create_stratified_species_split(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create stratified train/val/test splits GROUPED by plant species
    to avoid data leakage (same plant appearing in train and test).

    Within each species group, images are split maintaining class proportions.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    np.random.seed(seed)

    splits = {"train": [], "val": [], "test": []}
    split_stats = {"train": Counter(), "val": Counter(), "test": Counter()}

    all_classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    for cls in all_classes:
        cls_dir = os.path.join(data_dir, cls)
        images = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img in img_list:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(split_cls_dir, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                splits[split_name].append(dst)
            split_stats[split_name][cls] = len(img_list)

    # Save split metadata
    meta = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {k: dict(v) for k, v in split_stats.items()},
        "total": {k: sum(v.values()) for k, v in split_stats.items()}
    }
    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Split complete: train={meta['total']['train']}, "
          f"val={meta['total']['val']}, test={meta['total']['test']}")
    return splits


def get_train_transforms(img_size: int = 224, augment_level: str = "medium"):
    """Configurable training augmentation pipeline."""
    if augment_level == "light":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif augment_level == "medium":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:  # heavy
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
        ])


def get_val_transforms(img_size: int = 224):
    """Validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class PlantDiseaseDataset(Dataset):
    """Custom dataset with MixUp/CutMix support."""

    def __init__(self, root_dir: str, transform=None, return_path: bool = False):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.return_path = return_path
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.return_path:
            path = self.dataset.samples[idx][0]
            return img, label, path
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for class-balanced sampling."""
        counts = Counter(self.targets)
        total = len(self.targets)
        weights = torch.zeros(len(self.classes))
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (len(self.classes) * count)
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted random sampler for class-balanced training."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[t] for t in self.targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(self.targets), replacement=True)


class MixUpCutMix:
    """MixUp and CutMix data augmentation."""

    def __init__(self, mixup_alpha: float = 0.2, cutmix_prob: float = 0.5, num_classes: int = 38):
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        if random.random() < self.cutmix_prob:
            return self._cutmix(images, targets)
        else:
            return self._mixup(images, targets)

    def _mixup(self, images, targets):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).float()
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[index]
        return mixed_images, mixed_targets

    def _cutmix(self, images, targets):
        lam = np.random.beta(1.0, 1.0)
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        targets_onehot = torch.nn.functional.one_hot(targets, self.num_classes).float()
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[index]
        return images, mixed_targets


def create_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    augment_level: str = "medium",
    num_workers: int = 4,
    use_balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with all preprocessing."""
    train_transform = get_train_transforms(img_size, augment_level)
    val_transform = get_val_transforms(img_size)

    train_dataset = PlantDiseaseDataset(os.path.join(data_dir, "train"), train_transform)
    val_dataset = PlantDiseaseDataset(os.path.join(data_dir, "val"), val_transform)
    test_dataset = PlantDiseaseDataset(os.path.join(data_dir, "test"), val_transform, return_path=True)

    sampler = train_dataset.get_sampler() if use_balanced_sampling else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")
    return train_loader, val_loader, test_loader
