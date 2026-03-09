"""
Two-stage ensemble system:
  Stage A: Validation-weighted soft voting
  Stage B: Stacked meta-learner (LightGBM) on k-fold OOF logits

Usage:
  python ml_pipeline/scripts/stacking.py --models_dir checkpoints --data_dir data/plantvillage --n_folds 5
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.data.dataset import PlantDiseaseDataset, get_val_transforms
from ml_pipeline.models.architectures import create_model


def load_trained_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(
        ckpt['model_type'],
        num_classes=ckpt['num_classes'],
        pretrained=False,
        dropout=0.0  # No dropout during inference
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_logits(model, loader, device):
    """Extract logits and labels from a model on a dataset."""
    all_logits, all_labels = [], []
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1]
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def compute_image_features(images_loader):
    """Compute brightness/contrast features for meta-learner input."""
    features = []
    for batch in images_loader:
        imgs = batch[0]  # (B, C, H, W)
        # Per-image mean brightness, std, channel means
        brightness = imgs.mean(dim=(1, 2, 3)).numpy()
        contrast = imgs.std(dim=(1, 2, 3)).numpy()
        r_mean = imgs[:, 0].mean(dim=(1, 2)).numpy()
        g_mean = imgs[:, 1].mean(dim=(1, 2)).numpy()
        b_mean = imgs[:, 2].mean(dim=(1, 2)).numpy()
        batch_feats = np.stack([brightness, contrast, r_mean, g_mean, b_mean], axis=1)
        features.append(batch_feats)
    return np.concatenate(features)


class WeightedSoftVoting:
    """Stage A: Validation-weighted soft voting ensemble."""

    def __init__(self, model_weights: dict):
        """
        model_weights: {model_name: weight} based on validation F1
        """
        self.weights = model_weights
        total = sum(model_weights.values())
        self.norm_weights = {k: v / total for k, v in model_weights.items()}

    def predict(self, logits_dict: dict) -> np.ndarray:
        """
        logits_dict: {model_name: (N, C) logits array}
        Returns: (N, C) weighted probability array
        """
        weighted_probs = None
        for name, logits in logits_dict.items():
            probs = self._softmax(logits)
            w = self.norm_weights.get(name, 1.0 / len(logits_dict))
            if weighted_probs is None:
                weighted_probs = w * probs
            else:
                weighted_probs += w * probs
        return weighted_probs

    @staticmethod
    def _softmax(logits):
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class StackedEnsemble:
    """
    Stage B: Stacked meta-learner using k-fold OOF predictions.
    Trains LightGBM on concatenated model logits + image features.
    """

    def __init__(self, num_classes: int = 38, n_folds: int = 5):
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.meta_model = None

    def generate_oof_predictions(self, models_dir: str, data_dir: str, device: torch.device):
        """
        Generate out-of-fold predictions for all base models using k-fold CV.
        This avoids data leakage in the stacking process.
        """
        # Load validation dataset
        val_dir = os.path.join(data_dir, "val")
        val_dataset = PlantDiseaseDataset(val_dir, get_val_transforms(224))
        labels = np.array(val_dataset.targets)
        n_samples = len(val_dataset)

        # Discover trained models
        model_dirs = []
        for d in sorted(os.listdir(models_dir)):
            ckpt_path = os.path.join(models_dir, d, "best_model.pt")
            if os.path.exists(ckpt_path):
                model_dirs.append((d, ckpt_path))

        print(f"Found {len(model_dirs)} trained models")

        # Collect all model logits
        all_model_logits = {}
        model_f1_scores = {}

        for model_name, ckpt_path in model_dirs:
            print(f"Extracting logits from {model_name}...")
            model, ckpt = load_trained_model(ckpt_path, device)
            val_f1 = ckpt.get('val_f1', 0.5)
            model_f1_scores[model_name] = val_f1

            loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
            logits, _ = extract_logits(model, loader, device)
            all_model_logits[model_name] = logits
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Generate OOF predictions using k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Stack all model logits: (N, num_models * num_classes)
        model_names = sorted(all_model_logits.keys())
        stacked_logits = np.hstack([all_model_logits[name] for name in model_names])

        # Compute image features
        loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
        img_features = compute_image_features(loader)

        # Confidence features: max prob, entropy per model
        confidence_features = []
        for name in model_names:
            probs = WeightedSoftVoting._softmax(all_model_logits[name])
            max_prob = probs.max(axis=1, keepdims=True)
            entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1, keepdims=True)
            confidence_features.append(np.hstack([max_prob, entropy]))
        confidence_features = np.hstack(confidence_features)

        # Final meta-features: logits + image features + confidence
        meta_features = np.hstack([stacked_logits, img_features, confidence_features])

        # K-fold OOF training
        oof_preds = np.zeros((n_samples, self.num_classes))

        for fold, (train_idx, val_idx) in enumerate(skf.split(meta_features, labels)):
            print(f"  Fold {fold + 1}/{self.n_folds}")

            X_train = meta_features[train_idx]
            y_train = labels[train_idx]
            X_val = meta_features[val_idx]

            # Train LightGBM
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                max_depth=6, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                n_jobs=-1, random_state=42 + fold, verbose=-1,
                num_class=self.num_classes, objective='multiclass'
            )
            lgb_model.fit(X_train, y_train)

            # OOF predictions
            oof_preds[val_idx] = lgb_model.predict_proba(X_val)

        # Evaluate OOF
        oof_labels = oof_preds.argmax(axis=1)
        oof_f1 = f1_score(labels, oof_labels, average='macro')
        oof_acc = accuracy_score(labels, oof_labels)
        print(f"\nOOF Results: F1={oof_f1:.4f}, Acc={oof_acc:.4f}")

        # Train final meta-model on all data
        self.meta_model = lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            max_depth=6, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, random_state=42, verbose=-1,
            num_class=self.num_classes, objective='multiclass'
        )
        self.meta_model.fit(meta_features, labels)

        return {
            "model_names": model_names,
            "model_f1_scores": model_f1_scores,
            "oof_f1": oof_f1,
            "oof_acc": oof_acc,
            "feature_dim": meta_features.shape[1],
        }

    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """Predict using the trained meta-model."""
        return self.meta_model.predict_proba(meta_features)

    def save(self, path: str):
        joblib.dump(self.meta_model, path)
        print(f"Meta-model saved to {path}")

    def load(self, path: str):
        self.meta_model = joblib.load(path)


def compute_uncertainty(logits_dict: dict) -> dict:
    """
    Compute uncertainty estimates:
    - Predictive entropy (aleatoric)
    - Model disagreement / std across models (epistemic)
    - Combined uncertainty score
    """
    model_probs = []
    for name, logits in logits_dict.items():
        probs = WeightedSoftVoting._softmax(logits)
        model_probs.append(probs)

    model_probs = np.stack(model_probs, axis=0)  # (M, N, C)
    mean_probs = model_probs.mean(axis=0)  # (N, C)

    # Predictive entropy
    pred_entropy = -(mean_probs * np.log(mean_probs + 1e-10)).sum(axis=1)

    # Model disagreement (mean pairwise std)
    model_std = model_probs.std(axis=0).mean(axis=1)

    # Combined uncertainty
    max_entropy = np.log(mean_probs.shape[1])
    norm_entropy = pred_entropy / max_entropy
    uncertainty = 0.5 * norm_entropy + 0.5 * (model_std / model_std.max())

    return {
        "predictive_entropy": pred_entropy,
        "model_disagreement": model_std,
        "uncertainty": uncertainty,
    }


def main():
    parser = argparse.ArgumentParser(description="Run stacking ensemble")
    parser.add_argument("--models_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="checkpoints/ensemble")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage A: Weighted soft voting
    print("=" * 60)
    print("Stage A: Validation-weighted soft voting")
    print("=" * 60)

    model_dirs = []
    model_weights = {}
    for d in sorted(os.listdir(args.models_dir)):
        ckpt_path = os.path.join(args.models_dir, d, "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model_dirs.append((d, ckpt_path))
            model_weights[d] = ckpt.get('val_f1', 0.5)

    voting = WeightedSoftVoting(model_weights)
    print(f"Model weights: {voting.norm_weights}")

    # Stage B: Stacked ensemble
    print("\n" + "=" * 60)
    print("Stage B: Stacked meta-learner (LightGBM)")
    print("=" * 60)

    stacker = StackedEnsemble(num_classes=38, n_folds=args.n_folds)
    results = stacker.generate_oof_predictions(args.models_dir, args.data_dir, device)

    # Save
    stacker.save(os.path.join(args.output_dir, "meta_model.joblib"))

    with open(os.path.join(args.output_dir, "ensemble_config.json"), "w") as f:
        json.dump({
            "model_names": results["model_names"],
            "model_f1_scores": results["model_f1_scores"],
            "voting_weights": voting.norm_weights,
            "oof_f1": results["oof_f1"],
            "oof_acc": results["oof_acc"],
            "n_folds": args.n_folds,
            "feature_dim": results["feature_dim"],
        }, f, indent=2)

    print(f"\nEnsemble artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
