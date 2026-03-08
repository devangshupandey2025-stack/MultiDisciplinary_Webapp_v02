"""
Comprehensive evaluation pipeline.
Computes: per-class P/R/F1, macro F1, confusion matrix, ROC/AUC,
top-k accuracy, ECE, reliability diagrams, robustness tests.

Usage:
  python ml_pipeline/scripts/evaluate.py --ensemble_dir checkpoints/ensemble --data_dir data/plantvillage
"""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_auc_score, precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.data.dataset import PLANTVILLAGE_CLASSES
from ml_pipeline.scripts.calibrate import compute_ece, compute_ace, plot_reliability_diagram


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return correct.mean()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='.1f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def robustness_test(model, test_loader, device, perturbation="brightness", severity=0.3):
    """Simple robustness test with brightness/blur perturbation."""
    from torchvision import transforms
    
    model.eval()
    all_preds, all_labels = [], []

    for batch in test_loader:
        images = batch[0].to(device)
        labels = batch[1]

        if perturbation == "brightness":
            images = images + severity * torch.randn_like(images) * 0.1
            images = torch.clamp(images, 0, 1)
        elif perturbation == "blur":
            import torch.nn.functional as F
            kernel_size = int(severity * 10) * 2 + 1
            images = F.avg_pool2d(images, kernel_size, stride=1,
                                   padding=kernel_size // 2)

        with torch.no_grad():
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return {"perturbation": perturbation, "severity": severity, "f1": f1, "accuracy": acc}


def full_evaluation(probs: np.ndarray, labels: np.ndarray,
                    output_dir: str, class_names: list = None):
    """Run full evaluation suite."""
    os.makedirs(output_dir, exist_ok=True)
    if class_names is None:
        class_names = PLANTVILLAGE_CLASSES

    preds = probs.argmax(axis=1)

    # Basic metrics
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    top3_acc = top_k_accuracy(probs, labels, k=3)
    top5_acc = top_k_accuracy(probs, labels, k=5)
    ece = compute_ece(probs, labels)
    ace = compute_ace(probs, labels)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Top-1 Accuracy:  {acc:.4f}")
    print(f"Top-3 Accuracy:  {top3_acc:.4f}")
    print(f"Top-5 Accuracy:  {top5_acc:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"ECE:             {ece:.4f}")
    print(f"ACE:             {ace:.4f}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    per_class = {}
    low_f1_classes = []
    for i, name in enumerate(class_names[:len(f1)]):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        if f1[i] < 0.85:
            low_f1_classes.append((name, f1[i]))

    if low_f1_classes:
        print(f"\n⚠️  Classes with F1 < 0.85:")
        for name, score in sorted(low_f1_classes, key=lambda x: x[1]):
            print(f"  {name}: F1={score:.4f}")

    # Classification report
    report = classification_report(labels, preds, target_names=class_names[:len(np.unique(labels))],
                                    zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    plot_confusion_matrix(labels, preds, class_names[:len(np.unique(labels))],
                          os.path.join(output_dir, "confusion_matrix.png"))

    # Reliability diagram
    plot_reliability_diagram(probs, labels,
                             title=f"Reliability Diagram (ECE={ece:.4f})",
                             save_path=os.path.join(output_dir, "reliability_diagram.png"))

    # ROC AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        labels_bin = label_binarize(labels, classes=range(probs.shape[1]))
        if labels_bin.shape[1] > 1:
            auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
            print(f"ROC AUC (macro): {auc:.4f}")
        else:
            auc = None
    except Exception:
        auc = None

    # Save results
    results = {
        "top1_accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "macro_f1": float(macro_f1),
        "ece": float(ece),
        "ace": float(ace),
        "roc_auc_macro": float(auc) if auc else None,
        "per_class": per_class,
        "low_f1_classes": [(n, float(s)) for n, s in low_f1_classes],
    }
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble")
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--logits_path", type=str, default=None)
    parser.add_argument("--labels_path", type=str, default=None)
    args = parser.parse_args()

    output_dir = os.path.join(args.ensemble_dir, "evaluation")

    if args.logits_path:
        logits = np.load(args.logits_path)
        labels = np.load(args.labels_path)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    else:
        # Look for logits in checkpoint dirs
        parent = os.path.dirname(args.ensemble_dir)
        for d in os.listdir(parent):
            logits_path = os.path.join(parent, d, "best_val_logits.npy")
            if os.path.exists(logits_path):
                logits = np.load(logits_path)
                labels = np.load(os.path.join(parent, d, "best_val_labels.npy"))
                probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
                break
        else:
            print("No logits found. Run training first.")
            return

    full_evaluation(probs, labels, output_dir)


if __name__ == "__main__":
    main()
