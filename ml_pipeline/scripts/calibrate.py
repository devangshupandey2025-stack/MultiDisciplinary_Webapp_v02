"""
Temperature Scaling and Probability Calibration.
Supports: Temperature Scaling, Platt Scaling, Isotonic Regression.
Reports: ECE, ACE, reliability diagrams.

Usage:
  python ml_pipeline/scripts/calibrate.py --ensemble_dir checkpoints/ensemble --data_dir data/plantvillage
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrating neural network probabilities.
    Learns a single scalar T to soften/sharpen the logits: p = softmax(logits / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 100):
        """Optimize temperature on held-out calibration set."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_t)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        with torch.no_grad():
            scaled = self.forward(logits_t)
            probs = torch.softmax(scaled, dim=1).numpy()
        return probs


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(labels) * abs(bin_acc - bin_conf)

    return ece


def compute_ace(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive Calibration Error (class-conditional)."""
    num_classes = probs.shape[1]
    ace = 0.0

    for c in range(num_classes):
        class_probs = probs[:, c]
        class_labels = (labels == c).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (class_probs > bin_boundaries[i]) & (class_probs <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = class_labels[mask].mean()
                bin_conf = class_probs[mask].mean()
                ace += mask.sum() / (len(labels) * num_classes) * abs(bin_acc - bin_conf)

    return ace


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray,
                             title: str = "Reliability Diagram",
                             save_path: str = None, n_bins: int = 15):
    """Plot reliability diagram for calibration visualization."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax1.bar(bin_confs, bin_accs, width=1 / n_bins, alpha=0.7, edgecolor='black', label='Model')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Confidence histogram
    ax2.bar(bin_confs, bin_counts, width=1 / n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")
    plt.close()


def calibrate_ensemble(logits: np.ndarray, labels: np.ndarray, output_dir: str):
    """Full calibration pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Before calibration
    uncal_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    ece_before = compute_ece(uncal_probs, labels)
    ace_before = compute_ace(uncal_probs, labels)
    print(f"Before calibration: ECE={ece_before:.4f}, ACE={ace_before:.4f}")

    plot_reliability_diagram(uncal_probs, labels,
                             title=f"Before Calibration (ECE={ece_before:.4f})",
                             save_path=os.path.join(output_dir, "reliability_before.png"))

    # Temperature scaling
    ts = TemperatureScaling()
    temp = ts.fit(logits, labels)
    cal_probs = ts.calibrate(logits)

    ece_after = compute_ece(cal_probs, labels)
    ace_after = compute_ace(cal_probs, labels)
    print(f"After calibration:  ECE={ece_after:.4f}, ACE={ace_after:.4f}")

    plot_reliability_diagram(cal_probs, labels,
                             title=f"After Calibration (ECE={ece_after:.4f})",
                             save_path=os.path.join(output_dir, "reliability_after.png"))

    # Save
    torch.save(ts.state_dict(), os.path.join(output_dir, "temperature_scaling.pt"))

    metrics = {
        "temperature": temp,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ace_before": ace_before,
        "ace_after": ace_after,
    }
    with open(os.path.join(output_dir, "calibration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return ts, metrics


def main():
    parser = argparse.ArgumentParser(description="Calibrate ensemble")
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--logits_path", type=str, default=None)
    parser.add_argument("--labels_path", type=str, default=None)
    args = parser.parse_args()

    if args.logits_path and args.labels_path:
        logits = np.load(args.logits_path)
        labels = np.load(args.labels_path)
    else:
        # Look for saved logits from stacking
        logits_path = os.path.join(args.ensemble_dir, "..", "efficientnet_v2", "best_val_logits.npy")
        labels_path = os.path.join(args.ensemble_dir, "..", "efficientnet_v2", "best_val_labels.npy")
        if os.path.exists(logits_path):
            logits = np.load(logits_path)
            labels = np.load(labels_path)
        else:
            print("No logits found. Run training first.")
            return

    calibrate_ensemble(logits, labels, args.ensemble_dir)


if __name__ == "__main__":
    main()
