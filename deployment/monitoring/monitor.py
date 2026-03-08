"""
Monitoring and drift detection for production deployment.
Logs predictions, tracks distribution shifts, raises alerts.
"""
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


class PredictionMonitor:
    """Monitor prediction distributions and detect drift."""

    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.predictions = []
        self.baseline_distribution = None
        self.alerts = []

    def log_prediction(self, prediction: dict):
        """Log a prediction for monitoring."""
        entry = {
            "class": prediction.get("class", ""),
            "probability": prediction.get("probability", 0),
            "uncertainty": prediction.get("uncertainty", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.predictions.append(entry)

        # Keep window
        if len(self.predictions) > self.window_size * 2:
            self.predictions = self.predictions[-self.window_size:]

    def set_baseline(self, class_distribution: Dict[str, float]):
        """Set baseline class distribution from training/validation data."""
        self.baseline_distribution = class_distribution

    def check_drift(self) -> Optional[dict]:
        """Check for distribution drift using KL divergence."""
        if not self.baseline_distribution or len(self.predictions) < 100:
            return None

        recent = self.predictions[-min(self.window_size, len(self.predictions)):]
        class_counts = defaultdict(int)
        for p in recent:
            class_counts[p["class"]] += 1

        total = len(recent)
        current_dist = {k: v / total for k, v in class_counts.items()}

        # KL divergence
        kl_div = 0
        for cls, baseline_prob in self.baseline_distribution.items():
            current_prob = current_dist.get(cls, 1e-10)
            kl_div += baseline_prob * np.log(baseline_prob / max(current_prob, 1e-10))

        if kl_div > self.alert_threshold:
            alert = {
                "type": "distribution_drift",
                "kl_divergence": float(kl_div),
                "threshold": self.alert_threshold,
                "timestamp": datetime.utcnow().isoformat(),
                "window_size": len(recent),
            }
            self.alerts.append(alert)
            return alert
        return None

    def get_metrics(self) -> dict:
        """Get current monitoring metrics."""
        if not self.predictions:
            return {"total_predictions": 0}

        recent = self.predictions[-min(100, len(self.predictions)):]
        probs = [p["probability"] for p in recent]
        uncertainties = [p["uncertainty"] for p in recent]

        class_counts = defaultdict(int)
        for p in recent:
            class_counts[p["class"]] += 1

        return {
            "total_predictions": len(self.predictions),
            "recent_window": len(recent),
            "avg_confidence": float(np.mean(probs)),
            "avg_uncertainty": float(np.mean(uncertainties)),
            "high_uncertainty_rate": float(np.mean([u > 0.3 for u in uncertainties])),
            "class_distribution": dict(class_counts),
            "num_alerts": len(self.alerts),
        }


# Prometheus metrics (optional integration)
PROMETHEUS_CONFIG = """
# Prometheus scrape config for PlantGuard
scrape_configs:
  - job_name: 'plantguard'
    scrape_interval: 15s
    static_configs:
      - targets: ['backend:8000']
    metrics_path: /metrics
"""

# Grafana dashboard skeleton
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "PlantGuard Monitoring",
        "panels": [
            {"title": "Predictions/min", "type": "graph"},
            {"title": "Average Confidence", "type": "gauge"},
            {"title": "Average Uncertainty", "type": "gauge"},
            {"title": "High Uncertainty Rate", "type": "stat"},
            {"title": "Class Distribution", "type": "piechart"},
            {"title": "Distribution Drift (KL)", "type": "timeseries"},
        ],
    }
}
