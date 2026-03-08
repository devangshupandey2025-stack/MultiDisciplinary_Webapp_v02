"""
Train all 5 base models sequentially with their configs.

Usage:
  python ml_pipeline/scripts/train_all.py --data_dir data/plantvillage
"""
import os
import sys
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.scripts.train import train


CONFIGS = [
    "ml_pipeline/configs/efficientnet_v2.yaml",
    "ml_pipeline/configs/resnet50.yaml",
    "ml_pipeline/configs/convnext_tiny.yaml",
    "ml_pipeline/configs/swin_tiny.yaml",
    "ml_pipeline/configs/mobilenet_v3.yaml",
]


def main():
    parser = argparse.ArgumentParser(description="Train all base models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()

    configs = args.configs or CONFIGS
    results = {}

    for config_path in configs:
        print(f"\n{'='*60}")
        print(f"Training: {config_path}")
        print(f"{'='*60}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        try:
            best_f1 = train(cfg, args.data_dir)
            results[config_path] = {"status": "success", "best_f1": best_f1}
        except Exception as e:
            print(f"Error training {config_path}: {e}")
            results[config_path] = {"status": "error", "error": str(e)}

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for config, result in results.items():
        model = os.path.basename(config).replace('.yaml', '')
        if result['status'] == 'success':
            print(f"  {model}: F1={result['best_f1']:.4f} ✓")
        else:
            print(f"  {model}: FAILED - {result['error']}")


if __name__ == "__main__":
    main()
