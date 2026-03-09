"""
Optuna hyperparameter sweep configuration.
Usage:
  python ml_pipeline/scripts/hyperparam_sweep.py --config ml_pipeline/configs/efficientnet_v2.yaml --data_dir data/plantvillage --n_trials 10
"""
import sys
import argparse
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.scripts.train import train


def objective(trial, base_cfg, data_dir):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_cfg.items()}
    cfg['training'] = dict(base_cfg['training'])
    cfg['data'] = dict(base_cfg['data'])

    # Search space
    cfg['training']['learning_rate'] = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    cfg['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    cfg['training']['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.15)
    cfg['training']['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.0, 0.4)
    cfg['training']['cutmix_prob'] = trial.suggest_float('cutmix_prob', 0.0, 0.7)
    cfg['model'] = dict(base_cfg['model'])
    cfg['model']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
    cfg['data']['augment_level'] = trial.suggest_categorical('augment_level', ['light', 'medium', 'heavy'])
    cfg['training']['epochs'] = 20  # Shorter for sweep
    cfg['checkpoint'] = dict(base_cfg['checkpoint'])
    cfg['checkpoint']['save_dir'] = f"checkpoints/sweep/trial_{trial.number}"
    cfg['loss'] = dict(base_cfg['loss'])

    best_f1 = train(cfg, data_dir)
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep with Optuna")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--study_name", type=str, default="plantguard_sweep")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(trial, cfg, args.data_dir),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial: F1={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
