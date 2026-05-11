import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..training.trainer import Trainer
from ..utils.config import base_dataset_config, default_logging_config, default_training_config, recursive_update
from ..utils.io import save_csv
from .search_space import random_hparam_sample


def build_run_name(model_name: str, trial: int) -> str:
    return f"stageB_{model_name}_trial{trial}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def should_skip(root_dir: str, model_name: str, trial: int) -> bool:
    prefix = f"stageB_{model_name}_trial{trial}_"
    for p in Path(root_dir).glob(f"{prefix}*"):
        if (p / "metrics.json").exists():
            return True
    return False


def run_stage_b(
    data_dir: str,
    model_names: List[str],
    trials_per_model: int = 5,
    seed: int = 42,
    skip_existing: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    dataset_cfg = base_dataset_config(data_dir)
    logging_cfg = default_logging_config()
    results: List[Dict[str, Any]] = []

    for name in model_names:
        for t in range(trials_per_model):
            if skip_existing and should_skip(logging_cfg["root_dir"], name, t):
                print(f"Skip stageB {name} trial {t} (metrics already present)")
                continue
            h = random_hparam_sample()
            model_cfg: Dict[str, Any] = {
                "name": name,
                "input_channels": dataset_cfg["input_channels"],
                "num_classes": dataset_cfg["num_classes"],
                "base_width": h.get("base_width", 64),
                "num_blocks": h.get("num_blocks", 4),
                "dropout": h["dropout"],
            }
            train_cfg = recursive_update(default_training_config(), h)
            train_cfg["batch_size"] = batch_size
            train_cfg["num_workers"] = num_workers
            run_name = build_run_name(name, t)
            trainer = Trainer(dataset_cfg, model_cfg, train_cfg, logging_cfg, run_name, seed=seed + t)
            metrics = trainer.train()
            metrics.update({"run_name": run_name, "model_name": name})
            results.append(metrics)
            print(f"Finished {run_name}: best_acc={metrics['best_val_acc']:.4f}")

    summary_path = os.path.join(logging_cfg["root_dir"], "stage_b_summary.csv")
    save_csv(results, summary_path)
    print(f"Saved Stage B summary to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Transfer_Learning/Data/XRF55")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="model names, e.g., plain resnet multiscale")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_skip_existing", action="store_true", help="do not skip trials with existing metrics")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_stage_b(
        args.data_dir,
        model_names=args.models,
        trials_per_model=args.trials,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
