import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..training.trainer import Trainer
from ..utils.config import base_dataset_config, default_logging_config, default_training_config, recursive_update
from ..utils.io import save_csv
from .search_space import stage_a_backbones


def build_run_name(model_cfg: Dict[str, Any]) -> str:
    tag = model_cfg.get("name", "model")
    width = model_cfg.get("base_width", 0)
    blocks = model_cfg.get("num_blocks", 0)
    return f"stageA_{tag}_w{width}_b{blocks}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def has_finished(root_dir: str, model_cfg: Dict[str, Any]) -> Optional[str]:
    """Check if a previous identical config run already produced metrics.

    This allows splitting Stage A into multiple shorter calls without重复训练。
    """
    tag = model_cfg.get("name", "model")
    width = model_cfg.get("base_width", 0)
    blocks = model_cfg.get("num_blocks", 0)
    prefix = f"stageA_{tag}_w{width}_b{blocks}_"
    root = Path(root_dir)
    for p in root.glob(f"{prefix}*"):
        metrics_path = p / "metrics.json"
        if metrics_path.exists():
            return str(metrics_path)
    return None


def run_stage_a(
    data_dir: str,
    epochs: int = 80,
    batch_size: int = 128,
    seed: int = 42,
    models: Optional[List[str]] = None,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    dataset_cfg = base_dataset_config(data_dir)
    train_cfg = recursive_update(default_training_config(), {"epochs": epochs, "batch_size": batch_size})
    logging_cfg = default_logging_config()

    results: List[Dict[str, Any]] = []
    model_list = stage_a_backbones()
    if models:
        wanted = set([m.lower() for m in models])
        model_list = [m for m in model_list if m.get("name", "").lower() in wanted]

    for model_cfg in model_list:
        model_cfg = recursive_update({
            "input_channels": dataset_cfg["input_channels"],
            "num_classes": dataset_cfg["num_classes"],
        }, model_cfg)

        if skip_existing:
            finished = has_finished(logging_cfg["root_dir"], model_cfg)
            if finished:
                print(f"Skip {model_cfg['name']} w{model_cfg.get('base_width')} b{model_cfg.get('num_blocks')} (found {finished})")
                continue

        run_name = build_run_name(model_cfg)
        trainer = Trainer(dataset_cfg, model_cfg, train_cfg, logging_cfg, run_name, seed=seed)
        metrics = trainer.train()
        metrics["run_name"] = run_name
        metrics["model_name"] = model_cfg["name"]
        results.append(metrics)
        print(f"Finished {run_name}: best_acc={metrics['best_val_acc']:.4f}")

    summary_path = os.path.join(logging_cfg["root_dir"], "stage_a_summary.csv")
    save_csv(results, summary_path)
    print(f"Saved Stage A summary to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Transfer_Learning/Data/XRF55")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, nargs="+", help="optional subset of backbones: plain resnet multiscale dilated attention depthwise")
    parser.add_argument("--no_skip_existing", action="store_true", help="do not skip if metrics.json already exists")
    args = parser.parse_args()

    run_stage_a(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        models=args.models,
        skip_existing=not args.no_skip_existing,
    )
