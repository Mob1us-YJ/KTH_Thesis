import random
from typing import Any, Dict, List


def stage_a_backbones() -> List[Dict[str, Any]]:
    return [
        {"name": "plain", "base_width": w, "num_blocks": 4} for w in (64, 96)
    ] + [
        {"name": "resnet", "base_width": w, "num_blocks": 4} for w in (64, 96)
    ] + [
        {"name": "multiscale", "base_width": 64, "num_blocks": 4, "kernel_sizes": [[3, 5, 7]] * 4},
        {"name": "dilated", "base_width": 64, "num_blocks": 4, "kernel_size": 3, "dilations": [1, 2, 4, 8]},
        {"name": "attention", "base_width": 64, "num_blocks": 4},
        {"name": "depthwise", "base_width": 64, "num_blocks": 4},
    ]


def random_hparam_sample() -> Dict[str, Any]:
    return {
        "base_width": random.choice([64, 96, 128]),
        "num_blocks": random.choice([3, 4, 5]),
        "optimizer": {
            "name": random.choice(["adam", "adamw"]),
            "lr": 10 ** random.uniform(-4.2, -2.8),
            "weight_decay": 10 ** random.uniform(-5, -3.5),
        },
        "dropout": random.uniform(0.2, 0.45),
        "label_smoothing": random.uniform(0.0, 0.08),
        "scheduler": random.choice([
            {"name": "cosine"},
            {"name": "step", "step_size": random.choice([8, 10, 12]), "gamma": random.choice([0.5, 0.6, 0.7])},
        ]),
        "mixup_alpha": random.choice([None, 0.2, 0.3]),
        "use_sampler": random.choice([False, False, True]),
        "focal_gamma": random.choice([None, None, 1.5, 2.0]),
    }


def pick_top_backbones(stage_a_results: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    sorted_runs = sorted(stage_a_results, key=lambda x: x.get("best_val_acc", 0), reverse=True)
    names = []
    for run in sorted_runs:
        name = run.get("model_name")
        if name not in names:
            names.append(name)
        if len(names) >= top_k:
            break
    return names
