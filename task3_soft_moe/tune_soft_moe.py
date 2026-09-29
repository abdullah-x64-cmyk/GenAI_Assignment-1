from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch.optim import AdamW

from task3_soft_moe.soft_moe import load_soft_moe
from task3_soft_moe.train_soft_moe import build_training_loader, set_expert_requires_grad, train_stage

SEARCH_SPACE = {
    "lambda_rec": "float [0.5, 2.0]",
    "lambda_ssim": "float [0.5, 2.0]",
    "lambda_cls": "float [0.01, 0.5]",
    "lambda_bal": "float [1e-4, 0.2], log scale",
    "temperature": "float [0.25, 2.0]",
    "joint_learning_rate": "float [1e-6, 1e-4], log scale",
}


def objective(
    trial: optuna.Trial,
    project_root: Path,
    image_size: int,
    batch_size: int,
    max_batches: int | None,
    seed: int,
) -> float:
    params = {
        "lambda_rec": trial.suggest_float("lambda_rec", 0.5, 2.0),
        "lambda_ssim": trial.suggest_float("lambda_ssim", 0.5, 2.0),
        "lambda_cls": trial.suggest_float("lambda_cls", 0.01, 0.5),
        "lambda_bal": trial.suggest_float("lambda_bal", 1e-4, 0.2, log=True),
        "temperature": trial.suggest_float("temperature", 0.25, 2.0),
        "joint_learning_rate": trial.suggest_float("joint_learning_rate", 1e-6, 1e-4, log=True),
    }
    model = load_soft_moe(
        "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
        {
            "salt": "task2_hard_routing/checkpoints/salt_expert_best.pt",
            "blur": "task2_hard_routing/checkpoints/blur_expert_best.pt",
            "occlusion": "task2_hard_routing/checkpoints/occlusion_expert_best.pt",
        },
        project_root=project_root,
        temperature=params["temperature"],
        device=torch.device("cpu"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = build_training_loader(project_root, batch_size, image_size, seed)

    set_expert_requires_grad(model, enabled=False)
    gate_optimizer = AdamW(model.gate.parameters(), lr=1e-4)
    train_stage(model, loader, gate_optimizer, device, params["lambda_rec"], params["lambda_ssim"], params["lambda_cls"], params["lambda_bal"], max_batches, True)

    set_expert_requires_grad(model, enabled=True)
    joint_optimizer = AdamW(model.parameters(), lr=params["joint_learning_rate"])
    metrics = train_stage(model, loader, joint_optimizer, device, params["lambda_rec"], params["lambda_ssim"], params["lambda_cls"], params["lambda_bal"], max_batches, True)
    trial.set_user_attr("joint_metrics", metrics)
    return metrics["loss"]


def run_optuna(
    n_trials: int = 3,
    image_size: int = 64,
    batch_size: int = 8,
    max_batches: int | None = 1,
    seed: int = 42,
    output: str | Path = "reports/task3_soft_moe_optuna_results.json",
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project_root = Path(__file__).resolve().parents[1]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial: objective(trial, project_root, image_size, batch_size, max_batches, seed), n_trials=n_trials)
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": trial.params,
            "joint_metrics": trial.user_attrs.get("joint_metrics", {}),
        }
        for trial in study.trials
    ]
    result = {
        "search_space": SEARCH_SPACE,
        "number_of_trials_requested": n_trials,
        "number_of_completed_trials": len(trials),
        "image_size": image_size,
        "batch_size": batch_size,
        "max_batches": max_batches,
        "initialization": "Task 2 classifier and three Task 2 specialists",
        "best_trial": study.best_trial.number,
        "best_joint_loss": study.best_value,
        "best_parameters": study.best_params,
        "best_joint_metrics": study.best_trial.user_attrs["joint_metrics"],
        "trials": trials,
    }
    output_file = project_root / output if not Path(output).is_absolute() else Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Task 3 soft MoE loss weights and temperature.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/task3_soft_moe_optuna_results.json")
    args = parser.parse_args()
    run_optuna(args.trials, args.image_size, args.batch_size, args.max_batches, args.seed, args.output)
