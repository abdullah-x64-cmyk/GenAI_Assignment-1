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
from torch.utils.data import DataLoader

from task1_universal.universal_autoencoder import UniversalAutoencoder
from task2_hard_routing.specialist_autoencoders import SPECIALIST_TYPES, build_specialist_datasets
from task2_hard_routing.train_specialists import evaluate, train_epoch

SEARCH_SPACE = {
    "learning_rate": "float [1e-4, 1e-3], log scale",
    "bottleneck": "categorical [32, 64, 128]",
    "channels": "categorical [16, 32, 48] base channels; encoder is [base, 2*base, 4*base, 8*base]",
    "batch_size": "categorical [4, 8, 16]",
    "alpha": "float [0.5, 0.95] for alpha*L1 + (1-alpha)*(1-SSIM)",
}


def run_specialist_trial(
    specialist: str,
    project_root: Path,
    params: dict[str, Any],
    image_size: int,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    seed: int,
) -> dict[str, Any]:
    train_dataset, val_dataset = build_specialist_datasets(project_root, specialist, image_size=image_size, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalAutoencoder(
        base_channels=params["channels"],
        latent_dim=params["bottleneck"],
        dropout=0.0,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)
    train_loss = 0.0
    validation: dict[str, float] = {}
    for _ in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, params["alpha"], max_train_batches)
        validation = evaluate(model, val_loader, device, params["alpha"], max_val_batches)
    return {
        "train_loss": train_loss,
        "validation": validation,
        "device": str(device),
    }


def objective(
    trial: optuna.Trial,
    project_root: Path,
    image_size: int,
    epochs: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    seed: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "bottleneck": trial.suggest_categorical("bottleneck", [32, 64, 128]),
        "channels": trial.suggest_categorical("channels", [16, 32, 48]),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "alpha": trial.suggest_float("alpha", 0.5, 0.95),
    }
    specialist_results = {
        specialist: run_specialist_trial(
            specialist,
            project_root,
            params,
            image_size,
            epochs,
            max_train_batches,
            max_val_batches,
            seed,
        )
        for specialist in SPECIALIST_TYPES
    }
    mean_validation_loss = float(np.mean([item["validation"]["loss"] for item in specialist_results.values()]))
    trial.set_user_attr("specialists", specialist_results)
    return mean_validation_loss


def run_optuna(
    n_trials: int = 3,
    epochs: int = 1,
    image_size: int = 64,
    max_train_batches: int | None = 1,
    max_val_batches: int | None = 1,
    seed: int = 42,
    output: str | Path = "reports/task2_specialist_optuna_results.json",
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project_root = Path(__file__).resolve().parents[1]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective(
            trial,
            project_root,
            image_size,
            epochs,
            max_train_batches,
            max_val_batches,
            seed,
        ),
        n_trials=n_trials,
    )
    trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "mean_validation_loss": trial.value,
            "parameters": trial.params,
            "specialists": trial.user_attrs.get("specialists", {}),
        }
        for trial in study.trials
    ]
    result = {
        "search_space": SEARCH_SPACE,
        "number_of_trials_requested": n_trials,
        "number_of_completed_trials": len(trials),
        "epochs_per_trial": epochs,
        "image_size": image_size,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "optimization_objective": "mean validation loss across salt, blur, and occlusion specialists",
        "best_trial": study.best_trial.number,
        "best_mean_validation_loss": study.best_value,
        "best_shared_parameters": study.best_params,
        "best_specialist_validation": study.best_trial.user_attrs["specialists"],
        "trials": trials,
    }
    output_file = project_root / output if not Path(output).is_absolute() else Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared Optuna search for Task 2 specialist architecture.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/task2_specialist_optuna_results.json")
    args = parser.parse_args()
    run_optuna(
        n_trials=args.trials,
        epochs=args.epochs,
        image_size=args.image_size,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        seed=args.seed,
        output=args.output,
    )
