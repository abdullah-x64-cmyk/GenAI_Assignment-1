from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task2_hard_routing.corruption_classifier import (
    CLASS_NAMES,
    BalancedBatchSampler,
    BalancedCorruptionDataset,
    CorruptionClassifier,
)

SEARCH_SPACE = {
    "learning_rate": "float [1e-4, 1e-3], log scale",
    "batch_size": "categorical [8, 16, 32]",
    "channels": "categorical [(16, 32, 64), (32, 64, 128), (48, 96, 192)]",
    "dropout": "float [0.0, 0.5]",
    "weight_decay": "float [1e-6, 1e-3], log scale",
}


def build_classifier_loaders(
    project_root: Path,
    batch_size: int,
    image_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    data_root = project_root / "data" / "oxford_iiit_pet"
    train_dataset = BalancedCorruptionDataset(
        project_root / "data" / "oxford_iiit_pet_splits" / "train.txt",
        data_root,
        image_size=image_size,
        seed=seed,
    )
    val_dataset = BalancedCorruptionDataset(
        project_root / "data" / "oxford_iiit_pet_splits" / "val.txt",
        data_root,
        image_size=image_size,
        seed=seed + 1,
    )
    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size, seed=seed)
    val_sampler = BalancedBatchSampler(val_dataset, batch_size=batch_size, seed=seed + 1)
    return (
        DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0),
        DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0),
    )


def train_epoch(
    model: CorruptionClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def evaluate_classifier(
    model: CorruptionClassifier,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            total_samples += labels.size(0)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())

    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
    per_class = {
        class_name: {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
        "sample_count": len(true_labels),
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
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    channels = trial.suggest_categorical("channels", [(16, 32, 64), (32, 64, 128), (48, 96, 192)])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    train_loader, val_loader = build_classifier_loaders(project_root, batch_size, image_size, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptionClassifier(channels=channels, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss = 0.0
    validation: dict[str, Any] = {}
    for _ in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, max_train_batches)
        validation = evaluate_classifier(model, val_loader, device, max_val_batches)
    trial.set_user_attr("train_loss", train_loss)
    trial.set_user_attr("validation", validation)
    return validation["macro_f1"]


def run_optuna(
    n_trials: int = 3,
    epochs: int = 1,
    image_size: int = 64,
    max_train_batches: int | None = 8,
    max_val_batches: int | None = 4,
    seed: int = 42,
    output: str | Path = "reports/task2_classifier_optuna_results.json",
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project_root = Path(__file__).resolve().parents[1]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective(
            trial, project_root, image_size, epochs, max_train_batches, max_val_batches, seed
        ),
        n_trials=n_trials,
    )
    complete_trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    best_validation = study.best_trial.user_attrs["validation"]
    result = {
        "search_space": SEARCH_SPACE,
        "number_of_trials_requested": n_trials,
        "number_of_completed_trials": len(complete_trials),
        "epochs_per_trial": epochs,
        "image_size": image_size,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "best_trial": study.best_trial.number,
        "best_macro_f1": study.best_value,
        "best_parameters": study.best_params,
        "best_validation_metrics": best_validation,
        "trials": complete_trials,
    }
    output_file = project_root / output if not Path(output).is_absolute() else Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and evaluate the Task 2 corruption classifier.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/task2_classifier_optuna_results.json")
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
