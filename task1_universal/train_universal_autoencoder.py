from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task1_universal.universal_autoencoder import (
    UniversalAutoencoder,
    build_loaders,
    save_model,
    ssim_score,
    universal_loss,
)

INITIAL_ALPHA = 0.8


SEARCH_SPACE = {
    "learning_rate": "float [1e-4, 1e-3], log scale",
    "batch_size": "categorical [8, 16, 32]",
    "latent_dim": "categorical [64, 128, 256]",
    "base_channels": "categorical [16, 32, 48]",
    "dropout": "float [0.0, 0.3]",
    "alpha": "float [0.5, 0.95]",
}


def evaluate(
    model: UniversalAutoencoder,
    loader: DataLoader,
    device: torch.device,
    alpha: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            corrupted = batch["corrupted"].to(device)
            clean = batch["clean"].to(device)
            pred = model(corrupted)
            batch_size = corrupted.size(0)
            totals["loss"] += universal_loss(pred, clean, alpha=alpha).item() * batch_size
            totals["l1"] += torch.nn.functional.l1_loss(pred, clean).item() * batch_size
            totals["ssim"] += ssim_score(pred, clean).item() * batch_size
            count += corrupted.size(0)
    model.train()
    return {name: value / max(count, 1) for name, value in totals.items()}


def train_one_epoch(
    model: UniversalAutoencoder,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    alpha: float = 0.8,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        corrupted = batch["corrupted"].to(device)
        clean = batch["clean"].to(device)
        optimizer.zero_grad()
        pred = model(corrupted)
        loss = universal_loss(pred, clean, alpha=alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * corrupted.size(0)
        count += corrupted.size(0)
    return total_loss / max(count, 1)


def objective(
    trial: optuna.Trial,
    image_size: int = 128,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> float:
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    alpha = trial.suggest_float("alpha", 0.5, 0.95)

    train_loader, val_loader = build_loaders(batch_size=batch_size, image_size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalAutoencoder(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, alpha=alpha, max_batches=max_train_batches)
        validation = evaluate(model, val_loader, device, alpha=alpha, max_batches=max_val_batches)
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("validation", validation)
        trial.report(validation["loss"], step=1)

    return validation["loss"]


def run_optuna(
    n_trials: int = 10,
    storage: str | None = None,
    output: str | Path = "reports/task1_optuna_results.json",
    image_size: int = 128,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict:
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=storage)
    study.optimize(
        lambda trial: objective(
            trial,
            image_size=image_size,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        ),
        n_trials=n_trials,
    )
    completed_trials = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
    result = {
        "search_space": SEARCH_SPACE,
        "number_of_trials_requested": n_trials,
        "number_of_completed_trials": len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]),
        "image_size": image_size,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "best_trial": study.best_trial.number,
        "best_validation_loss": study.best_value,
        "best_parameters": study.best_params,
        "best_validation_results": study.best_trial.user_attrs.get("validation", {}),
        "trials": completed_trials,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_quick_check() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(batch_size=8, image_size=128)
    model = UniversalAutoencoder(base_channels=16, latent_dim=64, dropout=0.1).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, alpha=INITIAL_ALPHA)
        validation = evaluate(model, val_loader, device, alpha=INITIAL_ALPHA)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={validation['loss']:.4f}")

    save_dir = Path(__file__).resolve().parent / "checkpoints"
    save_model(model, save_dir / "universal_autoencoder.pt")
    print(f"Saved quick-check model to {save_dir / 'universal_autoencoder.pt'}")


def run_final_training(
    config_path: str | Path = "reports/task1_optuna_results.json",
    epochs: int = 5,
    image_size: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    seed: int = 42,
) -> dict:
    """Train the final model from the selected Optuna configuration and save artifacts."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    project_root = Path(__file__).resolve().parents[1]
    config_file = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    study_report = json.loads(config_file.read_text(encoding="utf-8"))
    params = study_report["best_parameters"]
    selected_image_size = image_size or study_report.get("image_size", 64)

    train_loader, val_loader = build_loaders(batch_size=params["batch_size"], image_size=selected_image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalAutoencoder(
        base_channels=params["base_channels"],
        latent_dim=params["latent_dim"],
        dropout=params["dropout"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=1e-4)

    checkpoint_dir = project_root / "task1_universal" / "checkpoints"
    report_dir = project_root / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            alpha=params["alpha"],
            max_batches=max_train_batches,
        )
        validation = evaluate(
            model,
            val_loader,
            device,
            alpha=params["alpha"],
            max_batches=max_val_batches,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_l1": validation["l1"],
            "val_ssim": validation["ssim"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": params,
                    "image_size": selected_image_size,
                    "validation": validation,
                },
                checkpoint_dir / "universal_autoencoder_best.pt",
            )

    log = {
        "model": "UniversalAutoencoder",
        "config_source": str(config_file.relative_to(project_root)),
        "selected_parameters": params,
        "epochs": epochs,
        "image_size": selected_image_size,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "device": str(device),
        "seed": seed,
        "history": history,
        "best_validation_loss": best_loss,
        "best_checkpoint": "task1_universal/checkpoints/universal_autoencoder_best.pt",
    }
    log_path = report_dir / "task1_final_training_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    epochs_axis = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_axis, [entry["train_loss"] for entry in history], marker="o", label="Train loss")
    axes[0].plot(epochs_axis, [entry["val_loss"] for entry in history], marker="o", label="Validation loss")
    axes[0].set_title("Universal Autoencoder Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs_axis, [entry["val_l1"] for entry in history], marker="o", label="Validation L1")
    axes[1].plot(epochs_axis, [entry["val_ssim"] for entry in history], marker="o", label="Validation SSIM")
    axes[1].set_title("Validation Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(report_dir / "task1_final_training_curves.png", dpi=160)
    plt.close(figure)
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the universal restoration autoencoder.")
    parser.add_argument("--quick-check", action="store_true", help="Run a short smoke test instead of a full Optuna study.")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials to run when not using quick check.")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL.")
    parser.add_argument("--output", type=str, default="reports/task1_optuna_results.json", help="JSON path for the Optuna report.")
    parser.add_argument("--image-size", type=int, default=128, help="Image resolution used during tuning.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap on training batches per trial.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap on validation batches per trial.")
    parser.add_argument("--final", action="store_true", help="Train the final model from the saved best Optuna configuration.")
    parser.add_argument("--config", type=str, default="reports/task1_optuna_results.json", help="Optuna JSON report used for final training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of final-training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for final training.")
    args = parser.parse_args()

    if args.final:
        run_final_training(
            config_path=args.config,
            epochs=args.epochs,
            image_size=args.image_size,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            seed=args.seed,
        )
    elif args.quick_check:
        run_quick_check()
    else:
        run_optuna(
            n_trials=args.trials,
            storage=args.storage,
            output=args.output,
            image_size=args.image_size,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
