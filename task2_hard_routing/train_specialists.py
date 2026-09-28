from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task1_universal.universal_autoencoder import ssim_score, universal_loss
from task2_hard_routing.specialist_autoencoders import SPECIALIST_TYPES, build_specialist_datasets, create_specialist


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: AdamW, device: torch.device, alpha: float, max_batches: int | None) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        corrupted = batch["corrupted"].to(device)
        clean = batch["clean"].to(device)
        optimizer.zero_grad()
        loss = universal_loss(model(corrupted), clean, alpha=alpha)
        loss.backward()
        optimizer.step()
        total += loss.item() * corrupted.size(0)
        count += corrupted.size(0)
    return total / max(count, 1)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, alpha: float, max_batches: int | None) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            corrupted = batch["corrupted"].to(device)
            clean = batch["clean"].to(device)
            restored = model(corrupted)
            size = corrupted.size(0)
            totals["loss"] += universal_loss(restored, clean, alpha=alpha).item() * size
            totals["l1"] += torch.nn.functional.l1_loss(restored, clean).item() * size
            totals["ssim"] += ssim_score(restored, clean).item() * size
            count += size
    return {name: value / max(count, 1) for name, value in totals.items()}


def train_one_specialist(
    specialist: str,
    project_root: Path,
    epochs: int = 1,
    image_size: int = 64,
    batch_size: int = 8,
    learning_rate: float = 2.0e-4,
    alpha: float = 0.8,
    latent_dim: int = 64,
    base_channels: int = 16,
    dropout: float = 0.0,
    max_train_batches: int | None = 8,
    max_val_batches: int | None = 4,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_dataset, val_dataset = build_specialist_datasets(project_root, specialist, image_size=image_size, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_specialist(specialist, base_channels=base_channels, latent_dim=latent_dim, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    checkpoint_dir = project_root / "task2_hard_routing" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, alpha, max_train_batches)
        validation = evaluate(model, val_loader, device, alpha, max_val_batches)
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in validation.items()}}
        history.append(record)
        print(json.dumps({"specialist": specialist, **record}))
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "specialist": specialist,
                    "config": {
                        "image_size": image_size,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "alpha": alpha,
                        "latent_dim": latent_dim,
                        "base_channels": base_channels,
                        "dropout": dropout,
                    },
                    "validation": validation,
                },
                checkpoint_dir / f"{specialist}_expert_best.pt",
            )

    return {"specialist": specialist, "device": str(device), "history": history, "best_validation_loss": best_loss}


def train_all_specialists(**kwargs: Any) -> dict[str, Any]:
    project_root = Path(kwargs.pop("project_root"))
    results = {}
    for specialist in SPECIALIST_TYPES:
        results[specialist] = train_one_specialist(specialist, project_root, **kwargs)
    output = project_root / "reports" / "task2_specialist_training_log.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train independent Task 2 specialist autoencoders.")
    parser.add_argument("--specialist", choices=["all", *SPECIALIST_TYPES], default="all")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=8)
    parser.add_argument("--max-val-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    settings = vars(args)
    settings.pop("specialist")
    settings["project_root"] = project_root
    if args.specialist == "all":
        train_all_specialists(**settings)
    else:
        result = train_one_specialist(args.specialist, **settings)
        report = project_root / "reports" / f"task2_{args.specialist}_expert_training_log.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
