from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW

from task2_hard_routing.corruption_classifier import CorruptionClassifier
from task2_hard_routing.train_classifier import build_classifier_loaders, evaluate_classifier, train_epoch


def train_selected_classifier(
    config_path: str | Path = "reports/task2_classifier_optuna_results.json",
    epochs: int = 1,
    max_train_batches: int | None = 1,
    max_val_batches: int | None = 1,
    output: str | Path = "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    report_file = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    params = report["best_parameters"]
    image_size = int(report.get("image_size", 64))
    batch_size = int(params["batch_size"])
    train_loader, val_loader = build_classifier_loaders(project_root, batch_size, image_size, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorruptionClassifier(channels=tuple(params["channels"]), dropout=float(params["dropout"])).to(device)
    optimizer = AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    history = []
    best_f1 = -1.0
    output_file = project_root / output if not Path(output).is_absolute() else Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, max_train_batches)
        validation = evaluate_classifier(model, val_loader, device, max_val_batches)
        record = {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        history.append(record)
        print(json.dumps(record))
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": params,
                    "image_size": image_size,
                    "validation": validation,
                },
                output_file,
            )
    return {"selected_parameters": params, "history": history, "checkpoint": str(output_file.relative_to(project_root))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the classifier from the selected Optuna configuration.")
    parser.add_argument("--config", default="reports/task2_classifier_optuna_results.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--output", default="task2_hard_routing/checkpoints/corruption_classifier_best.pt")
    args = parser.parse_args()
    result = train_selected_classifier(
        config_path=args.config,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        output=args.output,
    )
    report_path = Path(__file__).resolve().parents[1] / "reports" / "task2_selected_classifier_training.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
