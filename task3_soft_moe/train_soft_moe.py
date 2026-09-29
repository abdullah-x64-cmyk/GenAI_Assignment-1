from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task1_universal.universal_autoencoder import ssim_score, universal_loss
from task2_hard_routing.corruption_classifier import BalancedBatchSampler, BalancedCorruptionDataset
from task3_soft_moe.soft_moe import load_soft_moe
from torchvision import transforms


class SoftMoETrainingDataset(BalancedCorruptionDataset):
    """Balanced runtime-corruption data with clean targets and class labels."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        corrupted, class_id = super().__getitem__(index)
        image_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        clean = self.transform(image)
        return {"corrupted": corrupted, "clean": clean, "label": torch.tensor(class_id, dtype=torch.long)}


def routing_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Penalize batch-average routing weights that drift from a uniform mixture."""
    target = torch.full((weights.size(1),), 1.0 / weights.size(1), device=weights.device, dtype=weights.dtype)
    return torch.mean((weights.mean(dim=0) - target) ** 2)


def joint_loss(
    result: dict[str, torch.Tensor],
    clean: torch.Tensor,
    labels: torch.Tensor,
    lambda_rec: float,
    lambda_ssim: float,
    lambda_cls: float,
    lambda_bal: float,
) -> dict[str, torch.Tensor]:
    l1 = torch.nn.functional.l1_loss(result["restored"], clean)
    ssim_loss = 1.0 - ssim_score(result["restored"], clean)
    classification = torch.nn.functional.cross_entropy(result["gate_logits"], labels)
    balance = routing_balance_loss(result["weights"])
    total = lambda_rec * l1 + lambda_ssim * ssim_loss + lambda_cls * classification + lambda_bal * balance
    return {"loss": total, "l1": l1, "ssim_loss": ssim_loss, "classification": classification, "balance": balance}


def build_training_loader(project_root: Path, batch_size: int, image_size: int, seed: int) -> DataLoader:
    dataset = SoftMoETrainingDataset(
        project_root / "data" / "oxford_iiit_pet_splits" / "train.txt",
        project_root / "data" / "oxford_iiit_pet",
        image_size=image_size,
        seed=seed,
    )
    sampler = BalancedBatchSampler(dataset, batch_size=batch_size, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0)


def set_expert_requires_grad(model: torch.nn.Module, enabled: bool) -> None:
    for expert in model.experts.values():
        for parameter in expert.parameters():
            parameter.requires_grad = enabled
        expert.train(enabled)
    for parameter in model.gate.parameters():
        parameter.requires_grad = True
    model.gate.train(True)


def train_stage(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    lambda_rec: float,
    lambda_ssim: float,
    lambda_cls: float,
    lambda_bal: float,
    max_batches: int | None,
    include_classification: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    totals = {"l1": 0.0, "ssim_loss": 0.0, "classification": 0.0, "balance": 0.0}
    total_classification = 0.0
    count = 0
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        corrupted = batch["corrupted"].to(device)
        clean = batch["clean"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        result = model(corrupted)
        terms = joint_loss(result, clean, labels, lambda_rec, lambda_ssim, lambda_cls if include_classification else 0.0, lambda_bal)
        loss = terms["loss"]
        loss.backward()
        optimizer.step()
        size = labels.size(0)
        total_loss += loss.item() * size
        for term_name in totals:
            totals[term_name] += terms[term_name].item() * size
        count += size
    return {
        "loss": total_loss / max(count, 1),
        **{term_name: value / max(count, 1) for term_name, value in totals.items()},
    }


def train_soft_moe(
    classifier_checkpoint: str | Path = "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
    expert_checkpoints: dict[str, str | Path] | None = None,
    warmup_epochs: int = 1,
    joint_epochs: int = 1,
    warmup_learning_rate: float = 1e-4,
    joint_learning_rate: float = 2e-5,
    batch_size: int = 8,
    image_size: int = 64,
    temperature: float = 1.0,
    lambda_rec: float = 1.0,
    lambda_ssim: float = 1.0,
    lambda_cls: float = 0.1,
    lambda_bal: float = 0.01,
    max_batches: int | None = 1,
    seed: int = 42,
    output: str | Path = "task3_soft_moe/checkpoints/soft_moe_best.pt",
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project_root = Path(__file__).resolve().parents[1]
    expert_checkpoints = expert_checkpoints or {
        "salt": "task2_hard_routing/checkpoints/salt_expert_best.pt",
        "blur": "task2_hard_routing/checkpoints/blur_expert_best.pt",
        "occlusion": "task2_hard_routing/checkpoints/occlusion_expert_best.pt",
    }
    model = load_soft_moe(
        classifier_checkpoint,
        expert_checkpoints,
        project_root=project_root,
        temperature=temperature,
        device=torch.device("cpu"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = build_training_loader(project_root, batch_size, image_size, seed)
    history: list[dict[str, Any]] = []

    set_expert_requires_grad(model, enabled=False)
    gate_optimizer = AdamW(model.gate.parameters(), lr=warmup_learning_rate)
    for epoch in range(1, warmup_epochs + 1):
        metrics = train_stage(model, loader, gate_optimizer, device, lambda_rec, lambda_ssim, lambda_cls, lambda_bal, max_batches, include_classification=True)
        history.append({"stage": "gate_warmup", "epoch": epoch, **metrics, "experts_frozen": True})

    set_expert_requires_grad(model, enabled=True)
    joint_optimizer = AdamW(model.parameters(), lr=joint_learning_rate)
    start_epoch = warmup_epochs
    for offset in range(1, joint_epochs + 1):
        metrics = train_stage(model, loader, joint_optimizer, device, lambda_rec, lambda_ssim, lambda_cls, lambda_bal, max_batches, include_classification=True)
        history.append({"stage": "joint_finetune", "epoch": start_epoch + offset, **metrics, "experts_frozen": False})

    output_file = project_root / output if not Path(output).is_absolute() else Path(output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "temperature": temperature,
            "lambda_rec": lambda_rec,
            "lambda_ssim": lambda_ssim,
            "lambda_cls": lambda_cls,
            "lambda_bal": lambda_bal,
            "batch_size": batch_size,
            "image_size": image_size,
            "warmup_learning_rate": warmup_learning_rate,
            "joint_learning_rate": joint_learning_rate,
            "warmup_epochs": warmup_epochs,
            "joint_epochs": joint_epochs,
        },
        "initialization": "Task 2 classifier gate and Task 2 Salt, Blur, Occlusion experts",
        "history": history,
    }
    torch.save(checkpoint, output_file)
    log_file = project_root / "reports" / "task3_soft_moe_training_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps({"checkpoint": str(output_file.relative_to(project_root)), "history": history}, indent=2), encoding="utf-8")
    return {"checkpoint": str(output_file.relative_to(project_root)), "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up and jointly fine-tune the Task 3 soft MoE.")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--joint-epochs", type=int, default=1)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.1)
    parser.add_argument("--lambda-bal", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()
    result = train_soft_moe(
        warmup_epochs=args.warmup_epochs,
        joint_epochs=args.joint_epochs,
        warmup_learning_rate=args.warmup_learning_rate,
        joint_learning_rate=args.joint_learning_rate,
        batch_size=args.batch_size,
        image_size=args.image_size,
        temperature=args.temperature,
        lambda_rec=args.lambda_rec,
        lambda_ssim=args.lambda_ssim,
        lambda_cls=args.lambda_cls,
        lambda_bal=args.lambda_bal,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, indent=2))
