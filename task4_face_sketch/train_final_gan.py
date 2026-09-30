from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from task4_face_sketch.conditional_gan import (
    ConditionalPatchGANDiscriminator,
    ConditionalUNetGenerator,
    discriminator_loss,
    generator_loss,
)


class FS2KPairedDataset(Dataset):
    def __init__(self, manifest: str | Path, split_dir: str | Path, image_size: int = 128):
        self.records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.split_dir = Path(split_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        photo = Image.open(self.split_dir / record["photo"]).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        sketch = Image.open(self.split_dir / record["sketch"]).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        photo_tensor = torch.from_numpy(np.asarray(photo, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        sketch_tensor = torch.from_numpy(np.asarray(sketch, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0)
        return {
            "photo": photo_tensor,
            "sketch": sketch_tensor,
            "style": torch.tensor(int(record["style"]), dtype=torch.long),
        }


def _save_generated_samples(generator: ConditionalUNetGenerator, batch: dict[str, torch.Tensor], output_dir: Path, epoch: int, device: torch.device) -> None:
    generator.eval()
    with torch.no_grad():
        generated = generator(batch["photo"].to(device), batch["style"].to(device)).cpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(3, generated.size(0))):
        panels = [batch["photo"][index], batch["sketch"][index], generated[index]]
        figure, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, panel, title in zip(axes, panels, ("Photo", "Target sketch", "Generated sketch")):
            image = ((panel.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(f"Style {int(batch['style'][index])} - epoch {epoch}")
        figure.tight_layout()
        figure.savefig(output_dir / f"epoch_{epoch:03d}_sample_{index:02d}.png", dpi=140)
        plt.close(figure)
    generator.train()


def train_final_gan(
    data_root: str | Path = "data/fs2k_prepared",
    epochs: int = 1,
    batch_size: int = 4,
    image_size: int = 128,
    generator_learning_rate: float = 2e-4,
    discriminator_learning_rate: float = 2e-4,
    lambda_rec: float = 100.0,
    max_train_batches: int | None = 1,
    max_val_batches: int | None = 1,
    seed: int = 42,
) -> dict[str, Any]:
    if lambda_rec < 0:
        raise ValueError("lambda_rec must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    project_root = Path(__file__).resolve().parents[1]
    prepared_root = project_root / data_root if not Path(data_root).is_absolute() else Path(data_root)
    train_dataset = FS2KPairedDataset(prepared_root / "train.json", prepared_root / "train", image_size)
    validation_dataset = FS2KPairedDataset(prepared_root / "validation.json", prepared_root / "validation", image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ConditionalUNetGenerator().to(device)
    discriminator = ConditionalPatchGANDiscriminator().to(device)
    generator_optimizer = Adam(generator.parameters(), lr=generator_learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.999))
    checkpoint_dir = project_root / "task4_face_sketch" / "checkpoints"
    sample_dir = project_root / "reports" / "task4_generated_validation_samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation_l1 = float("inf")
    sample_batch = next(iter(validation_loader))

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        totals = {"d_real_loss": 0.0, "d_fake_loss": 0.0, "g_adversarial_loss": 0.0, "g_reconstruction_loss": 0.0}
        count = 0
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            photo, target, style = batch["photo"].to(device), batch["sketch"].to(device), batch["style"].to(device)
            fake = generator(photo, style)
            real_logits = discriminator(photo, target, style)
            fake_logits = discriminator(photo, fake.detach(), style)
            d_real = torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            d_fake = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            discriminator_optimizer.zero_grad()
            (0.5 * (d_real + d_fake)).backward()
            discriminator_optimizer.step()

            fake_logits_for_generator = discriminator(photo, fake, style)
            g_adv = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits_for_generator, torch.ones_like(fake_logits_for_generator))
            g_rec = torch.nn.functional.l1_loss(fake, target)
            g_total = g_adv + lambda_rec * g_rec
            generator_optimizer.zero_grad()
            g_total.backward()
            generator_optimizer.step()
            size = style.size(0)
            totals["d_real_loss"] += d_real.item() * size
            totals["d_fake_loss"] += d_fake.item() * size
            totals["g_adversarial_loss"] += g_adv.item() * size
            totals["g_reconstruction_loss"] += g_rec.item() * size
            count += size
        train_metrics = {name: value / max(count, 1) for name, value in totals.items()}

        generator.eval()
        validation_totals = {"l1": 0.0, "adversarial": 0.0, "d_real": 0.0, "d_fake": 0.0}
        validation_count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if max_val_batches is not None and batch_index >= max_val_batches:
                    break
                photo, target, style = batch["photo"].to(device), batch["sketch"].to(device), batch["style"].to(device)
                fake = generator(photo, style)
                real_logits = discriminator(photo, target, style)
                fake_logits = discriminator(photo, fake, style)
                size = style.size(0)
                validation_totals["l1"] += torch.nn.functional.l1_loss(fake, target).item() * size
                validation_totals["adversarial"] += torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits)).item() * size
                validation_totals["d_real"] += torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)).item() * size
                validation_totals["d_fake"] += torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)).item() * size
                validation_count += size
        validation_metrics = {f"val_{name}": value / max(validation_count, 1) for name, value in validation_totals.items()}
        record = {"epoch": epoch, **train_metrics, **validation_metrics}
        history.append(record)
        print(json.dumps(record))
        _save_generated_samples(generator, sample_batch, sample_dir, epoch, device)
        if validation_metrics["val_l1"] < best_validation_l1:
            best_validation_l1 = validation_metrics["val_l1"]
            torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "config": {"image_size": image_size, "lambda_rec": lambda_rec}, "history": history}, checkpoint_dir / "conditional_gan_best.pt")

    log = {
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "generator_learning_rate": generator_learning_rate,
        "discriminator_learning_rate": discriminator_learning_rate,
        "lambda_rec": lambda_rec,
        "history": history,
        "generated_validation_samples": "reports/task4_generated_validation_samples",
        "checkpoint": "task4_face_sketch/checkpoints/conditional_gan_best.pt",
    }
    log_path = project_root / "reports" / "task4_final_gan_training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final conditional FS2K GAN.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lambda-rec", type=float, default=100.0)
    parser.add_argument("--max-train-batches", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int, default=1)
    args = parser.parse_args()
    train_final_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lambda_rec=args.lambda_rec,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
