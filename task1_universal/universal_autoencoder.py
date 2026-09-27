from __future__ import annotations

import random
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.corruption_pipeline import apply_corruption, sample_random_corruption


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class UniversalAutoencoder(nn.Module):
    """Convolutional encoder-decoder with a compressed latent bottleneck."""

    def __init__(self, base_channels: int = 32, latent_dim: int = 128, dropout: float = 0.1, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels, drop=dropout),
            ConvBlock(base_channels, base_channels * 2, stride=2, drop=dropout),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2, drop=dropout),
            ConvBlock(base_channels * 4, base_channels * 8, stride=2, drop=dropout),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.bottleneck(z)
        restored = self.decoder(z)
        if restored.shape[-2:] != x.shape[-2:]:
            restored = F.interpolate(restored, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return restored


class UniversalRestorationDataset(Dataset):
    """Loads clean images from the shared Oxford-IIIT Pet split and injects random corruption each time."""

    def __init__(
        self,
        split_file: str | Path,
        data_root: str | Path,
        image_size: int = 128,
        train: bool = True,
        corruption_manifest: str | Path | None = None,
    ):
        self.train = train
        self.image_size = image_size
        self.data_root = Path(data_root)
        self.samples: List[Tuple[str, int, str | None, Dict[str, Any]]] = []
        manifest_records: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        if corruption_manifest is not None:
            with open(corruption_manifest, "r", encoding="utf-8") as manifest_file:
                for record in json.load(manifest_file):
                    image_path = record["image_path"]
                    manifest_records[image_path] = []
                    for corruption_name, config in record["corruptions"].items():
                        settings = dict(config)
                        settings.pop("type", None)
                        settings.pop("severity", None)
                        settings.pop("seed", None)
                        manifest_records[image_path].append((corruption_name, settings))

        with open(split_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                image_path, label = line.split("\t")
                full_path = self.data_root / image_path
                if full_path.exists():
                    if image_path in manifest_records:
                        for corruption_name, settings in manifest_records[image_path]:
                            self.samples.append((str(full_path), int(label), corruption_name, settings))
                    else:
                        self.samples.append((str(full_path), int(label), None, {}))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path, _, corruption_name, settings = self.samples[idx]
        clean = Image.open(image_path).convert("RGB")
        clean_tensor = self.transform(clean)

        if self.train or corruption_name is not None:
            rng = random.Random()
            corruption = sample_random_corruption(rng) if self.train else {"type": corruption_name, "settings": settings}
            np_image = np.array(clean.resize((self.image_size, self.image_size)))
            corrupted_np = apply_corruption(np_image, corruption["type"], corruption["settings"])
            corrupted_tensor = torch.from_numpy(corrupted_np.transpose(2, 0, 1)).float() / 255.0
        else:
            corrupted_tensor = clean_tensor.clone()

        return {"clean": clean_tensor, "corrupted": corrupted_tensor}


def ssim_score(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Return differentiable mean SSIM for an NCHW batch in the [0, 1] range."""
    if pred.shape != target.shape:
        raise ValueError(f"SSIM inputs must have the same shape, got {pred.shape} and {target.shape}.")
    if pred.ndim != 4:
        raise ValueError(f"SSIM expects NCHW tensors, got {pred.ndim} dimensions.")
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("SSIM window_size must be an odd integer >= 3.")

    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    padding = window_size // 2
    channels = pred.shape[1]
    kernel = torch.ones((channels, 1, window_size, window_size), device=pred.device, dtype=pred.dtype)
    kernel = kernel / float(window_size * window_size)

    mean_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mean_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mean_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels)
    mean_target_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels)
    mean_product = F.conv2d(pred * target, kernel, padding=padding, groups=channels)

    variance_pred = (mean_pred_sq - mean_pred * mean_pred).clamp_min(0.0)
    variance_target = (mean_target_sq - mean_target * mean_target).clamp_min(0.0)
    covariance = mean_product - mean_pred * mean_target

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_pred * mean_target + c1) * (2.0 * covariance + c2)
    denominator = (mean_pred * mean_pred + mean_target * mean_target + c1) * (variance_pred + variance_target + c2)
    return (numerator / denominator.clamp_min(torch.finfo(pred.dtype).eps)).mean()


def universal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1.0 - ssim_score(pred, target)
    return alpha * l1 + (1.0 - alpha) * ssim_loss


def build_loaders(batch_size: int = 32, image_size: int = 128, data_root: str | Path = "data/oxford_iiit_pet") -> Tuple[DataLoader, DataLoader]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(data_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    train_split = project_root / "data" / "oxford_iiit_pet_splits" / "train.txt"
    val_split = project_root / "data" / "oxford_iiit_pet_splits" / "val.txt"

    train_dataset = UniversalRestorationDataset(train_split, dataset_root, image_size=image_size, train=True)
    val_manifest = project_root / "data" / "oxford_iiit_pet_splits" / "val_corruption_manifest.json"
    val_dataset = UniversalRestorationDataset(
        val_split,
        dataset_root,
        image_size=image_size,
        train=False,
        corruption_manifest=val_manifest,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def model_summary() -> str:
    return (
        "UniversalAutoencoder: convolutional encoder -> compressed bottleneck -> convolutional decoder. "
        "No corruption label is provided to the network; it learns a shared latent representation across clean, "
        "salt-and-pepper, Gaussian blur, and occlusion conditions."
    )
