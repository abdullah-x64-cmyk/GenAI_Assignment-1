from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.corruption_pipeline import CorruptionType, apply_corruption
from task1_universal.universal_autoencoder import UniversalAutoencoder


SPECIALIST_TYPES = {
    "salt": CorruptionType.SALT_AND_PEPPER,
    "blur": CorruptionType.GAUSSIAN_BLUR,
    "occlusion": CorruptionType.RECTANGULAR_OCCLUSION,
}


class SpecialistRestorationDataset(Dataset):
    """Pairs clean targets with one specific corruption type."""

    def __init__(
        self,
        split_file: str | Path,
        data_root: str | Path,
        specialist: str,
        image_size: int = 64,
        train: bool = True,
        manifest_file: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        if specialist not in SPECIALIST_TYPES:
            raise ValueError(f"specialist must be one of {sorted(SPECIALIST_TYPES)}, got {specialist}")
        self.specialist = specialist
        self.corruption_type = SPECIALIST_TYPES[specialist]
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.train = train
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.samples: List[Tuple[str, int]] = []
        with open(split_file, "r", encoding="utf-8") as split_handle:
            for line in split_handle:
                line = line.strip()
                if line:
                    image_path, label = line.split("\t")
                    full_path = self.data_root / image_path
                    if full_path.exists():
                        self.samples.append((str(full_path), int(label)))

        self.manifest_settings: Dict[str, Dict[str, Any]] = {}
        if manifest_file is not None:
            records = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
            for record in records:
                image_path = record["image_path"]
                self.manifest_settings[image_path] = {}
                for severity in ("low", "medium", "high"):
                    config = dict(record["severity_manifest"][severity][self.corruption_type.value])
                    config.pop("type", None)
                    config.pop("severity", None)
                    self.manifest_settings[image_path][severity] = config

    def __len__(self) -> int:
        return len(self.samples) if self.train else len(self.samples) * 3

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        base_index = index if self.train else index // 3
        severity = None if self.train else ("low", "medium", "high")[index % 3]
        image_path, _ = self.samples[base_index]
        clean_image = Image.open(image_path).convert("RGB")
        clean_array = np.asarray(clean_image.resize((self.image_size, self.image_size)), dtype=np.uint8)
        if self.train:
            rng = random.Random(self.seed + base_index * 1009)
            settings = self._random_settings(rng)
            corruption_seed = self.seed + index
        else:
            relative_path = str(Path(image_path).relative_to(self.data_root)).replace("/", "\\")
            settings = dict(self.manifest_settings[relative_path][severity])
            corruption_seed = int(settings.pop("seed", self.seed + index))
        np.random.seed(corruption_seed)
        corrupted_array = apply_corruption(clean_array, self.corruption_type, settings)
        return {
            "clean": self.transform(Image.fromarray(clean_array)),
            "corrupted": self.transform(Image.fromarray(corrupted_array)),
            "specialist": self.specialist,
            "severity": severity or "random",
        }

    def _random_settings(self, rng: random.Random) -> Dict[str, Any]:
        if self.corruption_type == CorruptionType.SALT_AND_PEPPER:
            return {"probability": rng.uniform(0.02, 0.15)}
        if self.corruption_type == CorruptionType.GAUSSIAN_BLUR:
            return {"kernel_size": rng.choice([3, 5, 7]), "sigma": rng.uniform(0.5, 2.5)}
        return {"num_masks": rng.randint(1, 3), "coverage": rng.uniform(0.10, 0.35)}


def build_specialist_datasets(
    project_root: str | Path,
    specialist: str,
    image_size: int = 64,
    seed: int = 42,
) -> Tuple[SpecialistRestorationDataset, SpecialistRestorationDataset]:
    root = Path(project_root)
    data_root = root / "data" / "oxford_iiit_pet"
    split_root = root / "data" / "oxford_iiit_pet_splits"
    train_dataset = SpecialistRestorationDataset(
        split_root / "train.txt", data_root, specialist, image_size=image_size, train=True, seed=seed
    )
    val_dataset = SpecialistRestorationDataset(
        split_root / "val.txt",
        data_root,
        specialist,
        image_size=image_size,
        train=False,
        manifest_file=split_root / "val_corruption_manifest.json",
        seed=seed,
    )
    return train_dataset, val_dataset


def create_specialist(specialist: str, base_channels: int = 16, latent_dim: int = 64, dropout: float = 0.0) -> UniversalAutoencoder:
    if specialist not in SPECIALIST_TYPES:
        raise ValueError(f"specialist must be one of {sorted(SPECIALIST_TYPES)}, got {specialist}")
    return UniversalAutoencoder(base_channels=base_channels, latent_dim=latent_dim, dropout=dropout)
