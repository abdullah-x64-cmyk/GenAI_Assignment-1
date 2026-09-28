from __future__ import annotations

import random
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from torchvision import transforms

from data.corruption_pipeline import CorruptionType, apply_corruption


class CorruptionClass(IntEnum):
    CLEAN = 0
    SALT_AND_PEPPER = 1
    GAUSSIAN_BLUR = 2
    OCCLUSION = 3


CLASS_TO_CORRUPTION = {
    CorruptionClass.CLEAN: CorruptionType.CLEAN,
    CorruptionClass.SALT_AND_PEPPER: CorruptionType.SALT_AND_PEPPER,
    CorruptionClass.GAUSSIAN_BLUR: CorruptionType.GAUSSIAN_BLUR,
    CorruptionClass.OCCLUSION: CorruptionType.RECTANGULAR_OCCLUSION,
}

CLASS_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]


class BalancedCorruptionDataset(Dataset):
    """Builds one runtime-corrupted sample for every image/class pair."""

    def __init__(
        self,
        split_file: str | Path,
        data_root: str | Path,
        image_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.seed = seed
        self.samples: List[Tuple[str, int]] = []
        with open(split_file, "r", encoding="utf-8") as split_handle:
            for line in split_handle:
                line = line.strip()
                if line:
                    image_path, _ = line.split("\t")
                    full_path = self.data_root / image_path
                    if full_path.exists():
                        for class_id in range(len(CLASS_NAMES)):
                            self.samples.append((str(full_path), class_id))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, class_id = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image.resize((self.image_size, self.image_size)), dtype=np.uint8)
        settings = self._settings_for(index, class_id, image_array.shape)
        np.random.seed(self.seed + index)
        corrupted = apply_corruption(image_array, CLASS_TO_CORRUPTION[CorruptionClass(class_id)], settings)
        return self.transform(Image.fromarray(corrupted)), class_id

    def _settings_for(self, index: int, class_id: int, shape: Sequence[int]) -> Dict[str, Any]:
        rng = random.Random(self.seed + index * 1009 + class_id)
        if class_id == CorruptionClass.CLEAN:
            return {}
        if class_id == CorruptionClass.SALT_AND_PEPPER:
            return {"probability": rng.uniform(0.02, 0.15)}
        if class_id == CorruptionClass.GAUSSIAN_BLUR:
            return {"kernel_size": rng.choice([3, 5, 7]), "sigma": rng.uniform(0.5, 2.5)}
        return {"num_masks": rng.randint(1, 3), "coverage": rng.uniform(0.10, 0.35)}

    def class_indices(self) -> Dict[int, List[int]]:
        indices: Dict[int, List[int]] = {class_id: [] for class_id in range(len(CLASS_NAMES))}
        for index, (_, class_id) in enumerate(self.samples):
            indices[class_id].append(index)
        return indices


class BalancedBatchSampler(BatchSampler):
    """Yields exactly equal class counts in every full batch."""

    def __init__(self, dataset: BalancedCorruptionDataset, batch_size: int = 32, seed: int = 42) -> None:
        if batch_size < len(CLASS_NAMES) or batch_size % len(CLASS_NAMES) != 0:
            raise ValueError("batch_size must be divisible by 4 and at least 4.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.per_class = batch_size // len(CLASS_NAMES)
        self.seed = seed
        self.epoch = 0
        self.indices_by_class = dataset.class_indices()
        if any(not indices for indices in self.indices_by_class.values()):
            raise ValueError("Every corruption class must have at least one sample.")
        self.batches_per_epoch = len(dataset) // batch_size

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {}
        for class_id, indices in self.indices_by_class.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pools[class_id] = shuffled
        for batch_number in range(self.batches_per_epoch):
            batch: List[int] = []
            for class_id in range(len(CLASS_NAMES)):
                start = batch_number * self.per_class
                selected = pools[class_id][start:start + self.per_class]
                if len(selected) < self.per_class:
                    selected += rng.choices(self.indices_by_class[class_id], k=self.per_class - len(selected))
                batch.extend(selected)
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return self.batches_per_epoch


class CorruptionClassifier(nn.Module):
    def __init__(self, channels: Tuple[int, int, int] = (32, 64, 128), dropout: float = 0.2, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_balanced_classifier_loader(
    split_file: str | Path,
    data_root: str | Path,
    batch_size: int = 32,
    image_size: int = 128,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[BalancedCorruptionDataset, DataLoader]:
    dataset = BalancedCorruptionDataset(split_file, data_root, image_size=image_size, seed=seed)
    batch_sampler: Sampler[List[int]] = BalancedBatchSampler(dataset, batch_size=batch_size, seed=seed)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    return dataset, loader
