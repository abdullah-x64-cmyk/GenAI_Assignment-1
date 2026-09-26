from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CorruptionType(str, Enum):
    CLEAN = "clean"
    SALT_AND_PEPPER = "salt_and_pepper"
    GAUSSIAN_BLUR = "gaussian_blur"
    RECTANGULAR_OCCLUSION = "rectangular_occlusion"


@dataclass
class CorruptionSample:
    type: CorruptionType
    settings: Dict[str, Any]


def _as_uint8_image(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.uint8, copy=False)


def validate_corruption_settings(corruption_type: CorruptionType, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize settings for a specific corruption type."""
    normalized = dict(settings)

    if corruption_type == CorruptionType.CLEAN:
        return normalized

    if corruption_type == CorruptionType.SALT_AND_PEPPER:
        p = float(normalized.get("probability", 0.08))
        if not (0.02 <= p <= 0.15):
            raise ValueError(f"Salt-and-pepper probability must be in [0.02, 0.15], got {p}.")
        normalized["probability"] = p
        return normalized

    if corruption_type == CorruptionType.GAUSSIAN_BLUR:
        kernel_size = int(normalized.get("kernel_size", 3))
        sigma = float(normalized.get("sigma", 1.0))
        if kernel_size not in {3, 5, 7}:
            raise ValueError(f"Gaussian blur kernel size must be one of {{3, 5, 7}}, got {kernel_size}.")
        if not (0.5 <= sigma <= 2.5):
            raise ValueError(f"Gaussian blur sigma must be in [0.5, 2.5], got {sigma}.")
        normalized["kernel_size"] = kernel_size
        normalized["sigma"] = sigma
        return normalized

    if corruption_type == CorruptionType.RECTANGULAR_OCCLUSION:
        num_masks = int(normalized.get("num_masks", 1))
        coverage = float(normalized.get("coverage", 0.2))
        if not (1 <= num_masks <= 3):
            raise ValueError(f"Occlusion mask count must be between 1 and 3, got {num_masks}.")
        if not (0.10 <= coverage <= 0.35):
            raise ValueError(f"Occlusion coverage must be in [0.10, 0.35], got {coverage}.")
        normalized["num_masks"] = num_masks
        normalized["coverage"] = coverage
        return normalized

    raise ValueError(f"Unsupported corruption type: {corruption_type}")


def sample_random_corruption(rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Randomly choose one of four input conditions with equal probability."""
    if rng is None:
        rng = random.Random()

    choices = [
        CorruptionType.CLEAN,
        CorruptionType.SALT_AND_PEPPER,
        CorruptionType.GAUSSIAN_BLUR,
        CorruptionType.RECTANGULAR_OCCLUSION,
    ]
    corruption_type = rng.choice(choices)

    if corruption_type == CorruptionType.CLEAN:
        return {"type": corruption_type, "settings": {}}

    if corruption_type == CorruptionType.SALT_AND_PEPPER:
        probability = rng.uniform(0.02, 0.15)
        return {"type": corruption_type, "settings": {"probability": probability}}

    if corruption_type == CorruptionType.GAUSSIAN_BLUR:
        kernel_size = rng.choice([3, 5, 7])
        sigma = rng.uniform(0.5, 2.5)
        return {"type": corruption_type, "settings": {"kernel_size": kernel_size, "sigma": sigma}}

    num_masks = rng.randint(1, 3)
    coverage = rng.uniform(0.10, 0.35)
    return {"type": corruption_type, "settings": {"num_masks": num_masks, "coverage": coverage}}


def clean(image: Union[np.ndarray, Image.Image], settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    return _as_uint8_image(image).copy()


def salt_and_pepper(image: Union[np.ndarray, Image.Image], settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply salt-and-pepper noise with uniform probability sampled in [0.02, 0.15]."""
    arr = _as_uint8_image(image)
    settings = settings or {}
    probability = float(settings.get("probability", 0.08))
    validate_corruption_settings(CorruptionType.SALT_AND_PEPPER, {"probability": probability})

    noise = np.random.random(arr.shape[:2])
    mask = noise < probability
    salt_mask = np.random.random(arr.shape[:2]) < 0.5
    out = arr.copy()
    out[mask & salt_mask] = 255
    out[mask & ~salt_mask] = 0
    return out.astype(np.uint8)


def gaussian_blur(image: Union[np.ndarray, Image.Image], settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply Gaussian blur using a kernel in {3, 5, 7} and sigma in [0.5, 2.5]."""
    arr = _as_uint8_image(image)
    settings = settings or {}
    kernel_size = int(settings.get("kernel_size", 3))
    sigma = float(settings.get("sigma", 1.0))
    validate_corruption_settings(CorruptionType.GAUSSIAN_BLUR, {"kernel_size": kernel_size, "sigma": sigma})
    return cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def _generate_occlusion_masks(height: int, width: int, num_masks: int, coverage: float) -> List[Tuple[int, int, int, int]]:
    masks: List[Tuple[int, int, int, int]] = []
    rng = random.Random()
    min_area = int(height * width * coverage)
    max_area = int(height * width * coverage)

    for _ in range(1000):
        masks = []
        remaining = max(0, min_area)
        target_area = max_area

        for _ in range(num_masks):
            if remaining <= 0:
                break
            max_w = max(4, width)
            max_h = max(4, height)
            w = rng.randint(max(4, int(math.sqrt(target_area * 0.5))), max_w)
            h = rng.randint(max(4, int(math.sqrt(target_area * 0.5))), max_h)
            x = rng.randint(0, max(0, width - w))
            y = rng.randint(0, max(0, height - h))
            masks.append((x, y, x + w, y + h))
            remaining = max(0, remaining - w * h)

        if len(masks) == num_masks:
            break

    if not masks:
        x, y = 0, 0
        w, h = max(4, width // 4), max(4, height // 4)
        masks.append((x, y, x + w, y + h))

    return masks


def rectangular_occlusion(image: Union[np.ndarray, Image.Image], settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply 1-3 black rectangular masks whose total coverage is in [10%, 35%]."""
    arr = _as_uint8_image(image)
    settings = settings or {}
    num_masks = int(settings.get("num_masks", 1))
    coverage = float(settings.get("coverage", 0.2))
    validate_corruption_settings(CorruptionType.RECTANGULAR_OCCLUSION, {"num_masks": num_masks, "coverage": coverage})

    height, width = arr.shape[:2]
    out = arr.copy()
    mask_coords = []

    if "mask_coords" in settings and settings["mask_coords"]:
        mask_coords = settings["mask_coords"]
    else:
        mask_coords = _generate_occlusion_masks(height, width, num_masks, coverage)

    for x1, y1, x2, y2 in mask_coords:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        out[y1:y2, x1:x2] = 0

    return out.astype(np.uint8)


def apply_corruption(image: Union[np.ndarray, Image.Image], corruption_type: CorruptionType, settings: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Apply the selected corruption to a single image."""
    settings = settings or {}
    corruption_type = CorruptionType(corruption_type) if isinstance(corruption_type, str) else corruption_type
    normalized = validate_corruption_settings(corruption_type, settings)

    if corruption_type == CorruptionType.CLEAN:
        return clean(image, normalized)
    if corruption_type == CorruptionType.SALT_AND_PEPPER:
        return salt_and_pepper(image, normalized)
    if corruption_type == CorruptionType.GAUSSIAN_BLUR:
        return gaussian_blur(image, normalized)
    if corruption_type == CorruptionType.RECTANGULAR_OCCLUSION:
        return rectangular_occlusion(image, normalized)
    raise ValueError(f"Unsupported corruption type: {corruption_type}")


class RandomCorruptionDataset(Dataset):
    """Dataset wrapper that applies a random corruption to each image during loading."""

    def __init__(self, base_dataset: Dataset, image_key: str = "image", target_key: str = "target", rng: Optional[random.Random] = None):
        self.base_dataset = base_dataset
        self.image_key = image_key
        self.target_key = target_key
        self.rng = rng or random.Random(42)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        image = sample[self.image_key] if isinstance(sample, dict) else sample[0]
        target = sample[self.target_key] if isinstance(sample, dict) else sample[1]

        image_np = _as_uint8_image(image)
        corruption = sample_random_corruption(self.rng)
        corrupted = apply_corruption(image_np, corruption["type"], corruption["settings"])

        if isinstance(sample, dict):
            sample = dict(sample)
            sample["image"] = corrupted
            sample["clean_image"] = image_np.copy()
            sample["corruption_type"] = corruption["type"].value
            sample["corruption_settings"] = corruption["settings"]
            sample["target"] = target
            return sample

        return corrupted, image_np.copy(), corruption["type"].value, corruption["settings"]
