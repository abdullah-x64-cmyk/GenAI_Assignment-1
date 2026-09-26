from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from data.corruption_pipeline import (
    CorruptionType,
    apply_corruption,
    validate_corruption_settings,
)


DATA_ROOT = Path(__file__).resolve().parent / "oxford_iiit_pet"
SPLIT_DIR = Path(__file__).resolve().parent / "oxford_iiit_pet_splits"


def _load_split(split_name: str) -> List[Tuple[str, int]]:
    split_file = SPLIT_DIR / f"{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    items: List[Tuple[str, int]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            image_path, label = line.strip().split("\t")
            items.append((image_path, int(label)))
    return items


def _random_seed_for_index(idx: int, prefix: str = "") -> int:
    return (idx + 1) * 9973 + len(prefix) * 101


def _build_salt_and_pepper_config(seed: int, severity: str) -> Dict[str, Any]:
    if severity == "low":
        probability = 0.03
    elif severity == "medium":
        probability = 0.08
    elif severity == "high":
        probability = 0.15
    else:
        raise ValueError(f"Unknown severity: {severity}")
    validate_corruption_settings(CorruptionType.SALT_AND_PEPPER, {"probability": probability})
    return {"type": CorruptionType.SALT_AND_PEPPER.value, "severity": severity, "probability": probability, "seed": seed}


def _build_gaussian_blur_config(seed: int, severity: str) -> Dict[str, Any]:
    mapping = {
        "low": {"kernel_size": 3, "sigma": 0.7},
        "medium": {"kernel_size": 5, "sigma": 1.5},
        "high": {"kernel_size": 7, "sigma": 2.5},
    }
    if severity not in mapping:
        raise ValueError(f"Unknown severity: {severity}")
    cfg = mapping[severity]
    validate_corruption_settings(CorruptionType.GAUSSIAN_BLUR, cfg)
    cfg = {"type": CorruptionType.GAUSSIAN_BLUR.value, "severity": severity, "kernel_size": cfg["kernel_size"], "sigma": cfg["sigma"], "seed": seed}
    return cfg


def _build_occlusion_config(seed: int, severity: str, image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    mapping = {
        "low": {"num_masks": 1, "coverage": 0.10},
        "medium": {"num_masks": 2, "coverage": 0.20},
        "high": {"num_masks": 3, "coverage": 0.35},
    }
    if severity not in mapping:
        raise ValueError(f"Unknown severity: {severity}")
    cfg = mapping[severity]
    validate_corruption_settings(CorruptionType.RECTANGULAR_OCCLUSION, cfg)

    rng = random.Random(seed)
    height, width = image_shape[:2]
    coords: List[Tuple[int, int, int, int]] = []
    target_area = int(height * width * cfg["coverage"])

    for _ in range(cfg["num_masks"]):
        max_w = max(4, width // 4)
        max_h = max(4, height // 4)
        w = rng.randint(max(4, min(max_w, max(8, int(math.sqrt(target_area * 0.5))))), max_w)
        h = rng.randint(max(4, min(max_h, max(8, int(math.sqrt(target_area * 0.5))))), max_h)
        x = rng.randint(0, max(0, width - w))
        y = rng.randint(0, max(0, height - h))
        coords.append((x, y, x + w, y + h))

    out = {
        "type": CorruptionType.RECTANGULAR_OCCLUSION.value,
        "severity": severity,
        "num_masks": cfg["num_masks"],
        "coverage": cfg["coverage"],
        "mask_coords": coords,
        "seed": seed,
    }
    return out


def _safe_image_shape(image_path: str) -> Tuple[int, int, int]:
    img = Image.open(DATA_ROOT / image_path)
    arr = np.array(img.convert("RGB"))
    return arr.shape


def _deterministic_manifest_for_split(split_name: str) -> List[Dict[str, Any]]:
    items = _load_split(split_name)
    manifest: List[Dict[str, Any]] = []

    for idx, (image_path, target) in enumerate(items):
        seed = _random_seed_for_index(idx, split_name)
        image_shape = _safe_image_shape(image_path)
        entry = {
            "index": idx,
            "image_path": image_path,
            "label": target,
            "corruptions": {
                "clean": {"type": CorruptionType.CLEAN.value, "seed": seed},
                "salt_and_pepper": _build_salt_and_pepper_config(seed, "low"),
                "gaussian_blur": _build_gaussian_blur_config(seed, "medium"),
                "rectangular_occlusion": _build_occlusion_config(seed, "high", image_shape),
            },
        }

        # Rebuild exact low/medium/high variants for final evaluation on every image.
        entry["severity_manifest"] = {
            "low": {
                "salt_and_pepper": _build_salt_and_pepper_config(seed + 1, "low"),
                "gaussian_blur": _build_gaussian_blur_config(seed + 2, "low"),
                "rectangular_occlusion": _build_occlusion_config(seed + 3, "low", image_shape),
            },
            "medium": {
                "salt_and_pepper": _build_salt_and_pepper_config(seed + 4, "medium"),
                "gaussian_blur": _build_gaussian_blur_config(seed + 5, "medium"),
                "rectangular_occlusion": _build_occlusion_config(seed + 6, "medium", image_shape),
            },
            "high": {
                "salt_and_pepper": _build_salt_and_pepper_config(seed + 7, "high"),
                "gaussian_blur": _build_gaussian_blur_config(seed + 8, "high"),
                "rectangular_occlusion": _build_occlusion_config(seed + 9, "high", image_shape),
            },
        }
        manifest.append(entry)

    return manifest


def _write_manifest(split_name: str, manifest: List[Dict[str, Any]]) -> None:
    manifest_path = SPLIT_DIR / f"{split_name}_corruption_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest)} entries to {manifest_path}")


def build_deterministic_manifests() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["val", "test"]:
        manifest = _deterministic_manifest_for_split(split_name)
        _write_manifest(split_name, manifest)


if __name__ == "__main__":
    import math
    build_deterministic_manifests()
