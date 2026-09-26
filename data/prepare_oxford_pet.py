from __future__ import annotations

import json
import random
from pathlib import Path

from torchvision.datasets import OxfordIIITPet


DATA_ROOT = Path(__file__).resolve().parent / "oxford_iiit_pet"
SPLIT_DIR = Path(__file__).resolve().parent / "oxford_iiit_pet_splits"


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    """Download the Oxford-IIIT Pet trainval/test data if it is not already present."""
    OxfordIIITPet(root=str(DATA_ROOT), split="trainval", download=True)
    OxfordIIITPet(root=str(DATA_ROOT), split="test", download=True)


def write_split_manifest(split_name: str, samples: list[tuple[str, int]]) -> None:
    split_path = SPLIT_DIR / f"{split_name}.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for img_path, target in samples:
            f.write(f"{img_path}\t{target}\n")

    manifest = {
        "split": split_name,
        "count": len(samples),
        "entries": [
            {"image_path": img_path, "target": target}
            for img_path, target in samples
        ],
    }
    manifest_path = SPLIT_DIR / f"{split_name}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _dataset_records(dataset: OxfordIIITPet) -> list[tuple[str, int]]:
    """Compatibility helper for torchvision versions that expose _images/_labels instead of .samples."""
    if hasattr(dataset, "samples"):
        return [(str(Path(img_path).relative_to(DATA_ROOT)), int(target)) for img_path, target in dataset.samples]

    images = getattr(dataset, "_images", [])
    labels = getattr(dataset, "_labels", [])
    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch: {len(images)} images vs {len(labels)} labels")

    return [
        (str(Path(str(img_path)).relative_to(DATA_ROOT)), int(label))
        for img_path, label in zip(images, labels)
    ]


def build_shared_split() -> None:
    trainval = OxfordIIITPet(root=str(DATA_ROOT), split="trainval")
    all_samples = _dataset_records(trainval)

    rng = random.Random(42)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)

    train_count = int(0.8 * len(all_samples))
    train_idx = sorted(indices[:train_count])
    val_idx = sorted(indices[train_count:])

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    # Save relative paths for use across Tasks 1-3.
    train_records = list(train_samples)
    val_records = list(val_samples)

    write_split_manifest("train", train_records)
    write_split_manifest("val", val_records)

    # The official test set remains untouched and is saved in full separately.
    test_data = OxfordIIITPet(root=str(DATA_ROOT), split="test")
    test_records = _dataset_records(test_data)
    write_split_manifest("test", test_records)

    summary = {
        "dataset": "Oxford-IIIT Pet",
        "seed": 42,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "train_fraction": 0.8,
        "val_fraction": 0.2,
        "same_split_for_tasks_1_2_3": True,
        "data_root": str(DATA_ROOT),
        "split_dir": str(SPLIT_DIR),
    }

    summary_path = SPLIT_DIR / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ensure_dirs()
    print("Downloading Oxford-IIIT Pet dataset...")
    download_dataset()
    print("Building shared 80/20 train/val split with seed 42...")
    build_shared_split()
    print(f"Split files saved to: {SPLIT_DIR}")
