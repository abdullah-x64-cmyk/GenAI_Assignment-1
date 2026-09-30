from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHOTO_KEYS = ("photo", "photograph", "image", "input", "face", "img")
SKETCH_KEYS = ("sketch", "target", "drawing", "label")
STYLE_KEYS = ("style", "sketch_style", "style_id", "category")
SPLIT_KEYS = ("split", "subset", "partition")
EXPECTED_TOTAL = 2104
EXPECTED_STYLES = 3


def _value(record: dict[str, Any], keys: Iterable[str]) -> str | None:
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    for key in keys:
        if key in normalized and str(normalized[key]).strip():
            return str(normalized[key]).strip()
    return None


def _canonical_style(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    return value if value else "unknown"


def _parse_records(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("pairs", "annotations", "data", "records"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return []
        rows = payload
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = [part.strip() for part in line.replace(",", " ").split() if part.strip()]
            if len(parts) >= 3:
                rows.append({"photo": parts[0], "sketch": parts[1], "style": parts[2]})
            elif len(parts) == 2:
                rows.append({"photo": parts[0], "sketch": parts[1]})

    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        photo = _value(row, PHOTO_KEYS)
        sketch = _value(row, SKETCH_KEYS)
        if not photo or not sketch:
            continue
        style = _value(row, STYLE_KEYS)
        split = _value(row, SPLIT_KEYS)
        records.append({"photo": photo, "sketch": sketch, "style": _canonical_style(style or "unknown"), "split": (split or "").lower()})
    return records


def _find_metadata_files(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".tsv"}:
            lowered = path.name.lower()
            if any(token in lowered for token in ("train", "test", "split", "anno", "label", "style", "pair")):
                candidates.append(path)
    return sorted(candidates)


def _find_image_with_stem(folder: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def _parse_fs2k_annotations(annotation_file: Path, root: Path) -> list[dict[str, str]]:
    """Read official FS2K image_name/style annotations and reconstruct paired paths."""
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    records = []
    for item in payload:
        if not isinstance(item, dict) or "image_name" not in item or "style" not in item:
            continue
        image_name = str(item["image_name"]).replace("\\", "/")
        photo_folder, photo_stem = image_name.rsplit("/", maxsplit=1)
        style_id = int(item["style"])
        photo = _find_image_with_stem(root / "photo" / photo_folder, photo_stem)
        # FS2K folder numbers identify the photo source group; style is annotation metadata.
        source_group = photo_folder.removeprefix("photo")
        sketch = _find_image_with_stem(root / "sketch" / f"sketch{source_group}", f"sketch{photo_stem.removeprefix('image')}")
        if photo is None or sketch is None:
            continue
        records.append({
            "photo": str(photo),
            "sketch": str(sketch),
            "style": str(style_id),
            "split": "",
        })
    return records


def _resolve_image(root: Path, value: str) -> Path | None:
    candidate = Path(value)
    normalized_value = value.replace("/", "\\")
    options = [candidate] if candidate.is_absolute() else [root / candidate, root / candidate.as_posix(), root / normalized_value]
    for option in options:
        if option.exists() and option.suffix.lower() in IMAGE_EXTENSIONS:
            return option
    basename_matches = list(root.rglob(candidate.name)) if candidate.name else []
    return basename_matches[0] if basename_matches else None


def _extract_if_needed(source: Path, workspace: Path) -> Path:
    if source.is_dir():
        return source
    extract_root = workspace / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extract_root)
    elif source.suffix.lower() in {".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        with tarfile.open(source) as archive:
            archive.extractall(extract_root)
    else:
        raise ValueError(f"Unsupported FS2K source format: {source}")
    return extract_root


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def _discover_official_records(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    official_train = root / "anno_train.json"
    official_test = root / "anno_test.json"
    if official_train.exists() and official_test.exists():
        train_records = _parse_fs2k_annotations(official_train, root)
        test_records = _parse_fs2k_annotations(official_test, root)
        if train_records and test_records:
            return train_records, test_records

    train_records: list[dict[str, str]] = []
    test_records: list[dict[str, str]] = []
    for metadata_file in _find_metadata_files(root):
        records = _parse_records(metadata_file)
        if not records:
            continue
        relative_records = []
        for record in records:
            photo = _resolve_image(root, record["photo"])
            sketch = _resolve_image(root, record["sketch"])
            if photo is None or sketch is None:
                continue
            relative_records.append({**record, "photo": str(photo), "sketch": str(sketch)})
        if not relative_records:
            continue
        filename = metadata_file.name.lower()
        if "train" in filename:
            train_records.extend(relative_records)
        elif "test" in filename:
            test_records.extend(relative_records)
        elif any(record["split"] in {"train", "training", "test", "testing"} for record in relative_records):
            train_records.extend(record for record in relative_records if record["split"] in {"train", "training"})
            test_records.extend(record for record in relative_records if record["split"] in {"test", "testing"})

    def unique(records: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        result = []
        for record in records:
            key = (record["photo"], record["sketch"])
            if key not in seen:
                seen.add(key)
                result.append(record)
        return result

    train_records, test_records = unique(train_records), unique(test_records)
    if not train_records or not test_records:
        raise FileNotFoundError("Could not identify official FS2K train/test pair manifests with style metadata.")
    return train_records, test_records


def _stratified_split(records: list[dict[str, str]], validation_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_style: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_style[record["style"]].append(record)
    if len(by_style) != EXPECTED_STYLES:
        raise ValueError(f"Expected {EXPECTED_STYLES} sketch styles, found {sorted(by_style)}")
    validation, training = [], []
    for style, style_records in sorted(by_style.items()):
        shuffled = list(style_records)
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation.extend(shuffled[:validation_count])
        training.extend(shuffled[validation_count:])
    rng.shuffle(validation)
    rng.shuffle(training)
    return training, validation


def _write_image_pair(record: dict[str, str], destination: Path, image_size: int) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    pair_id = Path(record["photo"]).stem
    photo_name = f"{pair_id}_photo.png"
    sketch_name = f"{pair_id}_sketch.png"
    Image.open(record["photo"]).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS).save(destination / photo_name)
    Image.open(record["sketch"]).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS).save(destination / sketch_name)
    return {"pair_id": pair_id, "photo": photo_name, "sketch": sketch_name, "style": record["style"]}


def _write_split(records: list[dict[str, str]], name: str, output_root: Path, image_size: int) -> list[dict[str, str]]:
    split_root = output_root / name
    prepared = [_write_image_pair(record, split_root, image_size) for record in records]
    (output_root / f"{name}.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    return prepared


def prepare_fs2k(
    source: str | Path | None = None,
    url: str | None = None,
    download_dir: str | Path = "data/fs2k_download",
    output_dir: str | Path = "data/fs2k_prepared",
    validation_fraction: float = 0.15,
    seed: int = 42,
    image_size: int = 128,
) -> dict[str, Any]:
    if source is None and url is None:
        raise ValueError("Provide --source pointing to official FS2K data or --url for its official archive.")
    project_root = Path(__file__).resolve().parents[1]
    download_root = project_root / download_dir
    source_path = project_root / source if source and not Path(source).is_absolute() else Path(source) if source else None
    if source_path is None:
        archive_name = Path(url.split("?", 1)[0]).name or "fs2k_archive"
        source_path = _download(url, download_root / archive_name)
    if not source_path.exists():
        raise FileNotFoundError(
            f"FS2K source was not found: {source_path}. Replace the placeholder path with the actual "
            "downloaded FS2K folder or .zip/.tar archive."
        )
    extracted_root = _extract_if_needed(source_path, download_root)
    nested_root = extracted_root / "FS2K"
    if nested_root.is_dir():
        extracted_root = nested_root
    train_records, test_records = _discover_official_records(extracted_root)
    if len(train_records) + len(test_records) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} total pairs, found {len(train_records) + len(test_records)}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_records, validation_records = _stratified_split(train_records, validation_fraction, seed)
    output_root = project_root / output_dir
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = {
        "train": _write_split(train_records, "train", output_root, image_size),
        "validation": _write_split(validation_records, "validation", output_root, image_size),
        "test": _write_split(test_records, "test", output_root, image_size),
    }
    summary = {
        "source": str(extracted_root),
        "output": str(output_root.relative_to(project_root)),
        "seed": seed,
        "image_size": image_size,
        "validation_fraction": validation_fraction,
        "official_total_pairs": len(train_records) + len(validation_records) + len(test_records),
        "counts": {split: len(items) for split, items in prepared.items()},
        "styles": {split: dict(Counter(item["style"] for item in items)) for split, items in prepared.items()},
        "official_test_untouched": True,
        "pairing_preserved": all(item["pair_id"] in item["photo"] and item["pair_id"] in item["sketch"] for items in prepared.values() for item in items),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the official FS2K paired face-sketch dataset.")
    parser.add_argument("--source", default=None, help="Official FS2K directory or archive already downloaded.")
    parser.add_argument("--url", default=None, help="Official FS2K archive URL.")
    parser.add_argument("--download-dir", default="data/fs2k_download")
    parser.add_argument("--output-dir", default="data/fs2k_prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    prepare_fs2k(
        source=args.source,
        url=args.url,
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
    )
