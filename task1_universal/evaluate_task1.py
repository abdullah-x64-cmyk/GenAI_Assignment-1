from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.corruption_pipeline import apply_corruption
from task1_universal.universal_autoencoder import UniversalAutoencoder, ssim_score


CORRUPTION_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
SEVERITIES = ["low", "medium", "high"]


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[UniversalAutoencoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = UniversalAutoencoder(
        base_channels=config["base_channels"],
        latent_dim=config["latent_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def prepare_sample(
    record: dict[str, Any], corruption_name: str, data_root: Path, image_size: int, config_override: dict[str, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image_path = data_root / record["image_path"]
    clean_image = Image.open(image_path).convert("RGB")
    clean_array = np.asarray(clean_image.resize((image_size, image_size)), dtype=np.uint8)
    config = dict(config_override or record["corruptions"][corruption_name])
    seed = int(config.pop("seed", 0))
    config.pop("type", None)
    severity = config.pop("severity", "all")
    random.seed(seed)
    np.random.seed(seed)
    corrupted_array = apply_corruption(clean_array, corruption_name, config)
    to_tensor = transforms.ToTensor()
    return to_tensor(Image.fromarray(clean_array)), to_tensor(Image.fromarray(corrupted_array)), {
        "image_path": record["image_path"],
        "corruption": corruption_name,
        "severity": severity,
    }


def make_visualization(
    target: torch.Tensor,
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    target_image = target.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    corrupted_image = corrupted.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    restored_image = restored.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    error_map = np.abs(target_image - restored_image).mean(axis=2)

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    panels = [target_image, corrupted_image, restored_image, error_map]
    labels = ["Clean target", "Corrupted input", "Restored output", "Absolute error map"]
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(panel, cmap="magma" if label == "Absolute error map" else None, vmin=0.0 if label == "Absolute error map" else None, vmax=1.0 if label == "Absolute error map" else None)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_task1(
    checkpoint_path: str | Path = "task1_universal/checkpoints/universal_autoencoder_best.pt",
    manifest_path: str | Path = "data/oxford_iiit_pet_splits/test_corruption_manifest.json",
    max_per_group: int = 32,
    image_size: int | None = None,
    output_dir: str | Path = "reports/task1_evaluation",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_file = project_root / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    manifest_file = project_root / manifest_path if not Path(manifest_path).is_absolute() else Path(manifest_path)
    output_root = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    examples_dir = output_root / "representative_examples"
    failures_dir = output_root / "failure_cases"
    examples_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)
    for old_visual in (*examples_dir.glob("*.png"), *failures_dir.glob("*.png")):
        old_visual.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_file, device)
    selected_image_size = image_size or int(checkpoint.get("image_size", 64))
    data_root = project_root / "data" / "oxford_iiit_pet"
    records = json.loads(manifest_file.read_text(encoding="utf-8"))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clean_key = "clean/all"
        if len(grouped[clean_key]) < max_per_group:
            grouped[clean_key].append({"record": record, "corruption": "clean", "config": record["corruptions"]["clean"]})
        for severity in SEVERITIES:
            for corruption_name in CORRUPTION_NAMES[1:]:
                key = f"{corruption_name}/{severity}"
                if len(grouped[key]) < max_per_group:
                    grouped[key].append({
                        "record": record,
                        "corruption": corruption_name,
                        "severity": severity,
                        "config": record["severity_manifest"][severity][corruption_name],
                    })

    metrics: dict[str, dict[str, Any]] = {}
    failure_candidates: list[dict[str, Any]] = []
    representative_count = 0
    for key, samples in sorted(grouped.items()):
        loss_values: list[float] = []
        l1_values: list[float] = []
        ssim_values: list[float] = []
        corruption_name, severity = key.split("/", maxsplit=1)
        for sample_index, sample in enumerate(samples):
            target, corrupted, metadata = prepare_sample(
                sample["record"],
                corruption_name,
                data_root,
                selected_image_size,
                config_override=sample["config"],
            )
            with torch.no_grad():
                restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0).cpu()
            target_batch = target.unsqueeze(0)
            restored_batch = restored.unsqueeze(0)
            l1_value = torch.nn.functional.l1_loss(restored_batch, target_batch).item()
            ssim_value = ssim_score(restored_batch, target_batch).item()
            alpha = float(checkpoint["config"]["alpha"])
            loss_value = alpha * l1_value + (1.0 - alpha) * (1.0 - ssim_value)
            loss_values.append(loss_value)
            l1_values.append(l1_value)
            ssim_values.append(ssim_value)
            failure_candidates.append({
                "loss": loss_value,
                "target": target,
                "corrupted": corrupted,
                "restored": restored,
                "metadata": metadata,
            })

            if (corruption_name == "clean" and sample_index < 3) or (corruption_name != "clean" and severity in SEVERITIES and sample_index == 0):
                representative_count += 1
                filename = f"{representative_count:02d}_{corruption_name}_{severity}.png"
                make_visualization(target, corrupted, restored, examples_dir / filename, f"Representative: {corruption_name} / {severity}")

        metrics[key] = {
            "corruption": corruption_name,
            "severity": severity,
            "sample_count": len(samples),
            "loss": float(np.mean(loss_values)),
            "l1": float(np.mean(l1_values)),
            "ssim": float(np.mean(ssim_values)),
        }

    failure_candidates.sort(key=lambda item: item["loss"], reverse=True)
    for failure_index, candidate in enumerate(failure_candidates[:4], start=1):
        metadata = candidate["metadata"]
        make_visualization(
            candidate["target"],
            candidate["corrupted"],
            candidate["restored"],
            failures_dir / f"failure_{failure_index:02d}.png",
            f"Failure case {failure_index}: {metadata['corruption']} / {metadata['severity']}",
        )

    result = {
        "checkpoint": str(checkpoint_file.relative_to(project_root)),
        "manifest": str(manifest_file.relative_to(project_root)),
        "device": str(device),
        "image_size": selected_image_size,
        "max_per_group": max_per_group,
        "metrics": metrics,
        "representative_examples": representative_count,
        "failure_cases": min(4, len(failure_candidates)),
        "representative_examples_dir": str(examples_dir.relative_to(project_root)),
        "failure_cases_dir": str(failures_dir.relative_to(project_root)),
    }
    (output_root / "task1_evaluation_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task 1 restoration by corruption and severity.")
    parser.add_argument("--checkpoint", default="task1_universal/checkpoints/universal_autoencoder_best.pt")
    parser.add_argument("--manifest", default="data/oxford_iiit_pet_splits/test_corruption_manifest.json")
    parser.add_argument("--max-per-group", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output-dir", default="reports/task1_evaluation")
    args = parser.parse_args()
    evaluate_task1(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        max_per_group=args.max_per_group,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )
