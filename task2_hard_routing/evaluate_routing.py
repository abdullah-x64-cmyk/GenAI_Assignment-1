from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from task1_universal.evaluate_task1 import prepare_sample
from task1_universal.universal_autoencoder import ssim_score
from task2_hard_routing.corruption_classifier import CorruptionClass
from task2_hard_routing.hard_router import load_hard_router


CORRUPTIONS = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
SPECIALIST_FOR_CLASS = {1: "salt", 2: "blur", 3: "occlusion"}
CLASS_FOR_CORRUPTION = {"clean": 0, "salt_and_pepper": 1, "gaussian_blur": 2, "rectangular_occlusion": 3}
SEVERITIES = ["low", "medium", "high"]


def restoration_metrics(restored: torch.Tensor, target: torch.Tensor, alpha: float) -> dict[str, float]:
    restored_batch = restored.unsqueeze(0)
    target_batch = target.unsqueeze(0)
    l1 = torch.nn.functional.l1_loss(restored_batch, target_batch).item()
    ssim = ssim_score(restored_batch, target_batch).item()
    return {"loss": alpha * l1 + (1.0 - alpha) * (1.0 - ssim), "l1": l1, "ssim": ssim}


def evaluate_routing(
    max_per_group: int = 8,
    output_dir: str | Path = "reports/task2_routing_evaluation",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_file = project_root / "data" / "oxford_iiit_pet_splits" / "test_corruption_manifest.json"
    data_root = project_root / "data" / "oxford_iiit_pet"
    records = json.loads(manifest_file.read_text(encoding="utf-8"))
    checkpoint_root = project_root / "task2_hard_routing" / "checkpoints"
    router = load_hard_router(
        checkpoint_root / "corruption_classifier_best.pt",
        {
            "salt": checkpoint_root / "salt_expert_best.pt",
            "blur": checkpoint_root / "blur_expert_best.pt",
            "occlusion": checkpoint_root / "occlusion_expert_best.pt",
        },
        project_root=project_root,
    )
    router.eval()
    device = next(router.parameters()).device
    alpha = 0.5292732168433758
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clean_key = "clean/all"
        if len(grouped[clean_key]) < max_per_group:
            grouped[clean_key].append({"record": record, "corruption": "clean", "config": record["corruptions"]["clean"]})
        for severity in SEVERITIES:
            for corruption in CORRUPTIONS[1:]:
                key = f"{corruption}/{severity}"
                if len(grouped[key]) < max_per_group:
                    grouped[key].append({
                        "record": record,
                        "corruption": corruption,
                        "config": record["severity_manifest"][severity][corruption],
                    })

    mode_totals: dict[str, list[dict[str, float]]] = {"oracle": [], "predicted": []}
    group_totals: dict[str, dict[str, list[dict[str, float]]]] = {
        "oracle": defaultdict(list),
        "predicted": defaultdict(list),
    }
    mistakes: list[dict[str, Any]] = []
    classification_true: list[int] = []
    classification_predicted: list[int] = []

    for group, samples in sorted(grouped.items()):
        corruption, severity = group.split("/", maxsplit=1)
        true_class = CLASS_FOR_CORRUPTION[corruption]
        for sample in samples:
            target, corrupted, metadata = prepare_sample(
                sample["record"], corruption, data_root, 64, config_override=sample["config"]
            )
            with torch.no_grad():
                routed = router(corrupted.unsqueeze(0).to(device))
                probabilities = routed["probabilities"][0].cpu()
                predicted_class = int(routed["predicted_classes"][0].item())
                if true_class == 0:
                    oracle_restored = corrupted
                else:
                    expert_name = SPECIALIST_FOR_CLASS[true_class]
                    oracle_restored = router.experts[expert_name](corrupted.unsqueeze(0).to(device)).squeeze(0).cpu()
                predicted_restored = routed["restored"][0].cpu()

            oracle_metrics = restoration_metrics(oracle_restored, target, alpha)
            predicted_metrics = restoration_metrics(predicted_restored, target, alpha)
            for mode, metrics in (("oracle", oracle_metrics), ("predicted", predicted_metrics)):
                mode_totals[mode].append(metrics)
                group_totals[mode][group].append(metrics)
            classification_true.append(true_class)
            classification_predicted.append(predicted_class)
            if predicted_class != true_class:
                mistakes.append({
                    "image_path": metadata["image_path"],
                    "corruption": corruption,
                    "severity": severity,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "predicted_class_name": ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"][predicted_class],
                    "probabilities": probabilities.tolist(),
                    "oracle": oracle_metrics,
                    "predicted": predicted_metrics,
                    "loss_increase": predicted_metrics["loss"] - oracle_metrics["loss"],
                })

    def average(items: list[dict[str, float]]) -> dict[str, float]:
        return {metric: float(np.mean([item[metric] for item in items])) for metric in ("loss", "l1", "ssim")}

    confusion = np.zeros((4, 4), dtype=int)
    for true_class, predicted_class in zip(classification_true, classification_predicted):
        confusion[true_class, predicted_class] += 1
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    mistakes.sort(key=lambda item: item["loss_increase"], reverse=True)
    result = {
        "checkpoint_directory": str(checkpoint_root.relative_to(project_root)),
        "manifest": str(manifest_file.relative_to(project_root)),
        "max_per_group": max_per_group,
        "sample_count": len(classification_true),
        "oracle_metrics": {"overall": average(mode_totals["oracle"]), "by_group": {group: average(values) for group, values in group_totals["oracle"].items()}},
        "predicted_metrics": {"overall": average(mode_totals["predicted"]), "by_group": {group: average(values) for group, values in group_totals["predicted"].items()}},
        "classification_accuracy": float(np.mean(np.array(classification_true) == np.array(classification_predicted))),
        "classification_confusion_matrix": confusion.tolist(),
        "classification_normalized_confusion_matrix": normalized.tolist(),
        "classifier_mistake_count": len(mistakes),
        "classifier_mistakes_causing_restoration_failures": [mistake for mistake in mistakes if mistake["loss_increase"] > 0],
        "largest_classifier_mistakes": mistakes[:10],
    }
    output_file = output_root / "task2_routing_evaluation.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate oracle and predicted hard routing.")
    parser.add_argument("--max-per-group", type=int, default=8)
    parser.add_argument("--output-dir", default="reports/task2_routing_evaluation")
    args = parser.parse_args()
    evaluate_routing(max_per_group=args.max_per_group, output_dir=args.output_dir)
