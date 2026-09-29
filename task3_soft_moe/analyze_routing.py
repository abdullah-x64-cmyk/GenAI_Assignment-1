from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from task1_universal.evaluate_task1 import prepare_sample
from task3_soft_moe.soft_moe import load_soft_moe


CORRUPTIONS = ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
SEVERITIES = ["low", "medium", "high"]
WEIGHT_NAMES = ["Clean", "Salt", "Blur", "Occlusion"]


def build_groups(records: list[dict[str, Any]], max_per_group: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if len(groups["clean/all"]) < max_per_group:
            groups["clean/all"].append({"record": record, "corruption": "clean", "config": record["corruptions"]["clean"]})
        for severity in SEVERITIES:
            for corruption in CORRUPTIONS[1:]:
                key = f"{corruption}/{severity}"
                if len(groups[key]) < max_per_group:
                    groups[key].append({
                        "record": record,
                        "corruption": corruption,
                        "config": record["severity_manifest"][severity][corruption],
                    })
    return groups


def save_heatmap(group_names: list[str], averages: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(averages, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(4), WEIGHT_NAMES)
    axis.set_yticks(range(len(group_names)), group_names)
    axis.set_title("Average Soft Routing Weights")
    for row in range(averages.shape[0]):
        for column in range(averages.shape[1]):
            axis.text(column, row, f"{averages[row, column]:.2f}", ha="center", va="center", color="white" if averages[row, column] < 0.5 else "black")
    figure.colorbar(image, ax=axis, label="Average weight")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_distribution(group_names: list[str], weights_by_group: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    positions = np.arange(len(group_names))
    averages = np.array([weights_by_group[group].mean(axis=0) for group in group_names])
    axes[0].bar(positions, averages[:, 0], label=WEIGHT_NAMES[0])
    bottom = averages[:, 0].copy()
    for column in range(1, 4):
        axes[0].bar(positions, averages[:, column], bottom=bottom, label=WEIGHT_NAMES[column])
        bottom += averages[:, column]
    axes[0].set_ylabel("Average weight")
    axes[0].set_title("Weight Distribution by Condition")
    axes[0].legend(ncol=4)
    axes[0].set_ylim(0, 1)
    axes[1].boxplot([weights_by_group[group].max(axis=1) for group in group_names], tick_labels=group_names, showfliers=False)
    axes[1].set_ylabel("Largest expert weight")
    axes[1].set_title("Dominance Distribution")
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_example(image: torch.Tensor, weights: np.ndarray, metadata: dict[str, Any], output: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title(f"{metadata['corruption']} / {metadata['severity']}")
    axes[0].axis("off")
    axes[1].bar(WEIGHT_NAMES, weights, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Gate weight")
    axes[1].set_title(title)
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def analyze_routing(
    max_per_group: int = 16,
    temperature: float | None = None,
    output_dir: str | Path = "reports/task3_routing_analysis",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dominant_dir = output_root / "dominant_examples"
    mixed_dir = output_root / "multiple_expert_examples"
    dominant_dir.mkdir(exist_ok=True)
    mixed_dir.mkdir(exist_ok=True)
    for old_file in (*dominant_dir.glob("*.png"), *mixed_dir.glob("*.png")):
        old_file.unlink()

    tuned_report = json.loads((project_root / "reports" / "task3_soft_moe_optuna_results.json").read_text(encoding="utf-8"))
    selected_temperature = temperature or float(tuned_report["best_parameters"]["temperature"])
    model = load_soft_moe(
        "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
        {
            "salt": "task2_hard_routing/checkpoints/salt_expert_best.pt",
            "blur": "task2_hard_routing/checkpoints/blur_expert_best.pt",
            "occlusion": "task2_hard_routing/checkpoints/occlusion_expert_best.pt",
        },
        project_root=project_root,
        temperature=selected_temperature,
        device=torch.device("cpu"),
    )
    model.eval()
    records = json.loads((project_root / "data" / "oxford_iiit_pet_splits" / "test_corruption_manifest.json").read_text(encoding="utf-8"))
    groups = build_groups(records, max_per_group)
    data_root = project_root / "data" / "oxford_iiit_pet"
    weights_by_group: dict[str, np.ndarray] = {}
    examples: list[dict[str, Any]] = []
    for group, samples in sorted(groups.items()):
        group_weights = []
        for sample in samples:
            _, corrupted, metadata = prepare_sample(sample["record"], sample["corruption"], data_root, 64, config_override=sample["config"])
            with torch.no_grad():
                result = model(corrupted.unsqueeze(0))
            weights = result["weights"][0].numpy()
            group_weights.append(weights)
            examples.append({"image": corrupted, "weights": weights, "metadata": metadata})
        weights_by_group[group] = np.asarray(group_weights)

    group_names = sorted(weights_by_group)
    averages = np.array([weights_by_group[group].mean(axis=0) for group in group_names])
    all_weights = np.concatenate(list(weights_by_group.values()), axis=0)
    dominant = np.argmax(all_weights, axis=1)
    dominant_share = (all_weights.max(axis=1) >= 0.5).mean()
    multi_share = ((all_weights >= 0.2).sum(axis=1) >= 2).mean()
    inactivity_threshold = 0.05
    inactive_experts = [WEIGHT_NAMES[index] for index, value in enumerate(all_weights.mean(axis=0)) if value < inactivity_threshold]

    save_heatmap(group_names, averages, output_root / "routing_heatmap.png")
    save_distribution(group_names, weights_by_group, output_root / "weight_distributions.png")
    dominant_examples = sorted(examples, key=lambda item: float(item["weights"].max()), reverse=True)[:4]
    mixed_examples = sorted(examples, key=lambda item: float(np.sort(item["weights"])[-2:].sum()), reverse=True)[:4]
    for index, example in enumerate(dominant_examples, start=1):
        save_example(example["image"], example["weights"], example["metadata"], dominant_dir / f"dominant_{index:02d}.png", "One expert dominates")
    for index, example in enumerate(mixed_examples, start=1):
        save_example(example["image"], example["weights"], example["metadata"], mixed_dir / f"multiple_{index:02d}.png", "Multiple experts contribute")

    result = {
        "temperature": selected_temperature,
        "max_per_group": max_per_group,
        "sample_count": int(all_weights.shape[0]),
        "groups": {
            group: {
                "sample_count": int(weights_by_group[group].shape[0]),
                "average_weights": {name: float(value) for name, value in zip(WEIGHT_NAMES, averages[index])},
            }
            for index, group in enumerate(group_names)
        },
        "overall_average_weights": {name: float(value) for name, value in zip(WEIGHT_NAMES, all_weights.mean(axis=0))},
        "dominant_expert_share": float(dominant_share),
        "multiple_expert_share": float(multi_share),
        "inactivity_threshold": inactivity_threshold,
        "inactive_experts": inactive_experts,
        "heatmap": str((output_root / "routing_heatmap.png").relative_to(project_root)),
        "weight_distribution": str((output_root / "weight_distributions.png").relative_to(project_root)),
        "dominant_examples": str(dominant_dir.relative_to(project_root)),
        "multiple_expert_examples": str(mixed_dir.relative_to(project_root)),
    }
    (output_root / "routing_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Task 3 soft routing weights.")
    parser.add_argument("--max-per-group", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output-dir", default="reports/task3_routing_analysis")
    args = parser.parse_args()
    analyze_routing(args.max_per_group, args.temperature, args.output_dir)
