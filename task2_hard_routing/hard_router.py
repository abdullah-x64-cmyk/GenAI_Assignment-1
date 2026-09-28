from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from task2_hard_routing.corruption_classifier import CLASS_NAMES, CorruptionClass, CorruptionClassifier
from task2_hard_routing.specialist_autoencoders import create_specialist


class HardRoutedRestoration(nn.Module):
    """Classifier followed by identity bypass or one corruption specialist."""

    def __init__(self, classifier: CorruptionClassifier, experts: Mapping[str, nn.Module]):
        super().__init__()
        required_experts = {"salt", "blur", "occlusion"}
        missing = required_experts.difference(experts)
        if missing:
            raise ValueError(f"Missing specialist experts: {sorted(missing)}")
        self.classifier = classifier
        self.experts = nn.ModuleDict(dict(experts))

    def forward(self, corrupted: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = torch.softmax(self.classifier(corrupted), dim=1)
        predicted_classes = probabilities.argmax(dim=1)
        restored = corrupted.clone()

        routes = {
            int(CorruptionClass.SALT_AND_PEPPER): "salt",
            int(CorruptionClass.GAUSSIAN_BLUR): "blur",
            int(CorruptionClass.OCCLUSION): "occlusion",
        }
        for class_id, expert_name in routes.items():
            sample_indices = torch.where(predicted_classes == class_id)[0]
            if sample_indices.numel() > 0:
                restored[sample_indices] = self.experts[expert_name](corrupted[sample_indices])

        selected_experts = torch.full(
            (corrupted.size(0),),
            fill_value=-1,
            dtype=torch.long,
            device=corrupted.device,
        )
        selected_experts[predicted_classes == int(CorruptionClass.SALT_AND_PEPPER)] = 1
        selected_experts[predicted_classes == int(CorruptionClass.GAUSSIAN_BLUR)] = 2
        selected_experts[predicted_classes == int(CorruptionClass.OCCLUSION)] = 3
        return {
            "restored": restored,
            "probabilities": probabilities,
            "predicted_classes": predicted_classes,
            "selected_experts": selected_experts,
        }


def build_hard_router(
    classifier: CorruptionClassifier,
    salt_expert: nn.Module,
    blur_expert: nn.Module,
    occlusion_expert: nn.Module,
) -> HardRoutedRestoration:
    return HardRoutedRestoration(
        classifier,
        {"salt": salt_expert, "blur": blur_expert, "occlusion": occlusion_expert},
    )


def load_hard_router(
    classifier_checkpoint: str | Path,
    expert_checkpoints: Mapping[str, str | Path],
    project_root: str | Path | None = None,
    device: torch.device | None = None,
) -> HardRoutedRestoration:
    """Load a classifier and three independently trained experts from checkpoints."""
    device = device or torch.device("cpu")
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]

    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else root / path

    classifier_file = resolve(classifier_checkpoint)
    classifier_payload = torch.load(classifier_file, map_location=device, weights_only=False)
    classifier_config = classifier_payload.get("config", {})
    classifier = CorruptionClassifier(
        channels=tuple(classifier_config.get("channels", (32, 64, 128))),
        dropout=float(classifier_config.get("dropout", 0.2)),
    )
    classifier.load_state_dict(classifier_payload["model_state_dict"])

    experts: dict[str, nn.Module] = {}
    for specialist in ("salt", "blur", "occlusion"):
        checkpoint_file = resolve(expert_checkpoints[specialist])
        payload = torch.load(checkpoint_file, map_location=device, weights_only=False)
        config = payload["config"]
        expert = create_specialist(
            specialist,
            base_channels=int(config["base_channels"]),
            latent_dim=int(config["latent_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )
        expert.load_state_dict(payload["model_state_dict"])
        experts[specialist] = expert

    router = build_hard_router(
        classifier,
        experts["salt"],
        experts["blur"],
        experts["occlusion"],
    ).to(device)
    router.eval()
    return router


def class_names_from_predictions(predicted_classes: torch.Tensor) -> list[str]:
    return [CLASS_NAMES[int(class_id)] for class_id in predicted_classes.detach().cpu()]
