from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from task2_hard_routing.corruption_classifier import CorruptionClassifier
from task2_hard_routing.specialist_autoencoders import create_specialist


class SoftMixtureOfExperts(nn.Module):
    """Identity plus three restoration experts combined by differentiable gate weights."""

    def __init__(
        self,
        gate: nn.Module,
        experts: Mapping[str, nn.Module],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero.")
        required = {"salt", "blur", "occlusion"}
        missing = required.difference(experts)
        if missing:
            raise ValueError(f"Missing experts: {sorted(missing)}")
        self.gate = gate
        self.experts = nn.ModuleDict(dict(experts))
        self.temperature = temperature

    def forward(self, corrupted: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_logits = self.gate(corrupted)
        weights = torch.softmax(gate_logits / self.temperature, dim=1)
        identity = corrupted
        salt_output = self.experts["salt"](corrupted)
        blur_output = self.experts["blur"](corrupted)
        occlusion_output = self.experts["occlusion"](corrupted)
        expert_outputs = torch.stack([identity, salt_output, blur_output, occlusion_output], dim=1)
        restored = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * expert_outputs).sum(dim=1)
        return {
            "restored": restored,
            "weights": weights,
            "gate_logits": gate_logits,
            "identity_output": identity,
            "salt_output": salt_output,
            "blur_output": blur_output,
            "occlusion_output": occlusion_output,
        }


def build_soft_moe(
    gate: nn.Module,
    salt_expert: nn.Module,
    blur_expert: nn.Module,
    occlusion_expert: nn.Module,
    temperature: float = 1.0,
) -> SoftMixtureOfExperts:
    return SoftMixtureOfExperts(
        gate,
        {"salt": salt_expert, "blur": blur_expert, "occlusion": occlusion_expert},
        temperature=temperature,
    )


def load_soft_moe(
    classifier_checkpoint: str | Path,
    expert_checkpoints: Mapping[str, str | Path],
    project_root: str | Path | None = None,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> SoftMixtureOfExperts:
    """Initialize the gate and experts from the trained Task 2 checkpoints."""
    device = device or torch.device("cpu")
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]

    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else root / path

    classifier_payload = torch.load(resolve(classifier_checkpoint), map_location=device, weights_only=False)
    classifier_config = classifier_payload["config"]
    gate = CorruptionClassifier(
        channels=tuple(classifier_config["channels"]),
        dropout=float(classifier_config["dropout"]),
    )
    gate.load_state_dict(classifier_payload["model_state_dict"])

    experts: dict[str, nn.Module] = {}
    for specialist in ("salt", "blur", "occlusion"):
        payload = torch.load(resolve(expert_checkpoints[specialist]), map_location=device, weights_only=False)
        config = payload["config"]
        expert = create_specialist(
            specialist,
            base_channels=int(config["base_channels"]),
            latent_dim=int(config["latent_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )
        expert.load_state_dict(payload["model_state_dict"])
        experts[specialist] = expert

    model = SoftMixtureOfExperts(gate, experts, temperature=temperature).to(device)
    model.eval()
    return model
