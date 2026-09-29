from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from task3_soft_moe.soft_moe import load_soft_moe


class SoftMoEOnnxWrapper(nn.Module):
    """Expose tensor outputs from the dictionary-returning SoftMixtureOfExperts."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, corrupted_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model(corrupted_image)
        return result["restored"], result["weights"]


def export_task3_onnx(
    soft_checkpoint: str | Path = "task3_soft_moe/checkpoints/soft_moe_best.pt",
    output_path: str | Path = "task3_soft_moe/checkpoints/soft_moe.onnx",
    report_path: str | Path = "reports/task3_onnx_verification.json",
    seed: int = 42,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_file = project_root / soft_checkpoint if not Path(soft_checkpoint).is_absolute() else Path(soft_checkpoint)
    output_file = project_root / output_path if not Path(output_path).is_absolute() else Path(output_path)
    report_file = project_root / report_path if not Path(report_path).is_absolute() else Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    soft_payload = torch.load(checkpoint_file, map_location=device, weights_only=False)
    config = soft_payload["config"]
    model = load_soft_moe(
        "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
        {
            "salt": "task2_hard_routing/checkpoints/salt_expert_best.pt",
            "blur": "task2_hard_routing/checkpoints/blur_expert_best.pt",
            "occlusion": "task2_hard_routing/checkpoints/occlusion_expert_best.pt",
        },
        project_root=project_root,
        temperature=float(config["temperature"]),
        device=device,
    )
    model.load_state_dict(soft_payload["model_state_dict"])
    model.eval()
    wrapper = SoftMoEOnnxWrapper(model).eval()
    image_size = int(config.get("image_size", 64))
    torch.manual_seed(seed)
    sample_input = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        pytorch_restored, pytorch_weights = wrapper(sample_input)
    torch.onnx.export(
        wrapper,
        sample_input,
        output_file,
        input_names=["corrupted_image"],
        output_names=["restored_image", "routing_weights"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    onnx_restored, onnx_weights = session.run(
        ["restored_image", "routing_weights"],
        {"corrupted_image": sample_input.numpy()},
    )
    restored_difference = np.abs(pytorch_restored.numpy() - onnx_restored)
    weight_difference = np.abs(pytorch_weights.numpy() - onnx_weights)
    result = {
        "checkpoint": str(checkpoint_file.relative_to(project_root)),
        "onnx_model": str(output_file.relative_to(project_root)),
        "input_shape": list(sample_input.shape),
        "restored_shape": list(onnx_restored.shape),
        "routing_weights_shape": list(onnx_weights.shape),
        "temperature": float(config["temperature"]),
        "opset_version": 17,
        "providers": session.get_providers(),
        "restored_max_absolute_difference": float(restored_difference.max()),
        "restored_mean_absolute_difference": float(restored_difference.mean()),
        "weights_max_absolute_difference": float(weight_difference.max()),
        "weights_mean_absolute_difference": float(weight_difference.mean()),
        "rtol": 1e-4,
        "atol": 1e-5,
        "restored_allclose": bool(np.allclose(pytorch_restored.numpy(), onnx_restored, rtol=1e-4, atol=1e-5)),
        "weights_allclose": bool(np.allclose(pytorch_weights.numpy(), onnx_weights, rtol=1e-4, atol=1e-5)),
    }
    result["allclose"] = result["restored_allclose"] and result["weights_allclose"]
    report_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["allclose"]:
        raise RuntimeError("Soft-MoE ONNX outputs failed PyTorch consistency verification.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the complete Task 3 soft MoE to ONNX.")
    parser.add_argument("--checkpoint", default="task3_soft_moe/checkpoints/soft_moe_best.pt")
    parser.add_argument("--onnx", default="task3_soft_moe/checkpoints/soft_moe.onnx")
    parser.add_argument("--report", default="reports/task3_onnx_verification.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_task3_onnx(args.checkpoint, args.onnx, args.report, args.seed)
