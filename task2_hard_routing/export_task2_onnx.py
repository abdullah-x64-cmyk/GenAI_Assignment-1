from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from task2_hard_routing.corruption_classifier import CorruptionClassifier
from task2_hard_routing.specialist_autoencoders import create_specialist


EXPERTS = ("salt", "blur", "occlusion")


def _resolve(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _compare_outputs(pytorch_output: np.ndarray, onnx_output: np.ndarray) -> dict[str, Any]:
    difference = np.abs(pytorch_output - onnx_output)
    return {
        "pytorch_shape": list(pytorch_output.shape),
        "onnx_shape": list(onnx_output.shape),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "rtol": 1e-4,
        "atol": 1e-5,
        "allclose": bool(np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5)),
    }


def _export_model(model: torch.nn.Module, sample_input: torch.Tensor, output_path: Path) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pytorch_output = model(sample_input).cpu().numpy()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["corrupted_image"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["output"], {"corrupted_image": sample_input.numpy()})[0]
    return _compare_outputs(pytorch_output, onnx_output)


def export_task2_models(
    classifier_checkpoint: str | Path = "task2_hard_routing/checkpoints/corruption_classifier_best.pt",
    expert_checkpoints: dict[str, str | Path] | None = None,
    output_dir: str | Path = "task2_hard_routing/checkpoints/onnx",
    report_path: str | Path = "reports/task2_onnx_verification.json",
    seed: int = 42,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    classifier_file = _resolve(project_root, classifier_checkpoint)
    expert_checkpoints = expert_checkpoints or {
        name: f"task2_hard_routing/checkpoints/{name}_expert_best.pt" for name in EXPERTS
    }
    output_root = _resolve(project_root, output_dir)
    report_file = _resolve(project_root, report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(seed)

    classifier_payload = torch.load(classifier_file, map_location=device, weights_only=False)
    classifier_config = classifier_payload["config"]
    classifier = CorruptionClassifier(
        channels=tuple(classifier_config["channels"]),
        dropout=float(classifier_config["dropout"]),
    ).to(device)
    classifier.load_state_dict(classifier_payload["model_state_dict"])
    image_size = int(classifier_payload.get("image_size", 64))
    sample_input = torch.rand(1, 3, image_size, image_size, device=device)
    results: dict[str, Any] = {
        "input_shape": list(sample_input.shape),
        "opset_version": 17,
        "providers": ["CPUExecutionProvider"],
        "models": {},
    }
    classifier_output = output_root / "classifier.onnx"
    results["models"]["classifier"] = {
        "onnx_model": str(classifier_output.relative_to(project_root)),
        **_export_model(classifier, sample_input, classifier_output),
    }

    for specialist in EXPERTS:
        checkpoint_file = _resolve(project_root, expert_checkpoints[specialist])
        payload = torch.load(checkpoint_file, map_location=device, weights_only=False)
        config = payload["config"]
        expert = create_specialist(
            specialist,
            base_channels=int(config["base_channels"]),
            latent_dim=int(config["latent_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        ).to(device)
        expert.load_state_dict(payload["model_state_dict"])
        expert_output = output_root / f"{specialist}_expert.onnx"
        results["models"][f"{specialist}_expert"] = {
            "onnx_model": str(expert_output.relative_to(project_root)),
            **_export_model(expert, sample_input, expert_output),
        }

    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not all(model_result["allclose"] for model_result in results["models"].values()):
        raise RuntimeError("At least one Task 2 ONNX model failed consistency verification.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Task 2 classifier and specialists to ONNX.")
    parser.add_argument("--output-dir", default="task2_hard_routing/checkpoints/onnx")
    parser.add_argument("--report", default="reports/task2_onnx_verification.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_task2_models(output_dir=args.output_dir, report_path=args.report, seed=args.seed)
