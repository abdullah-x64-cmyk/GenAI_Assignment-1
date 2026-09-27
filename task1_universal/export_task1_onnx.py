from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from task1_universal.evaluate_task1 import load_checkpoint


def export_and_verify(
    checkpoint_path: str | Path = "task1_universal/checkpoints/universal_autoencoder_best.pt",
    onnx_path: str | Path = "task1_universal/checkpoints/universal_autoencoder.onnx",
    report_path: str | Path = "reports/task1_onnx_verification.json",
    seed: int = 42,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_file = project_root / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    output_file = project_root / onnx_path if not Path(onnx_path).is_absolute() else Path(onnx_path)
    verification_file = project_root / report_path if not Path(report_path).is_absolute() else Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    verification_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(checkpoint_file, device)
    image_size = int(checkpoint.get("image_size", 64))
    torch.manual_seed(seed)
    sample_input = torch.rand(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        pytorch_output = model(sample_input).cpu().numpy()

    torch.onnx.export(
        model,
        sample_input,
        output_file,
        input_names=["corrupted_image"],
        output_names=["restored_image"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["restored_image"], {"corrupted_image": sample_input.numpy()})[0]

    difference = np.abs(pytorch_output - onnx_output)
    max_absolute_difference = float(difference.max())
    mean_absolute_difference = float(difference.mean())
    allclose = bool(np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5))
    result = {
        "checkpoint": str(checkpoint_file.relative_to(project_root)),
        "onnx_model": str(output_file.relative_to(project_root)),
        "input_shape": list(sample_input.shape),
        "pytorch_output_shape": list(pytorch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
        "opset_version": 17,
        "providers": session.get_providers(),
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "rtol": 1e-4,
        "atol": 1e-5,
        "allclose": allclose,
    }
    verification_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not allclose:
        raise RuntimeError("ONNX Runtime output does not match the PyTorch output within tolerance.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and verify the Task 1 autoencoder with ONNX Runtime.")
    parser.add_argument("--checkpoint", default="task1_universal/checkpoints/universal_autoencoder_best.pt")
    parser.add_argument("--onnx", default="task1_universal/checkpoints/universal_autoencoder.onnx")
    parser.add_argument("--report", default="reports/task1_onnx_verification.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_and_verify(args.checkpoint, args.onnx, args.report, args.seed)
