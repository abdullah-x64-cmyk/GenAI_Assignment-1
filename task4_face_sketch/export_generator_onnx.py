from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from task4_face_sketch.conditional_gan import ConditionalUNetGenerator


def export_generator_onnx(
    checkpoint_path: str | Path = "task4_face_sketch/checkpoints/conditional_gan_best.pt",
    output_path: str | Path = "task4_face_sketch/checkpoints/generator.onnx",
    report_path: str | Path = "reports/task4_generator_onnx_verification.json",
    seed: int = 42,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_file = project_root / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    output_file = project_root / output_path if not Path(output_path).is_absolute() else Path(output_path)
    report_file = project_root / report_path if not Path(report_path).is_absolute() else Path(report_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    image_size = int(checkpoint["config"].get("image_size", 128))
    generator = ConditionalUNetGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    torch.manual_seed(seed)
    photo = torch.rand(1, 3, image_size, image_size, device=device)
    styles = [torch.tensor([style_id], dtype=torch.long, device=device) for style_id in range(3)]
    style = styles[1]

    torch.onnx.export(
        generator,
        (photo, style),
        output_file,
        input_names=["photograph", "style_id"],
        output_names=["generated_sketch"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    style_results = {}
    for style in styles:
        with torch.no_grad():
            pytorch_output = generator(photo, style).numpy()
        onnx_output = session.run(
            ["generated_sketch"],
            {"photograph": photo.numpy(), "style_id": style.numpy()},
        )[0]
        difference = np.abs(pytorch_output - onnx_output)
        style_results[str(int(style.item()))] = {
            "output_shape": list(onnx_output.shape),
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "allclose": bool(np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5)),
        }
    result = {
        "checkpoint": str(checkpoint_file.relative_to(project_root)),
        "onnx_model": str(output_file.relative_to(project_root)),
        "input_photo_shape": list(photo.shape),
        "input_style_shape": list(style.shape),
        "style_results": style_results,
        "opset_version": 17,
        "providers": session.get_providers(),
        "rtol": 1e-4,
        "atol": 1e-5,
        "allclose": all(item["allclose"] for item in style_results.values()),
    }
    report_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["allclose"]:
        raise RuntimeError("Generator ONNX output does not match PyTorch output within tolerance.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the conditional GAN generator to ONNX.")
    parser.add_argument("--checkpoint", default="task4_face_sketch/checkpoints/conditional_gan_best.pt")
    parser.add_argument("--onnx", default="task4_face_sketch/checkpoints/generator.onnx")
    parser.add_argument("--report", default="reports/task4_generator_onnx_verification.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export_generator_onnx(args.checkpoint, args.onnx, args.report, args.seed)
