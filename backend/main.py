from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
CLASS_NAMES = ["clean", "salt_and_pepper", "gaussian_blur", "occlusion"]


class ModelRegistry:
    def __init__(self) -> None:
        self.sessions: dict[str, ort.InferenceSession] = {}

    def load(self, name: str, relative_path: str) -> ort.InferenceSession:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.sessions[name] = session
        return session

    def get(self, name: str) -> ort.InferenceSession:
        if name not in self.sessions:
            raise RuntimeError(f"Model session is not loaded: {name}")
        return self.sessions[name]


registry = ModelRegistry()
app = FastAPI(title="Restoration Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models() -> None:
    model_paths = {
        "universal": "task1_universal/checkpoints/universal_autoencoder.onnx",
        "classifier": "task2_hard_routing/checkpoints/onnx/classifier.onnx",
        "salt_expert": "task2_hard_routing/checkpoints/onnx/salt_expert.onnx",
        "blur_expert": "task2_hard_routing/checkpoints/onnx/blur_expert.onnx",
        "occlusion_expert": "task2_hard_routing/checkpoints/onnx/occlusion_expert.onnx",
        "soft_moe": "task3_soft_moe/checkpoints/soft_moe.onnx",
        "face_generator": "task4_face_sketch/checkpoints/generator.onnx",
    }
    for name, path in model_paths.items():
        registry.load(name, path)


@app.get("/health-check")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": sorted(registry.sessions), "provider": "CPUExecutionProvider"}


def _validate_upload(file: UploadFile, payload: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, or BMP image.")
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")


def _decode_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error


def _image_tensor(image: Image.Image, size: int, centered: bool = False) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if centered:
        array = array * 2.0 - 1.0
    return array[None, ...].astype(np.float32)


def _encode_image(array: np.ndarray) -> str:
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result(image: np.ndarray, started: float, **metadata: Any) -> dict[str, Any]:
    return {
        "image_base64": _encode_image(image),
        "image_format": "png",
        "inference_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
        **metadata,
    }


@app.post("/universal-restoration")
async def universal_restoration(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await file.read()
    _validate_upload(file, payload)
    image = _decode_image(payload)
    tensor = _image_tensor(image, 64)
    output = registry.get("universal").run(["restored_image"], {"corrupted_image": tensor})[0][0]
    return _result(output.transpose(1, 2, 0), started, operation="universal-restoration", input_size=[64, 64])


@app.post("/hard-routing")
async def hard_routing(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await file.read()
    _validate_upload(file, payload)
    image = _decode_image(payload)
    tensor = _image_tensor(image, 64)
    classifier = registry.get("classifier").run(None, {"corrupted_image": tensor})[0][0]
    probabilities = np.exp(classifier - classifier.max())
    probabilities = probabilities / probabilities.sum()
    predicted_class = int(probabilities.argmax())
    if predicted_class == 0:
        restored = tensor[0]
        selected_expert = "identity"
    else:
        expert_names = {1: "salt_expert", 2: "blur_expert", 3: "occlusion_expert"}
        selected_expert = expert_names[predicted_class]
        restored = registry.get(selected_expert).run(["output"], {"corrupted_image": tensor})[0][0]
    return _result(
        restored.transpose(1, 2, 0),
        started,
        operation="hard-routing",
        predicted_class=predicted_class,
        predicted_class_name=CLASS_NAMES[predicted_class],
        probabilities={name: round(float(value), 6) for name, value in zip(CLASS_NAMES, probabilities)},
        selected_expert=selected_expert,
    )


@app.post("/soft-mixture")
async def soft_mixture(file: UploadFile = File(...)) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await file.read()
    _validate_upload(file, payload)
    image = _decode_image(payload)
    tensor = _image_tensor(image, 64)
    restored, weights = registry.get("soft_moe").run(
        ["restored_image", "routing_weights"], {"corrupted_image": tensor}
    )
    weights = weights[0]
    return _result(
        restored[0].transpose(1, 2, 0),
        started,
        operation="soft-mixture",
        routing_weights={name: round(float(value), 6) for name, value in zip(CLASS_NAMES, weights)},
    )


@app.post("/face-to-sketch")
async def face_to_sketch(
    file: UploadFile = File(...),
    style_id: int = Query(0, ge=0, le=2),
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = await file.read()
    _validate_upload(file, payload)
    image = _decode_image(payload)
    tensor = _image_tensor(image, 128, centered=True)
    output = registry.get("face_generator").run(
        ["generated_sketch"], {"photograph": tensor, "style_id": np.asarray([style_id], dtype=np.int64)}
    )[0][0]
    output = (output.transpose(1, 2, 0) + 1.0) / 2.0
    return _result(output, started, operation="face-to-sketch", style_id=style_id, output_size=[128, 128])
