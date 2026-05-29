"""
API d'inférence FastAPI.

POST /predict : image → classe prédite, confiance, carte d'explication (PNG
base64) ET score de stabilité de l'explication. Servir l'explication AVEC la
prédiction est le parti pris du projet : un modèle « auditable par conception ».

Instrumentée pour Prometheus (/metrics) : nb de requêtes, latence, confiance,
stabilité.
"""
from __future__ import annotations

import base64
import io
import os
import time

from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from ivml.data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from ivml.interpret import by_design_explanation
from ivml.interpret.metrics import explanation_stability
from ivml.model import ModelConfig, build_model
from ivml.utils import get_device, get_logger

logger = get_logger("ivml.api")

CKPT = os.environ.get("MODEL_CHECKPOINT", "artifacts/best_model.pt")
device = get_device()
model = build_model(ModelConfig()).to(device)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=device))
        logger.info("Modèle chargé : %s", CKPT)
    else:
        logger.warning("Checkpoint introuvable (%s) — modèle non entraîné.", CKPT)
    model.eval()
    yield


app = FastAPI(title="Interpretable Vision API", version="0.1.0", lifespan=lifespan)

REQUESTS = Counter("predict_requests_total", "Nombre total de requêtes /predict")
LATENCY = Histogram("predict_latency_seconds", "Latence de /predict")
CONFIDENCE = Histogram("predict_confidence", "Confiance des prédictions")
STABILITY = Histogram("explanation_stability", "Stabilité des explications servies")


def _preprocess(img: Image.Image, size: int = 32) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def _heatmap_png(explanation: torch.Tensor) -> str:
    """Carte d'explication [H, W] → PNG base64 (colormap simple rouge)."""
    arr = (explanation.squeeze().cpu().numpy() * 255).astype(np.uint8)
    rgb = np.stack([arr, np.zeros_like(arr), 255 - arr], axis=-1)  # bleu→rouge
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((128, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": os.path.exists(CKPT)}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    REQUESTS.inc()
    start = time.perf_counter()

    image = Image.open(io.BytesIO(await file.read()))
    x = _preprocess(image)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())
    confidence = float(probs[pred])

    explanation = by_design_explanation(model, x)
    stability = explanation_stability(model, x, by_design_explanation, n_samples=4)["stability_score"]

    CONFIDENCE.observe(confidence)
    STABILITY.observe(stability)
    LATENCY.observe(time.perf_counter() - start)

    return JSONResponse(
        {
            "prediction": CIFAR10_CLASSES[pred],
            "class_index": pred,
            "confidence": round(confidence, 4),
            "explanation_stability": round(stability, 4),
            "explanation_heatmap_png_base64": _heatmap_png(explanation),
        }
    )
