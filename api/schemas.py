"""Schémas de réponse de l'API (documentation / typage)."""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Classe prédite")
    class_index: int
    confidence: float = Field(..., ge=0, le=1)
    explanation_stability: float = Field(..., ge=0, le=1)
    explanation_heatmap_png_base64: str
