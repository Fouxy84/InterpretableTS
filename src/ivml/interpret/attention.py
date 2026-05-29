"""Explication par conception : la carte d'attention intégrée au modèle."""
from __future__ import annotations

import torch

from ivml.model import InterpretableCNN


def by_design_explanation(model: InterpretableCNN, x: torch.Tensor) -> torch.Tensor:
    """
    Renvoie l'explication par conception (carte d'attention spatiale), [B, H, W],
    normalisée dans [0, 1]. Sortie directe du forward, sans gradient ni hook.
    """
    return model.explain(x)
