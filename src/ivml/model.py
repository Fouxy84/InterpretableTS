"""
Modèle de vision interprétable par conception.

Thèse du projet : une carte d'attention « dense » produite *après coup* n'est pas
une explication fiable (cf. Jain & Wallace, 2019). Ici l'attention spatiale fait
partie intégrante du chemin de décision : la représentation utilisée par le
classifieur EST la somme pondérée des features par l'attention. La carte
d'attention est donc une explication *par conception*, pas post-hoc.

    Image [B, 3, H, W]
        │
        ▼
    Backbone convolutif  ──►  features [B, C, h, w]
        │
        ▼
    Module d'attention (1×1 conv + softmax spatial)  ──►  α [B, 1, h, w]   (Σα = 1)
        │
        ▼
    Pooling pondéré  z = Σ_hw α · features   ──►  z [B, C]
        │
        ▼
    Classifieur linéaire  ──►  logits [B, n_classes]

`forward(return_attention=True)` renvoie aussi α : aucune rétro-propagation ni
hook n'est nécessaire pour obtenir l'explication, elle sort du forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    in_channels: int = 3
    n_classes: int = 10
    widths: tuple[int, ...] = (32, 64, 128)
    attention_temperature: float = 1.0


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool 2×2."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SpatialAttentionPool(nn.Module):
    """
    Pooling par attention spatiale, interprétable par conception.

    Produit une carte α normalisée par softmax sur les positions spatiales,
    puis agrège les features en une somme pondérée. α est exposé tel quel
    comme explication.
    """

    def __init__(self, channels: int, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature doit être > 0")
        self.temperature = temperature
        # Le scoreur d'attention : 1×1 conv → score scalaire par position.
        self.scorer = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = features.shape
        scores = self.scorer(features) / self.temperature          # [B, 1, h, w]
        attn = F.softmax(scores.view(b, 1, h * w), dim=-1)          # softmax spatial
        attn = attn.view(b, 1, h, w)                                # [B, 1, h, w], Σ = 1
        pooled = (features * attn).sum(dim=(2, 3))                  # [B, C]
        return pooled, attn


class InterpretableCNN(nn.Module):
    """CNN à attention spatiale intégrée (explication par conception)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        blocks: list[nn.Module] = []
        c_in = cfg.in_channels
        for c_out in cfg.widths:
            blocks.append(_conv_block(c_in, c_out))
            c_in = c_out
        self.backbone = nn.Sequential(*blocks)
        self.feature_channels = c_in

        self.attention = SpatialAttentionPool(c_in, cfg.attention_temperature)
        self.classifier = nn.Linear(c_in, cfg.n_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)               # [B, C, h, w]
        pooled, attn = self.attention(features)   # [B, C], [B, 1, h, w]
        logits = self.classifier(pooled)          # [B, n_classes]
        if return_attention:
            return logits, attn
        return logits

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> torch.Tensor:
        """
        Renvoie la carte d'attention par conception, ré-échantillonnée à la
        taille de l'image d'entrée, normalisée dans [0, 1] par échantillon.
        """
        self.eval()
        _, attn = self.forward(x, return_attention=True)
        attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = attn.squeeze(1)                                      # [B, H, W]
        flat = attn.flatten(1)
        amin = flat.min(dim=1, keepdim=True).values
        amax = flat.max(dim=1, keepdim=True).values
        attn = (flat - amin) / (amax - amin + 1e-8)
        return attn.view(x.shape[0], *x.shape[-2:])                 # [B, H, W]


def build_model(config: ModelConfig | None = None) -> InterpretableCNN:
    return InterpretableCNN(config)
