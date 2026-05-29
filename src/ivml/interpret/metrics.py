"""
Métriques quantitatives d'explication.

On ne *prétend* pas que l'explication est bonne : on la *mesure*.

- Fidélité (deletion) : si on masque progressivement les pixels jugés les plus
  importants par l'explication, la confiance du modèle doit chuter. Plus elle
  chute vite (faible AUC de la courbe de confiance), plus l'explication est
  fidèle au raisonnement du modèle.

- Stabilité : sur des entrées quasi identiques (bruit gaussien epsilon), une
  bonne explication doit rester cohérente. On mesure la distance moyenne entre
  cartes d'explication de l'entrée bruitée. Score = 1 - distance moyenne
  (1 = parfaitement stable). Répond directement à l'exigence « renforcer la
  stabilité des explications ».
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

ExplainFn = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]


@torch.no_grad()
def faithfulness_deletion(
    model: torch.nn.Module,
    x: torch.Tensor,
    explanation: torch.Tensor,
    steps: int = 20,
) -> dict[str, float]:
    """
    Courbe de suppression. `x` : [1, C, H, W]. `explanation` : [1, H, W] dans [0, 1].
    Renvoie l'AUC de la confiance (plus bas = plus fidèle) et la chute totale.
    """
    if x.shape[0] != 1:
        raise ValueError("faithfulness_deletion attend un seul échantillon (batch=1).")
    model.eval()
    device = x.device
    target = int(model(x).argmax(dim=1)[0])
    base_conf = F.softmax(model(x), dim=1)[0, target].item()

    order = explanation.flatten().argsort(descending=True)  # pixels les + importants d'abord
    h, w = explanation.shape[-2:]
    n_pixels = h * w
    confidences = [base_conf]
    x_mod = x.clone()
    chunk = max(1, n_pixels // steps)
    for i in range(steps):
        idx = order[i * chunk : (i + 1) * chunk]
        rows, cols = idx // w, idx % w
        x_mod[:, :, rows, cols] = 0.0                                 # masquage (baseline noire)
        conf = F.softmax(model(x_mod), dim=1)[0, target].item()
        confidences.append(conf)

    conf_t = torch.tensor(confidences, device=device)
    auc = torch.trapz(conf_t).item() / (len(conf_t) - 1)
    return {
        "deletion_auc": auc,
        "confidence_drop": base_conf - confidences[-1],
        "base_confidence": base_conf,
    }


def explanation_stability(
    model: torch.nn.Module,
    x: torch.Tensor,
    explain_fn: ExplainFn,
    epsilon: float = 0.03,
    n_samples: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """
    Stabilité de l'explication sous perturbation. `x` : [1, C, H, W].
    Renvoie un score dans [0, 1] (1 = parfaitement stable).
    """
    if x.shape[0] != 1:
        raise ValueError("explanation_stability attend un seul échantillon (batch=1).")
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = explain_fn(model, x).flatten(1)                            # [1, H*W]
    dists: list[float] = []
    for _ in range(n_samples):
        noise = torch.randn(x.shape, generator=g).to(x.device) * epsilon
        pert = explain_fn(model, x + noise).flatten(1)
        # distance L1 moyenne entre cartes normalisées dans [0, 1]
        dists.append((base - pert).abs().mean().item())
    mean_dist = sum(dists) / len(dists)
    return {
        "stability_score": max(0.0, 1.0 - mean_dist),
        "mean_explanation_shift": mean_dist,
        "epsilon": epsilon,
    }
