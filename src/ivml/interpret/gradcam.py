"""
Grad-CAM (Selvaraju et al., 2017) — explication POST-HOC, pour comparaison.

Sert de point de comparaison à l'attention par conception : on mesure ensuite
si l'explication intégrée est aussi fidèle et plus stable que ce post-hoc.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM sur la dernière couche convolutive du backbone."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> torch.Tensor:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0])
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)        # [B, C, 1, 1]
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        cam = (flat - flat.min(1, keepdim=True).values) / (
            flat.max(1, keepdim=True).values - flat.min(1, keepdim=True).values + 1e-8
        )
        return cam.view(x.shape[0], *x.shape[-2:])

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()
