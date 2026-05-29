"""
Évaluation de l'interprétabilité : compare attention par conception vs Grad-CAM
post-hoc sur la fidélité et la stabilité, et logge le tableau dans MLflow.

Usage :
    python -m ivml.evaluate --checkpoint artifacts/best_model.pt --n 50
"""
from __future__ import annotations

import argparse

import mlflow
import torch

from ivml.data import DataConfig, get_dataloaders
from ivml.interpret import GradCAM, by_design_explanation
from ivml.interpret.metrics import explanation_stability, faithfulness_deletion
from ivml.model import ModelConfig, build_model
from ivml.utils import get_device, get_logger, set_seed

logger = get_logger("ivml.evaluate")


def main(checkpoint: str, n: int) -> None:
    set_seed(42)
    device = get_device()
    model = build_model(ModelConfig()).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    _, test_dl = get_dataloaders(DataConfig(batch_size=1))
    gradcam = GradCAM(model, model.backbone[-1])

    def gradcam_fn(m, x):
        return gradcam(x)

    agg = {"bydesign_faith": [], "bydesign_stab": [], "gradcam_faith": [], "gradcam_stab": []}
    for i, (image, _) in enumerate(test_dl):
        if i >= n:
            break
        image = image.to(device)
        exp_design = by_design_explanation(model, image)
        agg["bydesign_faith"].append(
            faithfulness_deletion(model, image, exp_design)["deletion_auc"]
        )
        agg["bydesign_stab"].append(
            explanation_stability(model, image, by_design_explanation)["stability_score"]
        )
        exp_cam = gradcam(image)
        agg["gradcam_faith"].append(faithfulness_deletion(model, image, exp_cam)["deletion_auc"])
        agg["gradcam_stab"].append(
            explanation_stability(model, image, gradcam_fn)["stability_score"]
        )

    gradcam.remove()
    summary = {k: sum(v) / len(v) for k, v in agg.items() if v}
    logger.info("Résumé interprétabilité : %s", summary)
    with mlflow.start_run(run_name="interpretability-eval"):
        mlflow.log_metrics(summary)
    print(f"\n=== Interprétabilité (moyenne sur {n} images) ===")
    print(f"{'Méthode':<18}{'Fidélité (AUC↓)':<18}{'Stabilité (↑)':<14}")
    print(f"{'By design':<18}{summary['bydesign_faith']:<18.4f}{summary['bydesign_stab']:<14.4f}")
    print(f"{'Grad-CAM':<18}{summary['gradcam_faith']:<18.4f}{summary['gradcam_stab']:<14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    main(args.checkpoint, args.n)
