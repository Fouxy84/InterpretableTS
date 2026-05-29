"""
Entraînement avec suivi MLflow (paramètres, métriques, modèle, artefacts
d'explication). Reproductible (seed fixé, config loggée).

Usage :
    python -m ivml.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml

from ivml.data import DataConfig, get_dataloaders
from ivml.model import ModelConfig, build_model
from ivml.utils import get_device, get_logger, set_seed

logger = get_logger("ivml.train")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"val_loss": loss_sum / total, "val_accuracy": correct / total}


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    logger.info("Device : %s", device)

    train_dl, test_dl = get_dataloaders(DataConfig(**cfg["data"]))
    model = build_model(ModelConfig(**cfg["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(cfg.get("experiment_name", "interpretable-vision"))
    with mlflow.start_run():
        mlflow.log_params({**cfg["model"], **cfg["optimizer"], **cfg["train"], "seed": cfg.get("seed", 42)})

        best_acc = 0.0
        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            for images, labels in train_dl:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            metrics = evaluate(model, test_dl, device)
            mlflow.log_metrics(metrics, step=epoch)
            logger.info(
                "Epoch %d | val_loss=%.4f | val_acc=%.4f",
                epoch, metrics["val_loss"], metrics["val_accuracy"],
            )
            if metrics["val_accuracy"] > best_acc:
                best_acc = metrics["val_accuracy"]
                Path("artifacts").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "artifacts/best_model.pt")

        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.pytorch.log_model(model, name="model")
        if Path("artifacts/best_model.pt").exists():
            mlflow.log_artifact("artifacts/best_model.pt")
        logger.info("Terminé. Meilleure accuracy : %.4f", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
