"""
Chargement des données (CIFAR-10 par défaut, échangeable par config).

CIFAR-10 se télécharge automatiquement via torchvision → pipeline reproductible
sans étape manuelle. Pour des cartes d'attention plus lisibles, basculer sur
Imagenette en surchargeant `dataset_name` (cf. configs/config.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "avion", "auto", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
)


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 32


def _transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list = []
    if image_size != 32:
        ops.append(transforms.Resize((image_size, image_size)))
    if train:
        ops += [transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(ops)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = datasets.CIFAR10(
        cfg.data_dir, train=True, download=True, transform=_transforms(cfg.image_size, True)
    )
    test_ds = datasets.CIFAR10(
        cfg.data_dir, train=False, download=True, transform=_transforms(cfg.image_size, False)
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_dl, test_dl


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse la normalisation pour la visualisation."""
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)
