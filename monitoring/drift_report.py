"""
Détection de dérive (data drift) avec Evidently.

Compare un lot de référence (train) à un lot courant (production) sur des
statistiques d'images. Génère un rapport HTML versionnable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def image_features(images: np.ndarray) -> pd.DataFrame:
    """Statistiques simples par image : luminosité, contraste, moyennes RGB."""
    return pd.DataFrame(
        {
            "brightness": images.mean(axis=(1, 2, 3)),
            "contrast": images.std(axis=(1, 2, 3)),
            "r_mean": images[:, 0].mean(axis=(1, 2)),
            "g_mean": images[:, 1].mean(axis=(1, 2)),
            "b_mean": images[:, 2].mean(axis=(1, 2)),
        }
    )


def build_drift_report(reference: np.ndarray, current: np.ndarray, out: str = "drift_report.html") -> None:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=image_features(reference), current_data=image_features(current))
    report.save_html(out)
    print(f"Rapport de dérive écrit : {out}")
