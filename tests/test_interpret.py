"""Tests des métriques d'interprétabilité."""
import torch

from ivml.interpret import by_design_explanation
from ivml.interpret.metrics import explanation_stability, faithfulness_deletion
from ivml.model import ModelConfig, build_model


def test_faithfulness_returns_valid_floats():
    model = build_model(ModelConfig())
    x = torch.randn(1, 3, 32, 32)
    exp = by_design_explanation(model, x)
    out = faithfulness_deletion(model, x, exp, steps=5)
    assert 0.0 <= out["base_confidence"] <= 1.0
    assert "deletion_auc" in out


def test_stability_score_in_range():
    model = build_model(ModelConfig())
    x = torch.randn(1, 3, 32, 32)
    out = explanation_stability(model, x, by_design_explanation, n_samples=3)
    assert 0.0 <= out["stability_score"] <= 1.0
