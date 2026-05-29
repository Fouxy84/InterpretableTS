"""Tests du modèle interprétable."""
import torch

from ivml.model import ModelConfig, build_model


def test_forward_output_shape():
    model = build_model(ModelConfig(n_classes=10))
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_attention_is_normalized():
    """L'attention spatiale doit sommer à 1 par échantillon (softmax spatial)."""
    model = build_model(ModelConfig())
    x = torch.randn(2, 3, 32, 32)
    _, attn = model(x, return_attention=True)
    sums = attn.flatten(1).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_explain_shape_and_range():
    model = build_model(ModelConfig())
    x = torch.randn(3, 3, 32, 32)
    exp = model.explain(x)
    assert exp.shape == (3, 32, 32)
    assert float(exp.min()) >= 0.0 and float(exp.max()) <= 1.0
