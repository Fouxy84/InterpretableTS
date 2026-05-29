"""Test fumée de l'API."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
