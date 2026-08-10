"""Backend test suite."""
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add backend and root to sys.path
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(backend_root, ".."))
engine_root = os.path.abspath(os.path.join(workspace_root, "crafter-ai-engine"))

for p in [backend_root, workspace_root, engine_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_training_status():
    res = client.get("/api/training/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "global_step" in data


def test_training_metrics():
    res = client.get("/api/training/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_steps" in data
    assert "crafter_score" in data


def test_episodes_list():
    res = client.get("/api/episodes?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_models_list():
    res = client.get("/api/models")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_achievements_list():
    res = client.get("/api/achievements")
    assert res.status_code == 200
    data = res.json()
    assert "achievements" in data
    assert len(data["achievements"]) == 22


def test_experiments_list():
    res = client.get("/api/experiments")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
