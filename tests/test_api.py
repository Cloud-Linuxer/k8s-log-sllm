"""Tests for API server."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client (shared across module for efficiency)."""
    from src.api.server import create_app

    app = create_app(
        model_path="models/v2/model.onnx",
        tokenizer_path="models/v2/tokenizer",
        num_threads=1,
        lazy_load=True,
    )
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_health_check(self, client):
        """Test health endpoint."""
        # First make a prediction to load model
        client.post("/predict", json={"log": "test"})

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_single(self, client):
        """Test single prediction."""
        response = client.post(
            "/predict",
            json={"log": "INFO nginx started successfully"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "risk_label" in data
        assert "risk_score" in data
        assert "level" in data
        assert 0 <= data["risk_label"] <= 10
        assert data["level"] in ["trace", "debug", "info", "notice", "warning", "error", "critical", "emergency"]

    def test_predict_single_error_log(self, client):
        """Test prediction for error log."""
        response = client.post(
            "/predict",
            json={"log": "ERROR kernel panic - not syncing: VFS"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["risk_label"] >= 8  # Should be high risk
        assert data["level"] in ["critical", "emergency"]

    def test_predict_single_debug_log(self, client):
        """Test prediction for debug log."""
        response = client.post(
            "/predict",
            json={"log": "DEBUG entering function processRequest"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["risk_label"] <= 2  # Should be low risk
        assert data["level"] in ["trace", "debug", "info"]

    def test_predict_single_no_preprocess(self, client):
        """Test prediction without preprocessing."""
        response = client.post(
            "/predict",
            json={"log": "error something failed", "preprocess": False},
        )
        assert response.status_code == 200
        assert "risk_label" in response.json()

    def test_predict_batch(self, client):
        """Test batch prediction."""
        logs = [
            "INFO nginx GET /health 200",
            "ERROR connection refused",
            "CRITICAL OOMKilled container",
        ]

        response = client.post(
            "/predict/batch",
            json={"logs": logs},
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "count" in data
        assert data["count"] == 3
        assert len(data["results"]) == 3

        # Check each result
        for result in data["results"]:
            assert "risk_label" in result
            assert "risk_score" in result
            assert "level" in result

    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"logs": []},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_predict_batch_risk_ordering(self, client):
        """Test that risk levels are correctly ordered."""
        logs = [
            "DEBUG trace message",
            "INFO normal operation",
            "WARNING disk space low",
            "ERROR connection failed",
            "CRITICAL kernel panic",
        ]

        response = client.post(
            "/predict/batch",
            json={"logs": logs},
        )
        assert response.status_code == 200

        results = response.json()["results"]

        # Generally, risk should increase with severity
        # DEBUG/INFO should be lower than ERROR/CRITICAL
        debug_risk = results[0]["risk_label"]
        info_risk = results[1]["risk_label"]
        critical_risk = results[4]["risk_label"]

        assert debug_risk <= 2
        assert info_risk <= 3
        assert critical_risk >= 8

    def test_predict_invalid_request(self, client):
        """Test invalid request handling."""
        # Missing required field
        response = client.post(
            "/predict",
            json={},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_batch_invalid_request(self, client):
        """Test invalid batch request handling."""
        # Missing required field
        response = client.post(
            "/predict/batch",
            json={},
        )
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance."""

    def test_batch_performance(self, client):
        """Test batch processing doesn't timeout."""
        import time

        logs = ["INFO test message"] * 100

        start = time.time()
        response = client.post(
            "/predict/batch",
            json={"logs": logs},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert response.json()["count"] == 100
        assert elapsed < 10  # Should complete within 10 seconds


class TestRiskLevels:
    """Test risk level classification."""

    @pytest.mark.parametrize("log,min_risk,max_risk", [
        ("TRACE entering function", 0, 1),
        ("DEBUG processing request", 0, 2),
        ("INFO nginx started", 1, 3),
        ("WARNING disk space low", 3, 6),
        ("ERROR connection refused", 5, 8),
        ("CRITICAL OOMKilled", 8, 10),
        ("FATAL kernel panic", 9, 10),
    ])
    def test_risk_level_ranges(self, client, log, min_risk, max_risk):
        """Test that logs are classified within expected ranges."""
        response = client.post(
            "/predict",
            json={"log": log},
        )
        assert response.status_code == 200

        risk_label = response.json()["risk_label"]
        assert min_risk <= risk_label <= max_risk, \
            f"Log '{log}' got risk {risk_label}, expected {min_risk}-{max_risk}"
