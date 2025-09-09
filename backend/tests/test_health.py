"""Tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TunedIn Music Recommender"
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "uptime" in data


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Should return Prometheus metrics format
    assert "request_duration_ms" in response.text


def test_status_endpoint():
    """Test detailed status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "healthy" in data
    assert "uptime" in data
    assert "dependencies" in data

