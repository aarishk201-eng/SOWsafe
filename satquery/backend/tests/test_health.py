from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check_returns_ok():
    """Verify that GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    """Verify root documentation and status metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/health"
