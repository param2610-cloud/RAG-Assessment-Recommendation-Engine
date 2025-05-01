from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_search_query():
    response = client.get("/search?query=personality%20test%20for%20manager&is_url=false")
    assert response.status_code == 200
    assert "search_query" in response.json()
    assert response.json()["is_url"] == False

# Add more tests as needed