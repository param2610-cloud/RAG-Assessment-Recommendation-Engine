from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}  # Updated to match exact required format

def test_search_query():
    response = client.get("/search?query=personality%20test%20for%20manager&is_url=false")
    assert response.status_code == 200
    assert "search_query" in response.json()
    assert response.json()["is_url"] == False

def test_recommend_endpoint():
    # Test with a simple query
    request_data = {"query": "software developer python skills"}
    response = client.post("/recommend", json=request_data)
    
    # Check response format
    assert response.status_code == 200
    response_data = response.json()
    assert "recommended_assessments" in response_data
    
    # If there are any results, check their structure
    if response_data["recommended_assessments"]:
        assessment = response_data["recommended_assessments"][0]
        assert "url" in assessment
        assert "adaptive_support" in assessment
        assert "description" in assessment
        assert "duration" in assessment
        assert "remote_support" in assessment
        assert "test_type" in assessment
        assert isinstance(assessment["test_type"], list)

def test_evaluation_metrics_calculation():
    """Test that evaluation metrics are calculated correctly."""
    from app.utils.evaluation import calculate_recall_at_k, calculate_ap_at_k
    
    # Simple test case
    recommended_urls = ["url1", "url2", "url3"]
    relevant_urls = ["url1", "url3", "url4"]
    
    # Calculate metrics
    recall = calculate_recall_at_k(recommended_urls, relevant_urls, k=3)
    ap = calculate_ap_at_k(recommended_urls, relevant_urls, k=3)
    
    # Check results
    assert recall == 2/3, f"Expected Recall@3 = 2/3, got {recall}"
    # AP calculation: (1/1 + 0 + 3/3) / 3 = 2/3
    assert abs(ap - 2/3) < 0.001, f"Expected AP@3 ≈ 2/3, got {ap}"

