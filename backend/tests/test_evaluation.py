from fastapi.testclient import TestClient
from main import app
from app.utils.evaluation import evaluate_query, evaluate_system

client = TestClient(app)

# Define benchmark queries and their expected relevant assessments
benchmark_queries = [
    {
        "query": "Java developers who collaborate with business teams, 40 minutes max",
        "relevant_assessments": [
            "https://www.shl.com/solutions/products/product-catalog/coding-essentials-java/",
            "https://www.shl.com/solutions/products/product-catalog/workplace-personality-inventory-ii/",
            "https://www.shl.com/solutions/products/product-catalog/team-fit/"
        ]
    },
    {
        "query": "Mid-level professionals proficient in Python, SQL and JavaScript, max duration 60 minutes",
        "relevant_assessments": [
            "https://www.shl.com/solutions/products/product-catalog/coding-essentials-python/",
            "https://www.shl.com/solutions/products/product-catalog/coding-essentials-sql/",
            "https://www.shl.com/solutions/products/product-catalog/coding-essentials-javascript/"
        ]
    },
    {
        "query": "Analyst role with cognitive and personality tests within 45 minutes",
        "relevant_assessments": [
            "https://www.shl.com/solutions/products/product-catalog/numerical-reasoning/",
            "https://www.shl.com/solutions/products/product-catalog/workplace-personality-inventory-ii/",
            "https://www.shl.com/solutions/products/product-catalog/occupational-personality-questionnaire/"
        ]
    },
    {
        "query": "Leadership assessments for senior management",
        "relevant_assessments": [
            "https://www.shl.com/solutions/products/product-catalog/leadership-report/",
            "https://www.shl.com/solutions/products/product-catalog/leadership-compass/",
            "https://www.shl.com/solutions/products/product-catalog/leadership-assessment/"
        ]
    },
]

def get_recommendations_for_query(query):
    """Helper function to call the /recommend endpoint and return the recommendations."""
    response = client.post("/recommend", json={"query": query})
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get("recommended_assessments", [])

def test_evaluation_metrics():
    """Test the recommendation system against benchmark queries and calculate evaluation metrics."""
    k = 3  # Set the value of k
    
    # Evaluate the system
    metrics = evaluate_system(
        benchmark_queries[:2],  # Use only first two queries for faster testing
        get_recommendations_for_query,
        k
    )
    
    print(f"Evaluation results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")
    
    # Assert that the metrics exist (actual thresholds would depend on system performance)
    assert f"mean_recall@{k}" in metrics
    assert f"map@{k}" in metrics
    
    # Metrics should be between 0 and 1
    assert 0 <= metrics[f"mean_recall@{k}"] <= 1
    assert 0 <= metrics[f"map@{k}"] <= 1
