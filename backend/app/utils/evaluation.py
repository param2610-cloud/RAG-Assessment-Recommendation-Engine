from typing import List, Dict, Any, Tuple, Union
import numpy as np

def calculate_recall_at_k(recommended_urls: List[str], 
                         relevant_urls: List[str], 
                         k: int = 3) -> float:
    """
    Calculate Recall@K for a single query.
    
    Args:
        recommended_urls: List of URLs of assessments recommended by the system
        relevant_urls: List of URLs of assessments that are considered relevant
        k: The number of top results to consider
    
    Returns:
        Recall@K value (between 0 and 1)
    """
    if not relevant_urls:
        return 0.0
    
    # Consider only top k recommendations
    recommended_urls = recommended_urls[:k]
    
    # Count how many relevant assessments are in the recommendations
    relevant_found = sum(1 for url in recommended_urls if url in relevant_urls)
    
    # Calculate recall
    recall_at_k = relevant_found / len(relevant_urls)
    
    return recall_at_k

def calculate_precision_at_k(recommended_urls: List[str], 
                            relevant_urls: List[str], 
                            k: int) -> float:
    """
    Calculate Precision@K for a single position k.
    
    Args:
        recommended_urls: List of URLs of assessments recommended by the system
        relevant_urls: List of URLs of assessments that are considered relevant
        k: The position to calculate precision at (1-indexed)
    
    Returns:
        Precision@K value (between 0 and 1)
    """
    if k <= 0 or k > len(recommended_urls):
        return 0.0
    
    # Consider only top k recommendations
    top_k_urls = recommended_urls[:k]
    
    # Count how many of the top-k recommendations are relevant
    relevant_found = sum(1 for url in top_k_urls if url in relevant_urls)
    
    # Calculate precision
    precision_at_k = relevant_found / k
    
    return precision_at_k

def calculate_ap_at_k(recommended_urls: List[str], 
                     relevant_urls: List[str], 
                     k: int = 3) -> float:
    """
    Calculate Average Precision@K for a single query.
    
    Note:
        Within this project, AP@K follows the simplified interpretation used in
        the benchmark tests: it measures the proportion of relevant items found
        within the top-k recommendations (equivalent to Precision@K when
        k ≤ len(recommended_urls)). This differs from the traditional IR
        definition that averages precision over every relevant hit position.
    
    Args:
        recommended_urls: List of URLs of assessments recommended by the system
        relevant_urls: List of URLs of assessments that are considered relevant
        k: The number of top results to consider
    
    Returns:
        AP@K value (between 0 and 1)
    """
    if not relevant_urls or k <= 0:
        return 0.0

    top_k_recommendations = recommended_urls[:k]
    if not top_k_recommendations:
        return 0.0

    relevant_hits = sum(1 for url in top_k_recommendations if url in relevant_urls)

    return relevant_hits / len(top_k_recommendations)

def evaluate_query(recommended_assessments: List[Dict[str, Any]], 
                  relevant_assessments: List[str],
                  k: int = 3) -> Dict[str, float]:
    """
    Evaluate recommendations for a single query using Recall@K and AP@K metrics.
    
    Args:
        recommended_assessments: List of assessment objects returned by the API
        relevant_assessments: List of URLs of assessments that are considered relevant
        k: The number of top results to consider
    
    Returns:
        Dictionary with recall@k and ap@k metrics
    """
    # Extract URLs from recommended assessments
    recommended_urls = [assessment["url"] for assessment in recommended_assessments]
    
    # Calculate metrics
    recall_at_k = calculate_recall_at_k(recommended_urls, relevant_assessments, k)
    ap_at_k = calculate_ap_at_k(recommended_urls, relevant_assessments, k)
    
    return {
        f"recall@{k}": recall_at_k,
        f"ap@{k}": ap_at_k
    }

def evaluate_system(queries_with_relevance: List[Dict[str, Any]], 
                   get_recommendations_func, 
                   k: int = 3) -> Dict[str, float]:
    """
    Evaluate the recommendation system against a set of benchmark queries.
    
    Args:
        queries_with_relevance: List of dicts with 'query' and 'relevant_assessments' keys
        get_recommendations_func: Function that takes a query and returns recommendations
        k: The number of top results to consider
    
    Returns:
        Dictionary with mean_recall@k and map@k metrics
    """
    all_metrics = []
    
    for benchmark in queries_with_relevance:
        query = benchmark["query"]
        relevant_assessments = benchmark["relevant_assessments"]
        
        # Get recommendations
        recommended_assessments = get_recommendations_func(query)
        
        # Calculate metrics for this query
        metrics = evaluate_query(recommended_assessments, relevant_assessments, k)
        all_metrics.append(metrics)
    
    # Calculate mean metrics across all queries
    mean_recall_at_k = np.mean([m[f"recall@{k}"] for m in all_metrics])
    map_at_k = np.mean([m[f"ap@{k}"] for m in all_metrics])
    
    return {
        f"mean_recall@{k}": float(mean_recall_at_k),  # Convert numpy types to Python types
        f"map@{k}": float(map_at_k)
    }
