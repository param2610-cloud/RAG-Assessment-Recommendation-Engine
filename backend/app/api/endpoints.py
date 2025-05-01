import re
from fastapi import APIRouter, Query as QueryParam
from app.api.models import SearchResponse, AssessmentResult
from app.services.search import search_assessments
from app.utils.helpers import extract_url_from_query
from app.services.extraction import extract_job_description
from app.services.generation import generate_search_query

router = APIRouter()

@router.get("/search", response_model=SearchResponse)
async def search(
    query: str = QueryParam(..., description="Natural language query or job description URL"),
    is_url: bool = QueryParam(False, description="Whether the query is a URL to a job listing"),
    max_results: int = QueryParam(5, description="Maximum number of results to return", ge=1, le=10)
):
    """
    Search for assessments based on a natural language query or job description URL.
    
    - If is_url=True, the system will extract the job description from the URL and generate a search query.
    - If is_url=False, the query will be directly used to search for assessments.
    """
    # Process query based on whether it's a URL or direct query
    if is_url:
        # Extract URL from query if not already a URL
        url = query if query.startswith(('http://', 'https://')) else extract_url_from_query(query)
        
        if not url:
            return SearchResponse(
                search_query="",
                original_query=query,
                is_url=is_url,
                results=[]
            )
            
        # Extract job description from URL
        job_description = extract_job_description(url)
        if job_description.startswith("Error"):
            return SearchResponse(
                search_query=job_description,
                original_query=query,
                is_url=is_url,
                job_description_url=url,
                results=[]
            )
            
        # Generate search query based on job description
        search_query = generate_search_query(job_description)
        
        # Incorporate any time constraints from the original query
        time_pattern = r'(\d+)\s*minutes'
        time_match = re.search(time_pattern, query.lower())
        if time_match:
            max_duration = time_match.group(0)
            if "time" not in search_query.lower() and "minute" not in search_query.lower():
                search_query += f" Assessment duration less than {max_duration}."
    else:
        # Use the query directly
        url = None
        search_query = query
    
    try:
        # Search for assessments using existing function
        results = search_assessments(search_query, persist_directory="database/shl_vector_db")
        
        # Format results according to the response model
        formatted_results = []
        for result in results[:max_results]:  # Limit to max_results
            # Extract test types
            test_types = [t.replace('test_type_', '') for t in result.metadata 
                         if t.startswith('test_type_') and result.metadata[t]]
            
            # Create AssessmentResult object
            assessment = AssessmentResult(
                name=result.metadata.get('name', 'N/A'),
                url=result.metadata.get('url', 'N/A'),
                description=result.page_content[:500],  # Limit description length
                duration=float(result.metadata.get('duration', 0)),
                test_types=test_types
            )
            formatted_results.append(assessment)
        
        # Return the response
        return SearchResponse(
            search_query=search_query,
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=formatted_results
        )
        
    except Exception as e:
        # Return empty results with error message as search query
        return SearchResponse(
            search_query=f"Error searching for assessments: {str(e)}",
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=[]
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns a simple status message with the API version.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Assessment Recommendation System API"
    }