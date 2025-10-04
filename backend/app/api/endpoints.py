import logging
import re
from fastapi import APIRouter, Query as QueryParam, Body
from app.api.models import SearchResponse, AssessmentResult
from app.services.search import search_assessments
from app.utils.helpers import extract_url_from_query
from app.services.extraction import extract_job_description
from app.services.generation import generate_search_query
from pydantic import BaseModel
from typing import List

router = APIRouter()
logger = logging.getLogger(__name__)

# Define models for recommendation endpoint
class RecommendationRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

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
    logger.debug("Received /search request", extra={
        "query": query,
        "is_url": is_url,
        "max_results": max_results
    })
    # Process query based on whether it's a URL or direct query
    if is_url:
        logger.debug("Processing query as URL", extra={"raw_query": query})
        # Extract URL from query if not already a URL
        url = query if query.startswith(('http://', 'https://')) else extract_url_from_query(query)
        
        if not url:
            logger.warning("Failed to extract URL from query", extra={"query": query})
            return SearchResponse(
                search_query="",
                original_query=query,
                is_url=is_url,
                results=[]
            )
            
        # Extract job description from URL
        job_description = extract_job_description(url)
        logger.debug("Extracted job description", extra={
            "job_description_preview": job_description[:120] if job_description else "",
            "job_description_length": len(job_description) if job_description else 0,
            "job_description_source_url": url
        })
        if job_description.startswith("Error"):
            logger.error("Error extracting job description", extra={
                "url": url,
                "error": job_description
            })
            return SearchResponse(
                search_query=job_description,
                original_query=query,
                is_url=is_url,
                job_description_url=url,
                results=[]
            )
            
        # Generate search query based on job description
        search_query = generate_search_query(job_description)
        logger.debug("Generated search query from job description", extra={
            "search_query": search_query,
            "source_url": url
        })
        
        # Incorporate any time constraints from the original query
        time_pattern = r'(\d+)\s*minutes'
        time_match = re.search(time_pattern, query.lower())
        if time_match:
            max_duration = time_match.group(0)
            if "time" not in search_query.lower() and "minute" not in search_query.lower():
                search_query += f" Assessment duration less than {max_duration}."
                logger.debug("Appended time constraint from query", extra={
                    "max_duration": max_duration,
                    "final_search_query": search_query
                })
            else:
                logger.debug("Time constraint already present in generated query", extra={
                    "max_duration": max_duration
                })
    else:
        # Use the query directly
        url = None
        search_query = query
        logger.debug("Processing query as plain text", extra={
            "search_query": search_query
        })
    
    try:
        # Search for assessments using existing function
        logger.debug("Invoking search_assessments", extra={
            "search_query": search_query,
            "persist_directory": "database/vector_db"
        })
        results = search_assessments(search_query, persist_directory="database/vector_db")
        logger.debug("Received results from search_assessments", extra={
            "total_results": len(results) if results else 0
        })
        
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
            logger.debug("Formatted assessment result", extra={
                "assessment_name": assessment.name,
                "assessment_url": assessment.url,
                "duration": assessment.duration,
                "test_types": assessment.test_types
            })
        
        # Return the response
        logger.debug("Returning SearchResponse", extra={
            "search_query": search_query,
            "result_count": len(formatted_results),
            "job_description_url": url if is_url else None
        })
        return SearchResponse(
            search_query=search_query,
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=formatted_results
        )
        
    except Exception as e:
        logger.exception("Error while searching for assessments", extra={
            "search_query": search_query,
            "original_query": query,
            "is_url": is_url
        })
        # Return empty results with error message as search query
        return SearchResponse(
            search_query=f"Error searching for assessments: {str(e)}",
            original_query=query,
            is_url=is_url,
            job_description_url=url if is_url else None,
            results=[]
        )

@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Recommend assessments based on a job description or natural language query.
    
    Returns recommended relevant assessments based on the input query.
    """
    query = request.query
    
    # Process the query (reusing search logic)
    is_url = query.startswith(('http://', 'https://'))
    
    if is_url:
        url = query
        job_description = extract_job_description(url)
        if job_description.startswith("Error"):
            return RecommendationResponse(recommended_assessments=[])
        search_query = generate_search_query(job_description)
    else:
        search_query = query
    
    try:
        # Search for assessments
        results = search_assessments(search_query, persist_directory="database/vector_db")
        
        # Format results according to the required response format
        formatted_results = []
        for result in results[:10]:  # Limit to 10 results
            # Extract test types
            test_types = [t.replace('test_type_', '') for t in result.metadata 
                        if t.startswith('test_type_') and result.metadata[t]]
            
            # Create Assessment object with required format
            assessment = Assessment(
                url=result.metadata.get('url', 'N/A'),
                adaptive_support="Yes" if result.metadata.get('adaptive_support', False) else "No",
                description=result.page_content[:500],  # Limit description length
                duration=int(result.metadata.get('duration', 0)),
                remote_support="Yes" if result.metadata.get('remote_support', False) else "No",
                test_type=test_types if test_types else ["General"]
            )
            formatted_results.append(assessment)
        
        # Return the response
        return RecommendationResponse(recommended_assessments=formatted_results)
        
    except Exception as e:
        # Return empty results on error
        return RecommendationResponse(recommended_assessments=[])

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns a simple status message.
    """
    return {
        "status": "healthy"
    }