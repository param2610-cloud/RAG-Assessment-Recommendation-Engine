from typing import List, Optional
from pydantic import BaseModel

class AssessmentResult(BaseModel):
    name: str
    url: str
    description: str
    duration: float
    test_types: List[str]

class SearchResponse(BaseModel):
    search_query: str
    original_query: str
    is_url: bool
    job_description_url: Optional[str] = None
    results: List[AssessmentResult]