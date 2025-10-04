import logging
import os
from langchain_google_genai import GoogleGenerativeAI

logger = logging.getLogger(__name__)

def generate_search_query(job_description):
    """Generate a search query based on job description using Gemini."""
    prompt = f"""
    Based on the following job description, create a concise search query to find appropriate
    assessment tests that would help screen candidates for this position.
    
    JOB DESCRIPTION:
    {job_description[:3000]}  # Limit to avoid token limits
    
    Focus on:
    1. Technical skills required
    2. Soft skills mentioned
    3. Any specific assessment requirements
    4. Time constraints for assessments if mentioned
    
    Return ONLY the search query, nothing else.
    """
    
    api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
    logger.debug("Invoking Gemini to generate search query", extra={
        "model": "gemini-2.5-pro-exp-03-25",
        "api_key_present": api_key_present,
        "job_description_length": len(job_description) if job_description else 0
    })

    try:
        model = GoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = model.invoke(prompt)
        cleaned_response = response.strip() if isinstance(response, str) else str(response)
        logger.debug("Gemini response received", extra={
            "response_preview": cleaned_response[:200]
        })
        return cleaned_response
    except Exception as exc:
        logger.exception("Gemini query generation failed", extra={
            "model": "gemini-2.5-pro-exp-03-25",
            "api_key_present": api_key_present
        })
        raise