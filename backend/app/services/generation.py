from langchain_google_genai import GoogleGenerativeAI

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
    
    model = GoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
    response = model.invoke(prompt)
    return response.strip()