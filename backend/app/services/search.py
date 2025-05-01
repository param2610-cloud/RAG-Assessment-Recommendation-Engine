import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from app.utils.helpers import extract_filters_from_query, extract_url_from_query
from app.services.extraction import extract_job_description
from app.services.generation import generate_search_query

def search_assessments(query, persist_directory="database/shl_vector_db"):
    """Search for assessments based on the query."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Extract any time constraints from the query
    time_pattern = r'(\d+)\s*minutes'
    time_match = re.search(time_pattern, query.lower())
    max_duration = None
    
    if time_match:
        max_duration = int(time_match.group(1))
    
    # First try with filters
    filters = extract_filters_from_query(query)
    if filters:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=5,
            filter=filters
        )
        # Convert to Document objects
        filtered_results = [doc for doc, _ in results]
    else:
        # If no filters, perform standard search
        filtered_results = vector_store.similarity_search(
            query=query,
            k=5
        )
    
    # If we have time constraints, filter results manually
    if max_duration:
        duration_filtered = []
        for doc in filtered_results:
            duration = doc.metadata.get('duration')
            if duration and float(duration) <= max_duration:
                duration_filtered.append(doc)
        return duration_filtered if duration_filtered else filtered_results[:3]  # Return at least something
    
    return filtered_results

def process_user_query(user_query, persist_directory="database/shl_vector_db"):
    """Process user query with URL to extract job description and search for assessments."""
    # Extract URL from query
    url = extract_url_from_query(user_query)
    if not url:
        # If no URL, treat as direct search query
        print("No URL found. Treating as direct search query.")
        results = search_assessments(user_query, persist_directory)
        
        # Format results
        formatted_results = "\nRecommended Assessments:\n" + "-" * 40 + "\n"
        if not results:
            formatted_results += "No matching assessments found."
        else:
            for i, result in enumerate(results, 1):
                formatted_results += f"Assessment {i}:\n"
                formatted_results += f"Name: {result.metadata.get('name', 'N/A')}\n"
                formatted_results += f"Duration: {result.metadata.get('duration', 'N/A')} minutes\n"
                
                # Format test types
                test_types = [t.replace('test_type_', '') for t in result.metadata 
                              if t.startswith('test_type_') and result.metadata[t]]
                formatted_results += f"Test types: {', '.join(test_types) if test_types else 'N/A'}\n"
                
                formatted_results += f"URL: {result.metadata.get('url', 'N/A')}\n"
                formatted_results += f"Description: {result.page_content[:200]}...\n\n"
        
        return f"Search Query: \"{user_query}\"\n\n{formatted_results}"
    
    # Extract job description from URL
    print(f"Extracting job description from URL: {url}")
    job_description = extract_job_description(url)
    if job_description.startswith("Error"):
        return job_description
    
    # Generate search query based on job description
    print("Generating search query from job description...")
    search_query = generate_search_query(job_description)
    
    # Incorporate any time constraints from the original query
    time_pattern = r'(\d+)\s*minutes'
    time_match = re.search(time_pattern, user_query.lower())
    if time_match:
        max_duration = time_match.group(0)
        if "time" not in search_query.lower() and "minute" not in search_query.lower():
            search_query += f" Assessment duration less than {max_duration}."
    
    try:
        # Search for assessments
        print(f"Searching for assessments with query: {search_query}")
        results = search_assessments(search_query, persist_directory)
        
        # Format results
        formatted_results = "\nRecommended Assessments:\n" + "-" * 40 + "\n"
        if not results:
            formatted_results += "No matching assessments found."
        else:
            for i, result in enumerate(results, 1):
                formatted_results += f"Assessment {i}:\n"
                formatted_results += f"Name: {result.metadata.get('name', 'N/A')}\n"
                formatted_results += f"Duration: {result.metadata.get('duration', 'N/A')} minutes\n"
                
                # Format test types
                test_types = [t.replace('test_type_', '') for t in result.metadata 
                              if t.startswith('test_type_') and result.metadata[t]]
                formatted_results += f"Test types: {', '.join(test_types) if test_types else 'N/A'}\n"
                
                formatted_results += f"URL: {result.metadata.get('url', 'N/A')}\n"
                formatted_results += f"Description: {result.page_content[:200]}...\n\n"
        
        # Return summary
        summary = f"""
Job Description URL: {url}

Generated Search Query: "{search_query}"

{formatted_results}
        """
        return summary
        
    except Exception as e:
        return f"Error searching for assessments: {str(e)}"