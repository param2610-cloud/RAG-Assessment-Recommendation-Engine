import requests
from bs4 import BeautifulSoup

def extract_job_description(url):
    """Extract job description from a job listing webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # First try to find job description by common class names or IDs
        job_desc_selectors = [
            'div.description', 'div.job-description', '#job-description',
            '.job-details', '.description', '[data-test="job-description"]',
            'section.description', 'div.details', '.details-pane', 
            '.job-desc', '.show-more-less-html'
        ]
        
        for selector in job_desc_selectors:
            job_desc = soup.select_one(selector)
            if job_desc:
                return job_desc.get_text(separator='\n', strip=True)
        
        # If specific selectors fail, use a more generic approach
        # Find sections with job-related terms in them
        job_keywords = ['responsibilities', 'requirements', 'qualifications', 'about the job', 'job summary', 'what you&nbspll do', 'what we&nbspre looking for']
        
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong']):
            heading_text = heading.get_text().lower()
            if any(keyword in heading_text for keyword in job_keywords):
                # Get the next sibling elements which likely contain the job description
                description = []
                current = heading.find_next_sibling()
                while current and current.name not in ['h1', 'h2', 'h3', 'h4']:
                    if current.get_text(strip=True):
                        description.append(current.get_text(strip=True))
                    current = current.find_next_sibling()
                if description:
                    return '\n'.join(description)
        
        # If all else fails, get the main content area and try to extract job details
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        
        # Last resort: just get the page title and any text
        title = soup.title.string if soup.title else "Job Listing"
        return f"{title}{soup.get_text(strip=True)[:2000]}"
        
    except Exception as e:
        return f"Error extracting job description: {str(e)}"