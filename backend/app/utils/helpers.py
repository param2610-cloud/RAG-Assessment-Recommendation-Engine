import pandas as pd
import re

def clean_list_field(field):
    """Clean list fields that might be string representations of lists or comma-separated strings"""
    if pd.isna(field) or field == '':
        return []
    
    if isinstance(field, list):
        return field
    
    if isinstance(field, str) and field.startswith('[') and field.endswith(']'):
        try:
            # Try to eval it safely to convert string representation to actual list
            clean_str = field.replace("'", '"')
            result = eval(clean_str)
            if isinstance(result, list):
                return result
        except:
            pass
    
    if isinstance(field, str):
        return [item.strip() for item in field.split(',') if item.strip()]
    
    return []

def get_duration_range(duration):
    """Categorize duration into ranges for easier filtering"""
    try:
        duration = float(duration)
        if duration <= 15:
            return "very_short"
        elif duration <= 30:
            return "short"
        elif duration <= 45:
            return "medium"
        elif duration <= 60:
            return "standard"
        else:
            return "long"
    except (ValueError, TypeError):
        return "unknown"

def extract_filters_from_query(query: str):
    """Extract metadata filters from a natural language query."""
    # Filter logic implementation
    filter_conditions = []
    
    # Job level detection
    job_levels = [
        "analyst", "director", "entry-level", "executive", "front line manager",
        "general population", "graduate", "manager", "mid-professional", 
        "professional individual contributor", "supervisor"
    ]
    
    for level in job_levels:
        if re.search(r'\b' + re.escape(level) + r'\b', query.lower()):
            filter_conditions.append(
                {"job_level_{0}".format(level.replace(' ', '_').replace('-', '_')): True}
            )
    
    # Test type categories detection
    if re.search(r'\b(cognitive|ability|aptitude)\b', query.lower()):
        filter_conditions.append({"contains_cognitive": True})
    
    if re.search(r'\b(personality|behavior|behaviour)\b', query.lower()):
        filter_conditions.append({"contains_personality": True})
        
    if re.search(r'\b(technical|knowledge|skill)\b', query.lower()):
        filter_conditions.append({"contains_technical": True})
        
    if re.search(r'\b(soft skill|competenc|situational|judgment)\b', query.lower()):
        filter_conditions.append({"contains_soft_skill": True})
    
    # Duration detection
    duration_match = re.search(r'(\d+)\s*(min|mins|minutes)', query.lower())
    if duration_match:
        minutes = int(duration_match.group(1))
        if minutes <= 15:
            filter_conditions.append({"duration_range": "very_short"})
        elif minutes <= 30:
            filter_conditions.append({"duration_range": "short"})
        elif minutes <= 45:
            filter_conditions.append({"duration_under_45": True})
        elif minutes <= 60:
            filter_conditions.append({"duration_under_60": True})
        
    # Language detection
    languages = [
        "arabic", "chinese simplified", "chinese traditional", "czech", "danish",
        "dutch", "english", "estonian", "finnish", "flemish", "french", "german",
        "greek", "hungarian", "icelandic", "indonesian", "italian", "japanese",
        "korean", "latvian", "lithuanian", "malay", "norwegian", "polish",
        "portuguese", "romanian", "russian", "serbian", "slovak", "spanish",
        "swedish", "thai", "turkish", "vietnamese"
    ]
    
    for lang in languages:
        if re.search(r'\b' + re.escape(lang) + r'\b', query.lower()):
            clean_lang = lang.replace(' ', '_').replace('-', '_')
            filter_conditions.append({f"language_{clean_lang}": True})
    
    # Remote testing and adaptive features
    if "remote" in query.lower():
        filter_conditions.append({"remote_testing": True})
        
    if "adaptive" in query.lower():
        filter_conditions.append({"adaptive_irt": True})
    
    # Return proper ChromaDB filter structure
    if not filter_conditions:
        return None  # No filters found
    elif len(filter_conditions) == 1:
        return filter_conditions[0]  # Single filter
    else:
        return {"$or": filter_conditions}  # Multiple filters combined with OR

def extract_url_from_query(query):
    """Extract URLs from the user query."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, query)
    return urls[0] if urls else None