import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API settings
API_TITLE = "Assessment Recommendation System API"
API_DESCRIPTION = "API for searching and recommending SHL assessments"
API_VERSION = "1.0.0"

# Database settings
DB_PATH = os.getenv("DB_PATH", "database/shl_vector_db")

# CORS settings
CORS_ORIGINS = [
    "https://assessment-recommendation-engine.vercel.app",
    "https://shl-recommendation-engine.vercel.app",
    "http://localhost:3000"
]

# Model settings
EMBEDDING_MODEL = "models/embedding-001"
GENERATIVE_MODEL = "gemini-2.5-pro-exp-03-25"