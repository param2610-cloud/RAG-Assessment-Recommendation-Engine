import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, CORS_ORIGINS
from app.api.endpoints import router as api_router
from app.services.data import prepare_data_pipeline
from app.services.search import process_user_query

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description='Assessment Recommendation System')
    parser.add_argument('--prepare', type=str, help='Path to CSV file to prepare data pipeline')
    parser.add_argument('--query', type=str, help='Query string for assessment search')
    parser.add_argument('--db_path', type=str, default="database/shl_vector_db", 
                      help='Path to the vector database directory')
    parser.add_argument('--api', action='store_true', help='Run as FastAPI server')
    
    args = parser.parse_args()
    
    if args.prepare:
        # Prepare data pipeline
        prepare_data_pipeline(args.prepare, args.db_path)
    elif args.query:
        # Process user query
        result = process_user_query(args.query, args.db_path)
        print(result)
    elif args.api:
        # Run as FastAPI server
        print("Starting FastAPI server...")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    else:
        # Interactive mode
        print("Assessment Recommendation System")
        print("Enter 'exit' to quit")
        while True:
            query = input("\nEnter your query (or URL to job listing): ")
            if query.lower() == 'exit':
                break
                
            result = process_user_query(query, args.db_path)
            print(result)

if __name__ == "__main__":
    main()