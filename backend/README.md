backend/
├── app/                    # Main application package
│   ├── __init__.py
│   ├── api/                # API related modules
│   │   ├── __init__.py
│   │   ├── endpoints.py    # FastAPI endpoints
│   │   └── models.py       # Pydantic models
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   └── config.py       # Configuration settings
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── search.py       # Assessment search functionality
│   │   ├── data.py         # Data processing functions
│   │   ├── extraction.py   # Job description extraction
│   │   └── generation.py   # Query generation
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── helpers.py      # Helper functions
├── database/               # Keep existing structure
├── others/                 # Keep existing structure
├── tests/                  # Add tests folder
│   ├── __init__.py
│   └── test_api.py
├── main.py                 # Simplified entry point
├── .env
└── requirements.txt


# Run the API server
python main.py --api

# Process a query
python main.py --query "personality test for manager position"

# Prepare data pipeline from CSV
python main.py --prepare assessment.csv