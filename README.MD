# 🧩 SHL Assessment Recommendation System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

<p align="center">
  <img src="https://placekitten.com/800/200" alt="SHL Assessment Recommendation System" width="800">
  <br>
  <em>AI-powered assessment recommendation tool for HR professionals and recruiters</em>
</p>

## 📑 Table of Contents
- [Introduction](#introduction)
- [System Description](#system-description)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [Data Requirements](#data-requirements)
- [External Interface Requirements](#external-interface-requirements)
- [System Architecture](#system-architecture)
- [Constraints and Limitations](#constraints-and-limitations)
- [Optimization Efforts](OPTIMIZATION.md)
- [Appendices](#appendices)

## 🚀 Introduction

### Purpose
This Software Requirements Specification (SRS) document describes the requirements for the SHL Assessment Recommendation System, a web application designed to help HR professionals and recruiters find appropriate SHL assessments based on job descriptions or specific requirements.

### Scope
The SHL Assessment Recommendation System is an AI-powered tool that:

- Recommends suitable assessment tests from SHL's catalog
- Processes both natural language queries and job description URLs
- Provides detailed information about recommended assessments
- Allows for filtering and customization of search results

### Intended Audience
- Development team
- SHL stakeholders
- QA and testing team
- System administrators and maintainers

### Definitions and Acronyms
| Term | Definition |
|------|------------|
| SHL | A global provider of talent assessment solutions |
| IRT | Item Response Theory, an adaptive testing methodology |
| API | Application Programming Interface |
| UI | User Interface |
| SRS | Software Requirements Specification |

## 🔍 System Description

### System Context
The system serves as an intermediary between users (HR professionals/recruiters) and SHL's assessment catalog. It processes user requirements and matches them to appropriate assessment tests using natural language processing and vector search technologies.

### Product Features
- 📋 Job description analysis
- 💬 Natural language query processing
- 🤖 AI-powered assessment recommendations
- 📊 Detailed assessment information display
- 🔗 URL parsing for job descriptions
- ⚙️ Customizable search parameters

### User Classes and Characteristics
- **Primary Users**: HR professionals and recruiters seeking appropriate assessment tools
- **Secondary Users**: SHL administrators and content managers
- **User Experience Level**: Basic technical knowledge, familiarity with recruitment tools

## ⚙️ Functional Requirements

### User Interface
- FR-1.1: The system shall provide a clean, responsive web interface
- FR-1.2: The system shall include a search form for entering requirements
- FR-1.3: The system shall include a toggle for URL/text-based input
- FR-1.4: The system shall provide sample queries for user guidance
- FR-1.5: The system shall provide a results table displaying assessment details
- FR-1.6: The system shall allow users to specify the maximum number of results (1-10)
- FR-1.7: The system shall display loading indicators during processing

### Search Functionality
- FR-2.1: The system shall process natural language queries about job requirements
- FR-2.2: The system shall extract job descriptions from provided URLs
- FR-2.3: The system shall generate search queries from extracted job descriptions
- FR-2.4: The system shall detect time constraints in user queries
- FR-2.5: The system shall match query requirements to appropriate assessment types
- FR-2.6: The system shall return error messages for invalid URLs or failed searches
- FR-2.7: The system shall provide feedback when no matching assessments are found

### Assessment Recommendations
- FR-3.1: The system shall recommend assessments based on job roles and levels
- FR-3.2: The system shall recommend assessments based on required skills/languages
- FR-3.3: The system shall recommend assessments based on required assessment types
- FR-3.4: The system shall consider duration constraints in recommendations
- FR-3.5: The system shall provide information about remote testing availability
- FR-3.6: The system shall provide information about adaptive testing capabilities
- FR-3.7: The system shall rank recommendations by relevance to the query

## 🔒 Non-Functional Requirements

### Performance
- NFR-1.1: The system shall return search results within 5 seconds for direct queries
- NFR-1.2: The system shall process URL inputs and return results within 10 seconds
- NFR-1.3: The system shall handle up to 100 concurrent users
- NFR-1.4: The database operations shall support fast vector similarity search

### Security
- NFR-2.1: The system shall sanitize all user inputs to prevent injection attacks
- NFR-2.2: The API endpoints shall validate request parameters
- NFR-2.3: The system shall implement appropriate CORS protections
- NFR-2.4: External API credentials shall be stored securely using environment variables

### Usability
- NFR-3.1: The user interface shall be intuitive and require minimal training
- NFR-3.2: The system shall provide helpful error messages in user-friendly language
- NFR-3.3: The system shall provide sample queries to guide users
- NFR-3.4: The system shall be compatible with mobile and desktop browsers
- NFR-3.5: The system shall follow SHL brand guidelines for visual design

### Reliability
- NFR-4.1: The system shall be available 99.9% of the time
- NFR-4.2: The system shall gracefully handle network failures
- NFR-4.3: The system shall implement appropriate error handling and logging

### Scalability
- NFR-5.1: The system shall support future growth of the assessment catalog
- NFR-5.2: The backend shall be designed for horizontal scaling
- NFR-5.3: The database shall efficiently handle increasing vector data

## 💾 Data Requirements

### Assessment Data Model
Each assessment shall contain:
- Name
- URL
- Description
- Job levels (array)
- Languages (array)
- Duration (minutes)
- Test types (array)
- Remote testing availability (boolean)
- Adaptive IRT availability (boolean)

### Test Type Categories
The system shall support the following test type categories:
- A: Ability & Aptitude
- B: Biodata & Situational Judgment
- C: Competencies
- D: Development and 360
- E: Assessment Exercises
- K: Knowledge & Skills
- P: Personality & Behavior
- S: Simulation

### Vector Database
- DR-3.1: The system shall store assessment data in a vector database
- DR-3.2: Each assessment shall have associated metadata for filtering
- DR-3.3: Each assessment shall have embeddings for semantic search

### Data Processing
- DR-4.1: The system shall clean and validate incoming assessment data
- DR-4.2: The system shall convert string representations to appropriate data types
- DR-4.3: The system shall handle missing or null values appropriately
- DR-4.4: The system shall standardize data formatting

## 🌐 External Interface Requirements

### User Interfaces
- IR-1.1: The web interface shall be responsive and work on various screen sizes
- IR-1.2: The system shall implement SHL's brand colors and styling
- IR-1.3: The interface shall provide clear feedback during operations
- IR-1.4: The interface shall display assessment details in a clear, tabular format

### API Interfaces
- IR-2.1: The backend shall expose a RESTful API for the frontend
- IR-2.2: The API shall support the following endpoints:
  - GET /search: Search for assessments
  - GET /health: Check system health
- IR-2.3: The search endpoint shall accept the following parameters:
  - query: String (required)
  - is_url: Boolean (optional, default: false)
  - max_results: Integer (optional, default: 5)

### External Services
- IR-3.1: The system shall connect to Google Generative AI for embeddings
- IR-3.2: The system shall use appropriate HTTP libraries for URL scraping
- IR-3.3: The system shall handle external service failures gracefully

## 🏗️ System Architecture

### Frontend
- Built with React and TypeScript
- Implements responsive design using Tailwind CSS
- Communicates with backend via RESTful API calls
- Components:
  - Search form for input
  - Results display table
  - Loading indicators
  - Error messages
  - Navigation components

### Backend
- Built with Python using FastAPI framework
- Implements vector search using Chroma database
- Uses Google Generative AI for embeddings
- Modules:
  - API endpoints
  - Data processing
  - Search services
  - Job description extraction
  - Query generation

### Database
- Uses Chroma vector database for storing assessment data
- Stores both vector embeddings and metadata for filtering
- Supports similarity search and metadata filtering

## ⚠️ Constraints and Limitations

### Technical Constraints
- CL-1.1: The system must be compatible with modern web browsers
- CL-1.2: The system depends on external AI services for embeddings
- CL-1.3: The system requires internet connectivity for URL processing
- CL-1.4: The backend services must run in a Python environment

### Business Constraints
- CL-2.1: The system must comply with SHL's branding guidelines
- CL-2.2: The system must accurately represent SHL's assessment catalog
- CL-2.3: The system should be maintainable for future assessment updates

## 📎 Appendices

### Data Schema
Assessment data includes:
```json
{
  "name": "string",
  "url": "string",
  "description": "string",
  "job_levels": ["string"],
  "languages": ["string"],
  "duration": "number",
  "test_type": ["string"],
  "remote_testing": "boolean",
  "adaptive_irt": "boolean"
}
```

### Deployment Requirements
- **Frontend**: Node.js environment for React application
- **Backend**: Python 3.x environment with required dependencies
- **Database**: Storage system supporting Chroma vector database
- **Environment variables** for API keys and service configuration

### Development Environment
| Component | Technologies |
|-----------|--------------|
| Frontend | Vite, React, TypeScript, Tailwind CSS |
| Backend | Python, FastAPI, Langchain, BeautifulSoup |
| Version control | Git |
| Package management | npm (frontend), pip (backend) |

