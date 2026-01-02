# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent e-commerce intelligence system built with Python (FastAPI backend) and Node.js (frontend tooling). The system implements 5 core AI agents using RAG (Retrieval-Augmented Generation) for product recommendations, review analysis, content generation, conversational chatbots, and analytics.

## Development Commands

### Python Backend (Primary)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start development server

python src/main.py
# or with uvicorn directly
uvicorn src.main:app --reload

# Database operations (when implemented)
alembic upgrade head
python scripts/seed_database.py

# Testing (when implemented)
pytest
pytest-asyncio  # for async tests

# Code quality
black .         # formatting
flake8 .        # linting  
mypy .          # type checking
```

### Node.js Frontend Tooling (Secondary)
```bash
# Install Node.js dependencies
npm install

# Development
npm run dev     # watch mode
npm start       # production mode

# Testing & Quality
npm test        # jest
npm run lint    # eslint
npm run format  # prettier
```

## Architecture Overview

### Core Agents
1. **Product Recommendation Agent** - RAG-based personalized recommendations using vector embeddings
2. **Review Summarization Agent** - NLP sentiment analysis and review insights  
3. **Product Description Agent** - Automated SEO-optimized content generation
4. **Conversational Chatbot Agent** - Context-aware customer support with RAG
5. **Analytics & Insights Agent** - Business intelligence and performance metrics

### Technology Stack
- **Backend**: Python FastAPI with PostgreSQL and vector databases (Pinecone/Weaviate)
- **AI/ML**: OpenAI GPT-4, Claude, Sentence-Transformers for embeddings
- **Message Queue**: Redis/Celery for agent communication
- **Frontend**: React.js with TypeScript
- **Vector Search**: RAG implementation for semantic product search

### Key Data Sources
- Amazon Product Dataset (570M+ products)
- Amazon Review Dataset (233M+ reviews) 
- Customer support conversations for chatbot training
- Real-time user behavior and purchase history

## Code Organization

- `src/main.py` - FastAPI application entry point with basic health endpoints
- `docs/` - Comprehensive architecture and system design documentation
  - `architecture.md` - System overview and agent specifications
  - `rag-system.md` - Vector database and recommendation engine details
  - `chatbot-system.md` - Conversational AI design
  - `review-agent.md` - Review analysis specifications
- `config/` - Configuration files (currently empty)
- `tests/` - Test suites (currently empty)

## Development Patterns

### Agent Communication
Agents communicate through Redis/Celery message queues for asynchronous processing. Each agent is designed as a modular component that can be developed and deployed independently.

### Vector Database Integration
The system uses vector embeddings for semantic search across products and user preferences. Product embeddings combine title + description + key features using OpenAI text-embedding-3-large or sentence-transformers.

### RAG Implementation
Retrieval-Augmented Generation is implemented across multiple agents:
- Product recommendations use user preference embeddings + product catalog vectors
- Chatbot uses FAQ knowledge base + product information vectors  
- Review analysis uses review embeddings for theme extraction

## Performance Requirements
- <200ms recommendation response time
- Support for 100K+ concurrent users
- 95%+ sentiment classification accuracy
- 80%+ customer support automation rate