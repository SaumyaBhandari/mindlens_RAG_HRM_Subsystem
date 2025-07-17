# RAG Backend System

A comprehensive backend system with RESTful APIs for document processing, vector storage, and RAG-based querying with agentic capabilities.

## Features

### API 1: Document Processing
- File upload support (.pdf, .txt)
- Multiple chunking strategies (recursive, semantic, custom)
- Multiple embedding models (SentenceTransformer, OpenAI)
- Vector storage in Qdrant
- Metadata storage in PostgreSQL

### API 2: RAG Agent System
- LangChain-based agentic system
- Memory layer with Redis
- Multiple similarity search algorithms
- Interview booking with email notifications
- Tool reasoning capabilities

## Architecture

- **Framework**: FastAPI
- **Vector Database**: Qdrant
- **Relational Database**: PostgreSQL
- **Memory/Cache**: Redis
- **Embeddings**: SentenceTransformer, OpenAI
- **Agent Framework**: LangChain
- **Email**: SMTP (Gmail)

## Setup

Set up environment variables in `.env`
`.env
DATABASE_URL=postgresql://username:password@localhost/rag_db
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=
GEMINI_API_KEY=
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=s
SMTP_PASSWORD=
EMAIL_FROM=
EMAIL_TO=trishnakc94@gmail.com
`

### Option 1: Run Locally:
1. Install dependencies:
```bash
pip install -r requirements.txt
```


3. Or run locally:
```bash
uvicorn main:app --reload
```
### Option 2: Run with Docker:
1. Run with Docker:
```bash
docker-compose up
```

## API Endpoints

### Document Processing
- `POST /api/v1/upload` - Upload and process documents
- `GET /api/v1/files` - List uploaded files

### RAG Query System
- `POST /api/v1/query` - Query documents with RAG agent
- `POST /api/v1/book-interview` - Book interview appointments
- `GET /api/v1/bookings` - List bookings

### Health Check
- `GET /api/v1/health` - System health status
