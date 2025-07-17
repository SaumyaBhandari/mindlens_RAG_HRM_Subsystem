# RAG Backend System
A backend system with RESTful APIs for document processing, vector storage, and RAG-based querying with agentic capabilities.

## Architecture

- **Framework**: FastAPI
- **Vector Database**: Qdrant
- **Relational Database**: PostgreSQL
- **Memory/Cache**: Redis
- **Embeddings**: Gemini, Sentence Transformer
- **Agent Framework**: LangChain
- **Email**: SMTP (Gmail)

## Setup
Set up environment variables in `.env`
```.env
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
EMAIL_TO=hr@palmmind.com
```

### Option 1: Run Locally:

1. Create virtual environment:
```bash
python -m venv env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
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

**Proper Details of API endpoints can be found on SwaggerUI:**
```
http://localhost:8000/docs#/
```
Thank you!