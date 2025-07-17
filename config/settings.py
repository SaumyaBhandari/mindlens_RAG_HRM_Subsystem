import os
from typing import Optional
from dotenv import load_dotenv


load_dotenv()

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/rag_db")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Email
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: Optional[str] = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    EMAIL_FROM: Optional[str] = os.getenv("EMAIL_FROM")
    EMAIL_TO: Optional[str] = os.getenv("EMAIL_TO")
    
    # Chunking
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.7
    
    # Vector Search
    DEFAULT_SEARCH_LIMIT: int = 5
    
    # Memory
    CONVERSATION_HISTORY_LIMIT: int = 20
    MEMORY_EXPIRY_HOURS: int = 24

settings = Settings()