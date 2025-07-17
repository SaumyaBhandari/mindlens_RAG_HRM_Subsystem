from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from enum import Enum

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    CUSTOM = "custom"

class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI = "openai"
    GEMINI = "gemini"

class SimilarityAlgorithm(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class LLMModel(str, Enum):
    OPENAI_GPT_3_5_TURBO = "gpt-3.5-turbo"
    GEMINI_FLASH_SMALL = "gemini-1.5-flash"
    GEMINI_FLASH_LARGE = "gemini-2.5-flash"

class FileUploadResponse(BaseModel):
    file_id: int
    filename: str
    chunk_count: int
    chunking_method: ChunkingMethod
    embedding_model: EmbeddingModel
    message: str

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_memory: bool = True
    similarity_algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: Optional[str]
    similarity_algorithm: SimilarityAlgorithm

class InterviewBookingRequest(BaseModel):
    full_name: str
    email: EmailStr
    interview_date: str
    interview_time: str
    notes: Optional[str] = None

class InterviewBookingResponse(BaseModel):
    booking_id: int
    message: str
    full_name: str
    email: str
    interview_date: str
    interview_time: str
