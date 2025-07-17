from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from database import Base
from schemas import ChunkingMethod, EmbeddingModel, BookingStatus
import enum

class FileModel(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_text = Column(Text, nullable=False)
    chunking_method = Column(SQLEnum(ChunkingMethod), nullable=False)
    embedding_model = Column(SQLEnum(EmbeddingModel), nullable=False)
    chunk_count = Column(Integer, nullable=False)
    vector_ids = Column(JSON, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

class InterviewBooking(Base):
    __tablename__ = "interview_bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    interview_date = Column(String, nullable=False)
    interview_time = Column(String, nullable=False)
    notes = Column(Text)
    status = Column(SQLEnum(BookingStatus), default=BookingStatus.CONFIRMED)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
