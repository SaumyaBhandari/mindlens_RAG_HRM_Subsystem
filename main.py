from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
from typing import List, Optional
import asyncio
import os
from dotenv import load_dotenv

from database import get_db, init_db
from models import FileModel, InterviewBooking
from schemas import (
    FileUploadResponse, ChunkingMethod, EmbeddingModel,
    QueryRequest, QueryResponse, InterviewBookingRequest, InterviewBookingResponse,
    LLMModel
)
from services.file_service import FileService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from services.rag_service import RAGService
from services.email_service import EmailService
from utils.logger import get_logger

load_dotenv()

app = FastAPI(title="RAG Backend System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)

# Initialize services
file_service = FileService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_service = VectorService()
rag_service = RAGService()
email_service = EmailService()

@app.on_event("startup")
async def startup_event():
    await init_db()
    await vector_service.initialize()
    logger.info("Application started successfully")

@app.post("/api/v1/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE,
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMER,
    db: Session = Depends(get_db)
):
    try:
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed")
        content = await file.read()
        text = await file_service.extract_text(content, file.filename)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")

        chunks = await chunking_service.chunk_text(text, chunking_method)
        
        embeddings = await embedding_service.generate_embeddings(chunks, embedding_model)
        
        vector_ids = await vector_service.store_embeddings(
            embeddings, chunks, file.filename, chunking_method, embedding_model
        )
        
        file_record = FileModel(
            filename=file.filename,
            original_text=text,
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            chunk_count=len(chunks),
            vector_ids=vector_ids
        )
        
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        
        logger.info(f"File {file.filename} processed successfully with {len(chunks)} chunks")
        
        return FileUploadResponse(
            file_id=file_record.id,
            filename=file.filename,
            chunk_count=len(chunks),
            chunking_method=chunking_method,
            embedding_model=embedding_model,
            message="File uploaded and processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise Exception(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    llm_model: LLMModel = LLMModel.GEMINI_FLASH_LARGE,
):
    try:
        response = await rag_service.process_query(
            query=request.query,
            session_id=request.session_id,
            use_memory=request.use_memory,
            similarity_algorithm=request.similarity_algorithm
        )
        
        logger.info(f"Query processed successfully for session {request.session_id}")
        
        return QueryResponse(
            answer=response["answer"],
            sources=response.get("sources", []),
            session_id=request.session_id,
            similarity_algorithm=request.similarity_algorithm
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/book-interview", response_model=InterviewBookingResponse)
async def book_interview(
    request: InterviewBookingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        booking = InterviewBooking(
            full_name=request.full_name,
            email=request.email,
            interview_date=request.interview_date,
            interview_time=request.interview_time,
            notes=request.notes
        )
        
        db.add(booking)
        db.commit()
        db.refresh(booking)
        
        background_tasks.add_task(
            email_service.send_interview_notifications,
            booking.full_name,
            booking.email,
            booking.interview_date,
            booking.interview_time
        )
        
        logger.info(f"Interview booked successfully for {booking.full_name}")
        
        return InterviewBookingResponse(
            booking_id=booking.id,
            message="Interview booked successfully. Confirmation email will be sent.",
            full_name=booking.full_name,
            email=booking.email,
            interview_date=booking.interview_date,
            interview_time=booking.interview_time
        )
        
    except Exception as e:
        logger.error(f"Error booking interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error booking interview: {str(e)}")

@app.get("/api/v1/files")
async def list_files(db: Session = Depends(get_db)):
    files = db.query(FileModel).all()
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "chunking_method": f.chunking_method,
            "embedding_model": f.embedding_model,
            "chunk_count": f.chunk_count,
            "uploaded_at": f.uploaded_at
        }
        for f in files
    ]

@app.get("/api/v1/bookings")
async def list_bookings(db: Session = Depends(get_db)):
    bookings = db.query(InterviewBooking).all()
    return [
        {
            "id": b.id,
            "full_name": b.full_name,
            "email": b.email,
            "interview_date": b.interview_date,
            "interview_time": b.interview_time,
            "status": b.status,
            "created_at": b.created_at
        }
        for b in bookings
    ]

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "message": "RAG Backend System is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)