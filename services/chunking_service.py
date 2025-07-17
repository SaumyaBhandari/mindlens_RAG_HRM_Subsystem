import re
from typing import List
from schemas import ChunkingMethod
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChunkingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def chunk_text(self, text: str, method: ChunkingMethod) -> List[str]:
        if method == ChunkingMethod.RECURSIVE:
            return await self._recursive_chunking(text)
        elif method == ChunkingMethod.SEMANTIC:
            return await self._semantic_chunking(text)
        elif method == ChunkingMethod.CUSTOM:
            return await self._custom_chunking(text)
        else:
            raise ValueError(f"Unsupported chunking method: {method}")
    
    async def _recursive_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    async def _semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            current_embedding = embeddings[i]
            prev_embedding = embeddings[i-1]
            
            similarity = cosine_similarity([current_embedding], [prev_embedding])[0][0]
            
            if similarity >= similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    async def _custom_chunking(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        max_chunk_size = 800
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
