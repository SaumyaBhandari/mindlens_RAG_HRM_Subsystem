from typing import List, Dict, Any
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from schemas import ChunkingMethod, EmbeddingModel, SimilarityAlgorithm
import os
from services.embedding_service import EmbeddingService

class VectorService:
    def __init__(self):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.collection_name = "document_embeddings"
    
    async def initialize(self):
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
    
    async def store_embeddings(
        self,
        embeddings: List[List[float]],
        chunks: List[str],
        filename: str,
        chunking_method: ChunkingMethod,
        embedding_model: EmbeddingModel
    ) -> List[str]:
        points = []
        vector_ids = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            point_id = str(uuid.uuid4())
            vector_ids.append(point_id)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                    "chunk_index": i,
                    "chunking_method": chunking_method,
                    "embedding_model": embedding_model
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return vector_ids
    
    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    ) -> List[Dict[str, Any]]:

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "text": result.payload["text"],
                "filename": result.payload["filename"],
                "chunk_index": result.payload["chunk_index"]
            }
            for result in results
        ]