import pytest
import asyncio
from services.embedding_service import EmbeddingService
from schemas import EmbeddingModel

class TestEmbeddingService:
    def setup_method(self):
        self.service = EmbeddingService()
        self.sample_chunks = [
            "This is the first chunk of text.",
            "This is the second chunk of text.",
            "This is the third chunk of text."
        ]
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_embeddings(self):
        embeddings = await self.service.generate_embeddings(
            self.sample_chunks, EmbeddingModel.SENTENCE_TRANSFORMER
        )
        assert len(embeddings) == len(self.sample_chunks)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 384 for emb in embeddings)  # SentenceTransformer dimension
