import pytest
import asyncio
from services.chunking_service import ChunkingService
from schemas import ChunkingMethod

class TestChunkingService:
    def setup_method(self):
        self.service = ChunkingService()
        self.sample_text = """
        This is the first paragraph with some content.
        It contains multiple sentences to test chunking.
        
        This is the second paragraph with different content.
        It also contains multiple sentences for testing.
        
        This is the third paragraph with more content.
        It helps test the chunking algorithms properly.
        """
    
    @pytest.mark.asyncio
    async def test_recursive_chunking(self):
        chunks = await self.service.chunk_text(self.sample_text, ChunkingMethod.RECURSIVE)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_chunking(self):
        chunks = await self.service.chunk_text(self.sample_text, ChunkingMethod.SEMANTIC)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_custom_chunking(self):
        chunks = await self.service.chunk_text(self.sample_text, ChunkingMethod.CUSTOM)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)