from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import EmbeddingModel
from google import genai
import openai
import os

class EmbeddingService:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    async def generate_embeddings(self, chunks: List[str], model: EmbeddingModel) -> List[List[float]]:
        if model == EmbeddingModel.SENTENCE_TRANSFORMER:
            return await self._generate_sentence_transformer_embeddings(chunks)
        elif model == EmbeddingModel.OPENAI:
            return await self._generate_openai_embeddings(chunks)
        elif model == EmbeddingModel.GEMINI:
            return await self._generate_gemini_embeddings(chunks)
        else:
            raise ValueError(f"Unsupported embedding model: {model}")
        
    async def _generate_gemini_embeddings(self, chunks: List[str]) -> List[List[float]]:
        print(f"Embedding {len(chunks)} chunks of sentences using Gemini...")
        all_embeddings: List[List[float]] = [] 
        try:
            response = self.gemini_client.models.embed_content(
                model="models/embedding-001", 
                contents=chunks,
            )

            if response and hasattr(response, 'embeddings'):
                for embedding_obj in response.embeddings:
                    if hasattr(embedding_obj, 'values') and isinstance(embedding_obj.values, list):
                        all_embeddings.append(embedding_obj.values)
                    else:
                        print(f"Warning: Gemini embedding object missing or invalid 'values' attribute: {embedding_obj}")
            else:
                print(f"Warning: Gemini API response did not contain 'embeddings' attribute or it's empty: {response}")

            print("Gemini embeddings generated successfully.")
            return all_embeddings
        except Exception as e:
            print(f"Error during Gemini embedding: {e}")
            import traceback
            traceback.print_exc()
            raise


    async def _generate_sentence_transformer_embeddings(self, chunks: List[str]) -> List[List[float]]:
        try:
            print(f"Embedding {len(chunks)} chunks of sentences...")
            embeddings = self.sentence_transformer.encode(chunks)
            print("Embeddings generated successfully.")
            return embeddings.tolist()
        except Exception as e:
            print(f"Error during SentenceTransformer encoding: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _generate_openai_embeddings(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for chunk in chunks:
            response = openai.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embeddings.append(response['data'][0]['embedding'])
        return embeddings
