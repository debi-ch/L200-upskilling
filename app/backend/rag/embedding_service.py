"""
Embedding Service

This module provides a service for generating text embeddings
using Google's text-embedding-gecko model via Vertex AI.
"""

from typing import List, Optional
import vertexai
from vertexai.language_models import TextEmbeddingModel

from app.backend.rag.config import EmbeddingModelConfig
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("rag_embedding_service")

class EmbeddingService:
    """Service for generating embeddings using Vertex AI."""
    
    def __init__(self):
        self.model_name = EmbeddingModelConfig.MODEL_NAME
        self.project_id = EmbeddingModelConfig.PROJECT_ID
        self.location = EmbeddingModelConfig.LOCATION
        self.api_endpoint = EmbeddingModelConfig.API_ENDPOINT
        self.timeout = EmbeddingModelConfig.TIMEOUT_SECONDS
        self.model: Optional[TextEmbeddingModel] = None
        self.initialized = False
        
        logger.info(
            "EmbeddingService configured",
            model_name=self.model_name,
            project_id=self.project_id,
            location=self.location
        )

    def _initialize_vertex_ai(self):
        """Initializes Vertex AI if not already done."""
        if not self.initialized:
            try:
                logger.info(f"Initializing Vertex AI for Embedding Service: project={self.project_id}, location={self.location}")
                vertexai.init(project=self.project_id, location=self.location)
                # client_options = {"api_endpoint": self.api_endpoint}
                # self.model = TextEmbeddingModel.from_pretrained(self.model_name, client_options=client_options)
                self.model = TextEmbeddingModel.from_pretrained(self.model_name)
                self.initialized = True
                logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Vertex AI or load embedding model '{self.model_name}'",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                # Potentially re-raise or handle to prevent use of uninitialized service
                raise

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for a single piece of text."""
        self._initialize_vertex_ai()
        if not self.model:
            logger.error("Embedding model not available for generating single embedding.")
            return None
        try:
            embeddings = self.model.get_embeddings([text], auto_truncate=True) # auto_truncate can be useful
            return embeddings[0].values
        except Exception as e:
            logger.error(
                f"Failed to generate embedding for text: '{text[:100]}...'",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generates embeddings for a list of texts."""
        self._initialize_vertex_ai()
        if not self.model:
            logger.error("Embedding model not available for generating multiple embeddings.")
            return [None] * len(texts)
        
        results: List[Optional[List[float]]] = []
        try:
            # The get_embeddings method handles batching internally if the list is too large for one call.
            # However, there are still limits, so very large lists might need manual batching.
            # For simplicity here, we assume the list size is manageable.
            embeddings_response = self.model.get_embeddings(texts, auto_truncate=True)
            results = [embedding.values for embedding in embeddings_response]
            return results
        except Exception as e:
            logger.error(
                f"Failed to generate embeddings for {len(texts)} texts.",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            # Return a list of Nones matching the input length on error
            return [None] * len(texts)

# Singleton instance
embedding_service = EmbeddingService() 