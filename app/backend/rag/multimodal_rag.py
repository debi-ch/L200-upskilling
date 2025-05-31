"""
Multimodal RAG Engine

This module extends the base RAG system to support both text and image queries.
It combines text embeddings with image embeddings for comprehensive search.
"""

import os
from typing import List, Dict, Any, Optional
from PIL import Image
import io

from app.backend.rag.rag_engine import RAGEngine
from app.backend.rag.image_processor import ImageProcessor
from app.backend.rag.config import RAGConfig
from app.backend.rag.vector_db import VectorSearchDB
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("multimodal_rag")

class MultimodalRAG(RAGEngine):
    """RAG engine that supports both text and image queries"""
    
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()
        self.image_vector_db = VectorSearchDB(
            index_id="travel_images_index",
            dimensions=1408,  # Multimodal embedding dimension from Vertex AI
            description="Image embeddings for travel images"
        )
        self.has_images = False
    
    def load_images(self) -> bool:
        """Load and process images from the bucket"""
        try:
            logger.info("Processing images from bucket...")
            processed_images = self.image_processor.process_images()
            
            if not processed_images:
                logger.warning("No images were processed")
                return False
            
            # Initialize image vector DB
            if not self.image_vector_db.initialize():
                logger.error("Failed to initialize image vector database")
                return False
            
            # Store image embeddings in vector database
            for img in processed_images:
                self.image_vector_db.upsert_documents([{
                    'id': f"img_{img['image_path']}",
                    'embedding': img['embedding'],
                    'metadata': {
                        'type': 'image',
                        'path': img['image_path'],
                        'description': img['metadata']['description'],
                        'dimensions': f"{img['metadata']['width']}x{img['metadata']['height']}"
                    }
                }])
            
            self.has_images = True
            logger.info(f"Successfully loaded {len(processed_images)} images")
            return True
            
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            return False
    
    def query_with_image(self, image_bytes: bytes, query_text: str = "", top_k: int = 3) -> Dict[str, Any]:
        """Query using both image and optional text"""
        try:
            # Generate embedding for the query image
            image_data = self.image_processor.generate_image_embedding(image_bytes)
            if not image_data:
                return {'error': 'Failed to process query image'}
            
            # Get similar images from image vector DB
            image_results = self.image_vector_db.search(
                query_embedding=image_data['embedding'],
                top_k=top_k
            )
            
            # Process results
            results = []
            context_parts = []
            
            for result in image_results:
                if result['metadata'].get('type') == 'image':
                    results.append({
                        'type': 'image',
                        'path': result['metadata']['path'],
                        'description': result['metadata']['description'],
                        'score': result['score']
                    })
                    context_parts.append(f"Image: {result['metadata']['description']}")
            
            # If text query provided, also search text content
            if query_text:
                text_results = super().query(query_text, top_k=top_k)
                if 'context' in text_results:
                    context_parts.append(text_results['context'])
            
            # Combine contexts
            combined_context = "\n\n".join(context_parts)
            
            return {
                'results': results,
                'context': combined_context
            }
            
        except Exception as e:
            logger.error(f"Error in image query: {e}")
            return {'error': str(e)}
    
    def query(self, query_text: str, image_bytes: Optional[bytes] = None, top_k: int = 3) -> Dict[str, Any]:
        """Enhanced query that handles both text and optional image input"""
        try:
            if image_bytes:
                return self.query_with_image(image_bytes, query_text, top_k)
            else:
                return super().query(query_text, top_k)
        except Exception as e:
            logger.error(f"Error in multimodal query: {e}")
            return {'error': str(e)}

def test_multimodal_rag():
    """Test the MultimodalRAG functionality"""
    
    # Initialize RAG
    rag = MultimodalRAG()
    
    # Load text data first
    print("Loading text data...")
    ndjson_file_path = os.path.join("data", "travel", "sample_travel.ndjson")
    if not os.path.exists(ndjson_file_path):
        print(f"Error: Text data file not found at {ndjson_file_path}")
        return
    
    upserted_ids = rag.process_and_upsert_dataset(ndjson_file_path)
    if not upserted_ids:
        print("Failed to load text data")
        return
    print(f"Successfully loaded {len(upserted_ids)} text chunks")
    
    # Load images
    print("\nLoading images...")
    success = rag.load_images()
    print(f"Image loading {'successful' if success else 'failed'}")
    
    # Test text query
    print("\nTesting text query...")
    result = rag.query("Show me hotels with a pool")
    print(f"Text query results: {result}")
    
    # Test image query
    print("\nTesting image query...")
    # You would need to provide an actual image for this test
    # image_bytes = ... # Load test image
    # result = rag.query("Find similar hotels", image_bytes=image_bytes)
    # print(f"Image query results: {result}")

if __name__ == "__main__":
    test_multimodal_rag() 