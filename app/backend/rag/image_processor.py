"""
Image Processing Module for Multimodal RAG

This module handles image processing, including:
- Loading images from Google Cloud Storage
- Generating image embeddings
- Processing image metadata
"""

import os
from typing import List, Dict, Any, Optional
from google.cloud import storage
import vertexai
from vertexai.vision_models import Image as VertexImage, MultiModalEmbeddingModel, ImageTextModel
from PIL import Image
import io
import numpy as np

from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("image_processor")

class ImageProcessor:
    """Handles image processing and embedding generation for multimodal RAG"""
    
    def __init__(self, bucket_name: str = "genai-l200-training"):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize models
        self.embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        self.caption_model = ImageTextModel.from_pretrained("imagetext@001")
        
    def list_images(self) -> List[str]:
        """List all images in the bucket"""
        try:
            blobs = self.bucket.list_blobs(prefix="images/")
            return [blob.name for blob in blobs if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return []
    
    def download_image(self, image_path: str) -> Optional[bytes]:
        """Download image from GCS bucket"""
        try:
            blob = self.bucket.blob(image_path)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading image {image_path}: {e}")
            return None
    
    def generate_image_embedding(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Generate embedding and metadata for an image"""
        try:
            # Load image with PIL for metadata
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to Vertex AI Image
            vertex_image = VertexImage(image_bytes)
            
            # Get embedding
            embedding = self.embedding_model.get_embeddings(
                image=vertex_image,
                contextual_text="A travel destination or hotel image"
            )
            
            # Get image description
            description = self.caption_model.get_captions(
                image=vertex_image,
                number_of_results=1
            )[0]
            
            # The embedding is already a list
            return {
                'embedding': embedding.image_embedding,
                'metadata': {
                    'description': description,
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'format': pil_image.format
                }
            }
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None
    
    def process_images(self) -> List[Dict[str, Any]]:
        """Process all images in the bucket and return embeddings with metadata"""
        processed_images = []
        
        # List all images
        image_paths = self.list_images()
        logger.info(f"Found {len(image_paths)} images to process")
        
        for image_path in image_paths:
            try:
                # Download image
                image_bytes = self.download_image(image_path)
                if not image_bytes:
                    continue
                
                # Generate embedding and metadata
                result = self.generate_image_embedding(image_bytes)
                if result:
                    result['image_path'] = image_path
                    processed_images.append(result)
                    logger.info(f"Successfully processed {image_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue
        
        return processed_images

def test_image_processor():
    """Test the ImageProcessor functionality"""
    processor = ImageProcessor()
    
    # List images
    images = processor.list_images()
    print(f"Found {len(images)} images")
    
    # Process first image
    if images:
        image_bytes = processor.download_image(images[0])
        if image_bytes:
            result = processor.generate_image_embedding(image_bytes)
            if result:
                print(f"Generated embedding for {images[0]}")
                print(f"Description: {result['metadata']['description']}")
                print(f"Embedding dimension: {len(result['embedding'])}")
            else:
                print("Failed to generate embedding")

if __name__ == "__main__":
    test_image_processor() 