"""
Test script for multimodal search capabilities.
This script demonstrates searching with text, images, and combinations of both.
"""

import os
from google.cloud import storage
from app.backend.rag.multimodal_rag import MultimodalRAG
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("multimodal_test")

def download_test_image(bucket_name: str, image_path: str) -> bytes:
    """Download a test image from GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(image_path)
    return blob.download_as_bytes()

def test_multimodal_search():
    """Test various search capabilities"""
    
    # Initialize RAG
    rag = MultimodalRAG()
    
    print("\n1. Loading text data...")
    ndjson_file_path = os.path.join("data", "travel", "sample_travel.ndjson")
    upserted_ids = rag.process_and_upsert_dataset(ndjson_file_path)
    if not upserted_ids:
        print("❌ Failed to load text data")
        return
    print(f"✅ Successfully loaded {len(upserted_ids)} text chunks")
    
    print("\n2. Loading image data...")
    success = rag.load_images()
    if not success:
        print("❌ Failed to load images")
        return
    print("✅ Successfully loaded images")
    
    # Test 1: Pure text query
    print("\n3. Testing text-only query...")
    text_query = "Tell me about wine tours in Florence"
    results = rag.query(text_query)
    print(f"\nQuery: {text_query}")
    print(f"Response: {results.get('response', 'No response')}")
    
    # Test 2: Image similarity search
    print("\n4. Testing image similarity search...")
    # Use wine tour image as query
    wine_image = download_test_image(
        "genai-l200-training",
        "images/Chianti-Wine-tour-The-Tour-Guy-700.jpg"
    )
    results = rag.query_with_image(wine_image)
    print("\nSimilar images found:")
    for i, result in enumerate(results.get('results', []), 1):
        print(f"{i}. {result['description']} (Score: {result['score']:.3f})")
    
    # Test 3: Combined image and text query
    print("\n5. Testing combined image and text query...")
    combined_query = "Find tours similar to this wine tour"
    results = rag.query(combined_query, image_bytes=wine_image)
    print(f"\nQuery: {combined_query}")
    print(f"Response: {results.get('response', 'No response')}")
    print("\nRelevant images:")
    for i, result in enumerate(results.get('results', []), 1):
        print(f"{i}. {result['description']} (Score: {result['score']:.3f})")

if __name__ == "__main__":
    test_multimodal_search() 