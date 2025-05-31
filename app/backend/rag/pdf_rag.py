"""
PDF RAG Engine Module

This module extends the base RAG system to support PDF document search and retrieval.
It combines text embeddings with PDF chunks for comprehensive search.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from typing import List, Dict, Any, Optional
from app.backend.rag.rag_engine import RAGEngine
from app.backend.rag.pdf_processor import PDFProcessor
from app.backend.rag.vector_db import VectorSearchDB
from app.backend.rag.embedding_service import embedding_service
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("pdf_rag")

class PDFRagEngine(RAGEngine):
    """RAG engine that supports PDF document search and retrieval"""
    
    def __init__(self):
        # Initialize parent with our PDF-specific vector DB configuration
        self.pdf_processor = PDFProcessor()
        super().__init__(vector_db=VectorSearchDB(
            index_id="pdf_content_index",
            dimensions=768,  # Using text-embedding-gecko dimensions
            description="Vector search index for PDF content"
        ))
        
    def process_and_index_pdf(self, pdf_path: str) -> bool:
        """Process a PDF file and index its chunks for vector search"""
        try:
            # Process PDF into chunks
            chunks = self.pdf_processor.process_pdf(pdf_path)
            if not chunks:
                logger.error(f"Failed to process PDF {pdf_path}")
                return False
            
            # Generate embeddings and store in vector DB
            documents = []
            for chunk in chunks:
                # Generate embedding for the chunk
                embedding = embedding_service.generate_embedding(chunk["text"])
                if not embedding:
                    logger.warning(f"Failed to generate embedding for chunk from {pdf_path}")
                    continue
                
                # Create unique ID for this chunk
                chunk_id = f"{pdf_path}_{len(documents)}"
                
                # Store chunk data in parent class's map
                self.chunk_data_map[chunk_id] = {
                    "text": chunk["text"],
                    "metadata": {
                        "source": pdf_path,
                        "pages": chunk["metadata"]["pages"],
                        "sections": chunk["metadata"]["sections"]
                    }
                }
                
                # Create document for vector DB
                doc = {
                    "id": chunk_id,
                    "text": chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        "source": pdf_path,
                        "pages": chunk["metadata"]["pages"],
                        "sections": chunk["metadata"]["sections"]
                    }
                }
                documents.append(doc)
            
            if not documents:
                logger.error(f"No valid documents generated for PDF {pdf_path}")
                return False
            
            # Store documents with embeddings
            successful_ids = self.vector_db.upsert_documents(documents)
            if successful_ids:
                logger.info(f"Successfully indexed {len(successful_ids)} chunks from PDF {pdf_path}")
                return True
            else:
                logger.error(f"Failed to index chunks from PDF {pdf_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing and indexing PDF {pdf_path}: {e}")
            return False
    
    def search_pdf_content(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search PDF content using vector similarity"""
        try:
            # Search using vector DB
            results = self.vector_db.search(
                query_text=query,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Get the chunk ID from the result
                chunk_id = result.get('id')
                
                # Get the original chunk data from our map
                chunk_data = self.chunk_data_map.get(chunk_id, {})
                chunk_text = chunk_data.get('text', '')
                chunk_metadata = chunk_data.get('metadata', {})
                
                formatted_results.append({
                    "text": chunk_text or result.get("text", ""),
                    "metadata": chunk_metadata or result.get("metadata", {}),
                    "score": result.get("score", 0.0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching PDF content: {e}")
            return []

def test_pdf_rag():
    """Test the PDFRagEngine functionality"""
    engine = PDFRagEngine()
    
    # Process and index PDF
    pdf_path = "documents/Must-Have-guide-to-Florence-www.goinspired.com_.pdf"
    print(f"\nProcessing and indexing PDF: {pdf_path}")
    success = engine.process_and_index_pdf(pdf_path)
    
    if success:
        print("\n✅ Successfully processed and indexed PDF")
        
        # Test search
        query = "What are some popular restaurants in Florence?"
        print(f"\nSearching for: {query}")
        
        results = engine.search_pdf_content(query)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Text: {result['text'][:200]}...")
            metadata = result['metadata']
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Pages: {metadata.get('pages', 'N/A')}")
            print(f"Score: {result['score']:.3f}")
    else:
        print("\n❌ Failed to process and index PDF")

if __name__ == "__main__":
    test_pdf_rag() 