"""
PDF Processing Module for RAG

This module handles PDF document processing, including:
- Loading PDFs from Google Cloud Storage
- Extracting text and structure
- Chunking content for embedding
- Managing PDF metadata
"""

import os
from typing import List, Dict, Any, Optional
from google.cloud import storage
import fitz  # PyMuPDF
import io
import re
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("pdf_processor")

class PDFProcessor:
    """Handles PDF processing and chunking for RAG"""
    
    def __init__(self, bucket_name: str = "genai-l200-training"):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
    def download_pdf(self, pdf_path: str) -> Optional[bytes]:
        """Download PDF from GCS bucket"""
        try:
            blob = self.bucket.blob(pdf_path)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Error downloading PDF {pdf_path}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text and structure from PDF"""
        try:
            # Open PDF from memory
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_content = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract text with formatting
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        # Process text blocks
                        text_content = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_content += span["text"] + " "
                        
                        if text_content.strip():
                            extracted_content.append({
                                "page_num": page_num + 1,
                                "text": text_content.strip(),
                                "metadata": {
                                    "font_size": block.get("lines", [{}])[0].get("spans", [{}])[0].get("size", 0),
                                    "is_bold": block.get("lines", [{}])[0].get("spans", [{}])[0].get("flags", 0) & 2 > 0,
                                    "bbox": block.get("bbox", [])
                                }
                            })
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return []
    
    def chunk_pdf_content(self, content: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """Chunk PDF content into smaller pieces for embedding"""
        chunks = []
        current_chunk = ""
        current_metadata = {
            "pages": set(),
            "sections": []
        }
        
        for block in content:
            text = block["text"]
            page_num = block["page_num"]
            
            # Add to current chunk
            if len(current_chunk) + len(text) <= chunk_size:
                current_chunk += text + " "
                current_metadata["pages"].add(page_num)
                current_metadata["sections"].append({
                    "page": page_num,
                    "metadata": block["metadata"]
                })
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            "pages": sorted(list(current_metadata["pages"])),
                            "sections": current_metadata["sections"]
                        }
                    })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else ""
                current_chunk = overlap_text + " " + text
                current_metadata = {
                    "pages": {page_num},
                    "sections": [{
                        "page": page_num,
                        "metadata": block["metadata"]
                    }]
                }
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    "pages": sorted(list(current_metadata["pages"])),
                    "sections": current_metadata["sections"]
                }
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunks ready for embedding"""
        try:
            # Download PDF
            pdf_bytes = self.download_pdf(pdf_path)
            if not pdf_bytes:
                return []
            
            # Extract text and structure
            content = self.extract_text_from_pdf(pdf_bytes)
            if not content:
                return []
            
            # Chunk content
            chunks = self.chunk_pdf_content(content)
            
            # Add source information
            for chunk in chunks:
                chunk["metadata"]["source"] = pdf_path
            
            logger.info(f"Successfully processed PDF {pdf_path} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []

def test_pdf_processor():
    """Test the PDFProcessor functionality"""
    processor = PDFProcessor()
    
    # Process PDF from bucket
    pdf_path = "docs/sample.pdf"  # Update with actual PDF path
    chunks = processor.process_pdf(pdf_path)
    
    if chunks:
        print(f"Successfully processed PDF into {len(chunks)} chunks")
        print("\nSample chunk:")
        print(f"Text: {chunks[0]['text'][:200]}...")
        print(f"Pages: {chunks[0]['metadata']['pages']}")
    else:
        print("Failed to process PDF")

if __name__ == "__main__":
    test_pdf_processor() 