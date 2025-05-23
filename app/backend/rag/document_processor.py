"""
Document Processor for RAG

Handles reading NDJSON, chunking, metadata extraction, and embedding generation.
"""
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from app.backend.rag.config import DocumentProcessingConfig
from app.backend.rag.embedding_service import embedding_service
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("rag_document_processor")

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = DocumentProcessingConfig.CHUNK_SIZE
        self.chunk_overlap = DocumentProcessingConfig.CHUNK_OVERLAP
        self.metadata_fields = DocumentProcessingConfig.METADATA_FIELDS
        logger.info(
            "DocumentProcessor initialized", 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            metadata_fields=self.metadata_fields
        )

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunk = text[start_index:end_index]
            chunks.append(chunk)
            next_start = start_index + self.chunk_size - self.chunk_overlap
            if next_start <= start_index and len(text) > self.chunk_size:
                next_start = start_index + 1 
            start_index = next_start
            if start_index >= len(text):
                break
        return chunks

    def process_single_document(self, document: Dict[str, Any], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Processes a single document. 
           Returns: (list_of_chunks_for_upsert, chunk_id_to_data_map)
           chunk_id_to_data_map maps chunk_id to {'text': chunk_text, 'metadata': original_doc_metadata}
        """
        content_to_embed = document.get('hotel_description', '') 
        if not content_to_embed and 'text' in document: 
            content_to_embed = document['text']
        if not content_to_embed:
            logger.warning(f"Document {source_id} has no 'hotel_description' or 'text' field. Skipping.")
            return [], {}

        text_chunks = self._chunk_text(content_to_embed)
        processed_chunks_for_upsert = []
        chunk_id_to_data_map: Dict[str, Dict[str, Any]] = {}
        
        # Extract base metadata from the original document once
        original_doc_metadata = {field: document.get(field) for field in self.metadata_fields if document.get(field) is not None}
        original_doc_metadata['source_id'] = source_id # Keep original source_id for reference
        # Add other fixed fields from the document if needed, e.g., document title if it exists
        # original_doc_metadata['original_doc_title'] = document.get('hotel_name', '') 

        if not text_chunks:
            return [], {}
            
        embeddings = embedding_service.generate_embeddings([chunk for chunk in text_chunks])

        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"{source_id}_chunk_{i}"
            
            # Store text and original document metadata for this chunk_id
            chunk_id_to_data_map[chunk_id] = {
                "text": text_chunk,
                "metadata": original_doc_metadata # This is the metadata of the parent document
            }

            if embeddings and i < len(embeddings) and embeddings[i] is not None:
                # For upserting, we only need id and embedding. Metadata for restricts is handled by VectorSearchDB.
                # The metadata passed here to VectorSearchDB for its `restricts` field should be *filterable* metadata.
                # If we are not filtering yet, this can be minimal or just the source_id and chunk_id.
                upsert_metadata = {"source_id": source_id, "chunk_id_ref": chunk_id} # Example minimal metadata for upsert
                
                processed_chunks_for_upsert.append({
                    "id": chunk_id,
                    "embedding": embeddings[i],
                    "metadata": upsert_metadata # Pass minimal, filterable metadata if any, or just identifiers
                })
            else:
                logger.warning(f"Could not generate embedding for chunk {i} of document {source_id}. Skipping chunk from upsert list.")
                if chunk_id in chunk_id_to_data_map: 
                    del chunk_id_to_data_map[chunk_id]
        
        return processed_chunks_for_upsert, chunk_id_to_data_map

    def process_ndjson_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Reads an NDJSON file. 
           Returns: (all_chunks_for_upsert, combined_chunk_id_to_data_map)
        """
        all_chunks_for_upsert = []
        combined_chunk_id_to_data_map: Dict[str, Dict[str, Any]] = {}

        if not os.path.exists(file_path):
            logger.error(f"NDJSON file not found: {file_path}")
            return [], {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        document = json.loads(line.strip())
                        doc_identifier = document.get('hotel_name', f"line_{i+1}")
                        source_id = f"{os.path.basename(file_path)}_{doc_identifier}"
                        
                        chunks_for_doc, id_to_data_map_for_doc = self.process_single_document(document, source_id)
                        all_chunks_for_upsert.extend(chunks_for_doc)
                        combined_chunk_id_to_data_map.update(id_to_data_map_for_doc)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON on line {i+1} in {file_path}: {e}. Line: '{line.strip()[:200]}...'")
                    except Exception as e:
                        logger.error(f"Error processing document on line {i+1} in {file_path}: {e}", exc_info=True)
            logger.info(f"Successfully processed. Total chunks for upsert: {len(all_chunks_for_upsert)}. Total chunk data mappings: {len(combined_chunk_id_to_data_map)} from {file_path}.")
        except Exception as e:
            logger.error(f"Failed to read or process NDJSON file {file_path}: {e}", exc_info=True)
        
        return all_chunks_for_upsert, combined_chunk_id_to_data_map 