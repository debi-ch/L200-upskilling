"""
RAG Engine

Orchestrates document processing, vector DB interaction, and query augmentation.
"""
from typing import List, Dict, Any, Optional

# Ensure RAGPromptTemplates is defined in config.py if you uncomment its usage here
from app.backend.rag.config import VectorDBConfig, RAGPromptTemplates #, RAGPromptTemplates 
from app.backend.rag.vector_db import VectorSearchDB 
from app.backend.rag.document_processor import DocumentProcessor
# from app.backend.models.gemini_chat_refactored import chat_with_model_internal # Moved to query method
from app.config import GCP_PROJECT_ID, GCP_LOCATION, ModelConfig # For LLM call
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("rag_engine")

class RAGEngine:
    def __init__(self, vector_db: Optional[VectorSearchDB] = None):
        self.vector_db = vector_db or VectorSearchDB() 
        self.document_processor = DocumentProcessor()
        # This map will store chunk_id -> {'text': chunk_text, 'metadata': original_doc_metadata}
        self.chunk_data_map: Dict[str, Dict[str, Any]] = {} 
        logger.info("RAGEngine initialized.")

    def process_and_upsert_dataset(self, ndjson_file_path: str) -> List[str]:
        """Processes an NDJSON dataset, stores text map, and upserts embeddings to VectorSearchDB."""
        logger.info(f"RAGEngine: Starting dataset processing for {ndjson_file_path}")
        
        if not self.vector_db.initialize(): 
            logger.error("RAGEngine: VectorDB failed to initialize. Halting dataset processing.")
            return []

        chunks_for_upsert, id_to_data_map = self.document_processor.process_ndjson_file(ndjson_file_path)
        
        self.chunk_data_map.update(id_to_data_map)
        logger.info(f"RAGEngine: Stored/updated chunk data map with {len(id_to_data_map)} new entries. Total map size: {len(self.chunk_data_map)}.")

        if not chunks_for_upsert:
            logger.warning(f"RAGEngine: No chunks processed from {ndjson_file_path} suitable for upsert.")
            return []
        
        logger.info(f"RAGEngine: Upserting {len(chunks_for_upsert)} processed chunks to VectorDB.")
        upserted_ids = self.vector_db.upsert_documents(chunks_for_upsert)
        if upserted_ids:
            logger.info(f"RAGEngine: Completed upserting. {len(upserted_ids)} chunks confirmed upserted.")
        else:
            logger.warning("RAGEngine: Upsert operation returned no IDs. Check VectorDB logs.")
        return upserted_ids

    def query(self, user_query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        # Import here to break circular dependency
        from app.backend.models.gemini_chat_refactored import chat_with_model_internal

        logger.info(f"RAGEngine: Received query: '{user_query[:100]}...'")
        
        if not self.vector_db.initialize(): 
            logger.error("RAGEngine: VectorDB failed to initialize. Cannot perform query.")
            return {"response": "Error: Could not connect to knowledge base.", "search_results": [], "context_for_llm": ""}

        search_results_from_db = self.vector_db.search(user_query, top_k=top_k)
        
        augmented_search_results = []
        retrieved_context_entries = [] # List of formatted strings for LLM context

        for res_item in search_results_from_db: # res_item from DB has 'id', 'score', 'metadata' (empty from DB)
            chunk_id = res_item.get('id')
            chunk_data = self.chunk_data_map.get(chunk_id)

            if chunk_data:
                original_chunk_text = chunk_data.get("text", "[Text not found in map]")
                original_doc_metadata = chunk_data.get("metadata", {})
                hotel_name = original_doc_metadata.get('hotel_name', '')
                
                context_entry_prefix = f"Source chunk: {chunk_id}"
                if hotel_name:
                    context_entry_prefix = f"From hotel '{hotel_name}' (chunk: {chunk_id})"
                
                context_entry = f"{context_entry_prefix}:\n{original_chunk_text}"
                retrieved_context_entries.append(context_entry)
                
                # For returning to the user/UI, include all info
                res_item['text'] = original_chunk_text
                res_item['retrieved_doc_metadata'] = original_doc_metadata 
            else:
                logger.warning(f"Data for chunk ID {chunk_id} not found in RAGEngine.chunk_data_map.")
                res_item['text'] = "[Text data missing for this chunk ID]"
                res_item['retrieved_doc_metadata'] = {}
            
            augmented_search_results.append(res_item)

        context_str = "\n---\n".join(retrieved_context_entries)
        
        response_text = ""
        llm_call_succeeded = False
        final_augmented_prompt = user_query # Default if no context

        if not augmented_search_results or not context_str.strip():
            logger.info("No relevant context found or context is empty. Calling LLM with original query only.")
            try:
                response_text = chat_with_model_internal(
                    model_id=ModelConfig.GEMINI_BASE_MODEL_ID, 
                    project_id=GCP_PROJECT_ID, 
                    location=GCP_LOCATION, 
                    prompt_text=user_query, 
                    model_type_str="base"
                )
                llm_call_succeeded = True
                if not response_text: response_text = "I can try to answer generally, but I couldn't find specific information in the travel documents."
            except Exception as e:
                logger.error(f"LLM call (no context) failed: {e}", exc_info=True)
                response_text = "Error communicating with the language model."
        else:
            final_augmented_prompt = RAGPromptTemplates.CONTEXT_PROMPT_TEMPLATE.format(context=context_str, query=user_query)
            logger.info(f"Augmented prompt for LLM (first 300 chars): {final_augmented_prompt[:300]}...")
            try:
                response_text = chat_with_model_internal(
                    model_id=ModelConfig.GEMINI_BASE_MODEL_ID, 
                    project_id=GCP_PROJECT_ID, 
                    location=GCP_LOCATION, 
                    prompt_text=final_augmented_prompt,
                    model_type_str="base"
                )
                llm_call_succeeded = True
                if not response_text: response_text = "I received a response, but it was empty. Based on the documents, I found some information but couldn't synthesize a specific answer."
            except Exception as e:
                logger.error(f"LLM call (with context) failed: {e}", exc_info=True)
                response_text = "Error communicating with the language model after retrieving context."

        logger.info(f"RAGEngine: Query processing complete. LLM Call Succeeded: {llm_call_succeeded}. Found {len(augmented_search_results)} relevant chunks.")
        return {
            "response": response_text, 
            "search_results": augmented_search_results, 
            "context_for_llm": context_str,
            "augmented_prompt_for_llm": final_augmented_prompt
        }

# Optional: Create a singleton instance if desired for the application
# rag_engine = RAGEngine() 