"""
Gemini RAG Model Integration

This module provides RAG-enhanced chat functionality using the RAGEngine.
"""
import os
from typing import Dict, Any

from app.backend.rag.rag_engine import RAGEngine
from app.utils.logging_utils import ChatbotLogger
# Assuming your sample data path is relatively known or configurable
# For now, hardcoding a default relative path for the sample data
# This should ideally come from a config or be passed in.
# Path is relative to the L200-upskilling directory if script is run from there,
# or needs to be constructed carefully based on execution context.
# Let's assume an environment variable or a fixed path for simplicity in this module.
DEFAULT_NDJSON_DATA_PATH = os.environ.get(
    "RAG_DATA_FILE_PATH", 
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "L200-upskilling", "data", "travel", "sample_travel.ndjson")
)
# More robust path assuming this file is in L200-upskilling/app/backend/models/
module_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NDJSON_DATA_PATH_FROM_MODULE = os.path.normpath(os.path.join(module_dir, "..", "..", "..", "data", "travel", "sample_travel.ndjson"))


logger = ChatbotLogger("gemini_rag_model")

# Global instance of RAGEngine
# This instance will hold the in-memory text map after data is processed.
rag_engine_instance = RAGEngine()
_rag_data_loaded = False # Flag to track if data has been loaded

def _ensure_rag_data_loaded(data_file_path: str = None) -> bool:
    """Ensures the RAG data is processed and loaded into the engine."""
    global _rag_data_loaded
    if _rag_data_loaded:
        return True

    actual_data_file = data_file_path or DEFAULT_NDJSON_DATA_PATH_FROM_MODULE
    
    if not os.path.exists(actual_data_file):
        logger.error(f"RAG data file not found: {actual_data_file}. Cannot load RAG data.")
        # Attempt to find it relative to a common project structure if the default is wrong
        # This is a fallback, ideally path is correct via env var or direct pass.
        alt_path = os.path.join("L200-upskilling", "data", "travel", "sample_travel.ndjson")
        if os.path.exists(alt_path):
            logger.info(f"Trying alternative path: {alt_path}")
            actual_data_file = alt_path
        else:
            logger.error(f"Alternative path also not found: {alt_path}. RAG will not function correctly.")
            return False
            
    logger.info(f"Loading RAG data from: {actual_data_file}...")
    try:
        upserted_ids = rag_engine_instance.process_and_upsert_dataset(actual_data_file)
        if upserted_ids:
            logger.info(f"RAG data loaded and processed. {len(upserted_ids)} items upserted/indexed.")
            _rag_data_loaded = True
            return True
        else:
            logger.warning("RAG data processing/upsert returned no IDs. Check logs.")
            # _rag_data_loaded might remain False or be set to True with a warning
            # For now, let's assume if it ran without error but no IDs, it's still "attempted"
            _rag_data_loaded = True # Or False, depending on how strict we want to be
            return False # Indicate that loading might not have been fully successful
    except Exception as e:
        logger.error(f"Exception during RAG data loading: {e}", exc_info=True)
        return False

def chat_with_gemini_rag(user_prompt: str, top_k: int = 3, data_file_path: str = None) -> str:
    """
    Chat with Gemini using RAG enhancement.
    Ensures data is loaded, then queries the RAGEngine.
    """
    if not _ensure_rag_data_loaded(data_file_path):
        logger.error("RAG data not loaded. Falling back to a non-RAG response or error message.")
        # Fallback: Could call a non-RAG Gemini here, or return a specific error.
        # For now, let's indicate the RAG part failed.
        return "I am having trouble accessing the travel knowledge base right now. Please try again later."

    logger.info(f"RAG Chat: Received prompt: '{user_prompt[:100]}...'")
    rag_response_dict = rag_engine_instance.query(user_prompt, top_k=top_k)
    return rag_response_dict.get("response", "Sorry, I encountered an issue retrieving an answer.")

# Placeholder for other RAG-related control functions if needed in the future
# e.g., to explicitly reload data, get status, etc.

if __name__ == '__main__':
    # Simple test for this module
    print("Testing gemini_rag.py...")
    # Ensure data is loaded (it will try to load on first call to chat_with_gemini_rag)
    # You can set the DEFAULT_NDJSON_DATA_PATH via environment variable RAG_DATA_FILE_PATH if needed
    
    print(f"Data path check: {DEFAULT_NDJSON_DATA_PATH_FROM_MODULE}")
    if not os.path.exists(DEFAULT_NDJSON_DATA_PATH_FROM_MODULE):
        print(f"WARNING: Default data file not found at {DEFAULT_NDJSON_DATA_PATH_FROM_MODULE}. Test may not retrieve context.")

    test_queries = [
        "Are there any art-themed hotels in Recoleta?",
        "Tell me about hotels near San Telmo market with antique furnishings.",
        "What are the best surf spots in Buenos Aires?"
    ]

    for query in test_queries:
        print(f"\nUser Query: {query}")
        response = chat_with_gemini_rag(query)
        print(f"RAG Model Response: {response}")
        print("---") 