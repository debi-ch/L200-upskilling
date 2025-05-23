"""
Temporary test script for a simple RAG query.
"""
import os
import sys
import json # For pretty printing results

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.backend.rag.rag_engine import RAGEngine
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("test_rag_query")

def main():
    logger.info("Starting RAG query test...")

    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth application-default login has been run.
    # Ensure app.config has correct GCP_PROJECT_ID and GCP_LOCATION.
    
    # --- Data Ingestion Step (to populate the text map for this engine instance) ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    ndjson_file_path = os.path.join(current_script_dir, "data", "travel", "sample_travel.ndjson")

    if not os.path.exists(ndjson_file_path):
        logger.error(f"Data file not found: {ndjson_file_path}. Cannot run test.")
        return

    logger.info(f"Using data file for ingestion: {ndjson_file_path}")
    engine = RAGEngine() # Instantiate the RAG engine

    logger.info("Performing data ingestion first to populate text map and ensure index has data...")
    try:
        upserted_ids = engine.process_and_upsert_dataset(ndjson_file_path)
        if upserted_ids:
            logger.info(f"Data ingestion step completed. {len(upserted_ids)} items processed/upserted.")
        else:
            logger.warning("Data ingestion step completed, but no items were reported as upserted. Query test might fail or yield no text.")
    except Exception as e:
        logger.error("FAILURE during data ingestion step in query test script.", exc_info=True)
        return # Stop if ingestion fails
    # --- End Data Ingestion Step ---

    # --- Test Case 1: A query likely to have matches --- 
    user_query_1 = "Are there any art-themed hotels in Recoleta?"
    logger.info(f"\n--- Test Case 1: Querying for: '{user_query_1}' ---")
    
    try:
        rag_result_1 = engine.query(user_query_1, top_k=3) 
        
        print("\n--- RAG Engine Output (Test Case 1) ---")
        print(f"User Query: {user_query_1}")
        print(f"Engine Response: {rag_result_1.get('response')}")
        print("Search Results (Retrieved Chunks - IDs, Scores, and Text):")
        if rag_result_1.get('search_results'):
            for i, res in enumerate(rag_result_1['search_results']):
                print(f"  {i+1}. ID: {res.get('id')}, Score (Distance): {res.get('score'):.4f}")
                print(f"     Retrieved Text: {res.get('text', '')[:200]}...") 
        else:
            print("  No search results found.")
        # print(f"Context Prepared for LLM (first 500 chars):\n{rag_result_1.get('context_for_llm', '')[:500]}...")
        print("-----------------------------------------")

    except Exception as e:
        logger.error(f"FAILURE: An error occurred during RAG query test case 1: {user_query_1}", exc_info=True)

    # --- Test Case 2: A query less likely to have direct matches (to see fallback) --- 
    user_query_2 = "What are the best surf spots in Buenos Aires?"
    logger.info(f"\n--- Test Case 2: Querying for: '{user_query_2}' ---")
    
    try:
        rag_result_2 = engine.query(user_query_2, top_k=3)
        
        print("\n--- RAG Engine Output (Test Case 2) ---")
        print(f"User Query: {user_query_2}")
        print(f"Engine Response: {rag_result_2.get('response')}")
        print("Search Results (Retrieved Chunks - IDs, Scores, and Text):")
        if rag_result_2.get('search_results'):
            for i, res in enumerate(rag_result_2['search_results']):
                print(f"  {i+1}. ID: {res.get('id')}, Score (Distance): {res.get('score'):.4f}")
                print(f"     Retrieved Text: {res.get('text', '')[:200]}...")
        else:
            print("  No search results found.")
        # print(f"Context Prepared for LLM (first 500 chars):\n{rag_result_2.get('context_for_llm', '')[:500]}...")
        print("-----------------------------------------")

    except Exception as e:
        logger.error(f"FAILURE: An error occurred during RAG query test case 2: {user_query_2}", exc_info=True)

if __name__ == "__main__":
    main() 