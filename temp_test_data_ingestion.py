"""
Temporary test script to process and upsert the sample travel dataset.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.backend.rag.rag_engine import RAGEngine # Assuming RAGEngine is not a singleton yet, or we make one.
from app.utils.logging_utils import ChatbotLogger

logger = ChatbotLogger("test_data_ingestion")

def main():
    logger.info("Starting data ingestion test...")

    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth application-default login has been run.
    # Ensure app.config has correct GCP_PROJECT_ID and GCP_LOCATION.

    # Path to the sample data
    # Assumes this script is in L200-upskilling directory
    base_dir = os.path.dirname(__file__) # Should be L200-upskilling
    # Correctly join paths to go up to .cursor-tutor and then down to L200-upskilling/data/travel
    # However, since this script itself is in L200-upskilling, we can use relative paths from there or construct from project root assumption.
    # For simplicity, if running from .cursor-tutor as CWD, the path is as below.
    # If this script is run from L200-upskilling, then data_file_path should be "data/travel/sample_travel.ndjson"
    
    # Let's assume the script is run from the parent of L200-upskilling (e.g. .cursor-tutor)
    # ndjson_file_path = os.path.join("L200-upskilling", "data", "travel", "sample_travel.ndjson")
    
    # Simpler: Assuming the script is located at L200-upskilling/temp_test_data_ingestion.py
    # and the data is at L200-upskilling/data/travel/sample_travel.ndjson
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    ndjson_file_path = os.path.join(current_script_dir, "data", "travel", "sample_travel.ndjson")

    if not os.path.exists(ndjson_file_path):
        logger.error(f"Data file not found: {ndjson_file_path}")
        logger.error(f"Please ensure the path is correct. Current working directory: {os.getcwd()}")
        return

    logger.info(f"Using data file: {ndjson_file_path}")

    try:
        # If RAGEngine is designed as a singleton imported as rag_engine, use that.
        # For now, assuming we instantiate it.
        engine = RAGEngine()
        
        logger.info("Calling RAGEngine.process_and_upsert_dataset()...")
        upserted_ids = engine.process_and_upsert_dataset(ndjson_file_path)
        
        if upserted_ids:
            logger.info(f"SUCCESS: Data ingestion process completed. {len(upserted_ids)} items upserted.")
            logger.info(f"First few upserted IDs: {upserted_ids[:5]}")
        else:
            logger.warning("Data ingestion process completed, but no items were reported as upserted.")
            logger.warning("Check previous logs for errors in document processing, embedding, or upserting.")
            
    except Exception as e:
        logger.error("FAILURE: An error occurred during the data ingestion test.", exc_info=True)
        # exc_info=True will include the stack trace in the log if using a logger that supports it.
        # If using basic print, you might want: print(f"Error: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    # It might be necessary to set the project ID via environment variable if app.config relies on it
    # and it's not picked up correctly when run as a script.
    # Example: os.environ['GOOGLE_CLOUD_PROJECT'] = 'learningemini'
    main() 