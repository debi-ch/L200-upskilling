"""
Temporary test script to check VectorSearchDB initialization.
"""
import os
import sys

# Add the project root to the Python path
# This allows us to import from 'app.' directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.backend.rag.vector_db import VectorSearchDB
from app.utils.logging_utils import ChatbotLogger # For any top-level logging

logger = ChatbotLogger("test_vector_db_init")

def main():
    logger.info("Attempting to initialize VectorSearchDB...")
    
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set if using a service account key,
    # or that gcloud auth application-default login has been run.
    # Also ensure GOOGLE_CLOUD_PROJECT is set in the environment or app.config is correct.
    
    db = VectorSearchDB()
    initialized_successfully = db.initialize()
    
    if initialized_successfully:
        logger.info("SUCCESS: VectorSearchDB initialized successfully!")
        logger.info(f"Endpoint display name: {db.index_endpoint.display_name if db.index_endpoint else 'N/A'}")
        logger.info(f"Endpoint resource name: {db.index_endpoint.resource_name if db.index_endpoint else 'N/A'}")
        
        # You could potentially try a dummy search here if your index has data,
        # but for now, just initialization is the goal.
        # Example: results = db.search("test query")
        # logger.info(f"Dummy search results: {results}")

    else:
        logger.error("FAILURE: VectorSearchDB did NOT initialize successfully.")

if __name__ == "__main__":
    main() 