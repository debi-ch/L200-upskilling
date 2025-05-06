#!/usr/bin/env python3

import os
import time
import logging
from google.cloud import storage
import vertexai
from vertexai.tuning import sft

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID", "your-project-id")
LOCATION = os.environ.get("LOCATION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME", f"{PROJECT_ID}-tuning-bucket")

# Set up paths
DATASET_URI = os.environ.get("DATASET_URI", "gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"gs://{BUCKET_NAME}/tuning-output")

def setup_env():
    """Set up the environment for fine-tuning."""
    logger.info("Setting up environment for fine-tuning...")
    
    # Check if bucket exists, create if it doesn't
    storage_client = storage.Client(project=PROJECT_ID)
    if not storage_client.lookup_bucket(BUCKET_NAME):
        logger.info(f"Creating bucket {BUCKET_NAME}...")
        bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
        logger.info(f"Bucket {bucket.name} created.")
    else:
        logger.info(f"Bucket {BUCKET_NAME} already exists.")
    
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Initialized Vertex AI with project {PROJECT_ID} in {LOCATION}")

def fine_tune_gemini():
    """Fine-tune a Gemini 2.0 Flash model."""
    logger.info("Starting fine-tuning process...")
    
    # Model to fine-tune
    source_model = "gemini-2.0-flash-001"
    tuned_model_name = "tuned-gemini-flash"
    
    # Start fine-tuning job
    logger.info(f"Creating fine-tuning job for {source_model}...")
    sft_tuning_job = sft.train(
        source_model=source_model,
        train_dataset=DATASET_URI,
        # Optional parameters:
        # validation_dataset="gs://your-bucket/validation.jsonl",  # If you have a validation dataset
        tuned_model_display_name=tuned_model_name,
        # hyperparameters can be customized if needed:
        # epochs=3,  
        # learning_rate_multiplier=1.0,
        # adapter_size="MEDIUM",  # Options: SMALL, MEDIUM, LARGE
    )
    
    logger.info("Fine-tuning job submitted. Waiting for completion...")
    
    # Poll for job completion
    while not sft_tuning_job.has_ended:
        time.sleep(60)  # Check every minute
        sft_tuning_job.refresh()
        logger.info("Fine-tuning job in progress...")
    
    if sft_tuning_job.state == "SUCCEEDED":
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
        logger.info(f"Tuned model endpoint: {sft_tuning_job.tuned_model_endpoint_name}")
        return sft_tuning_job
    else:
        logger.error(f"Fine-tuning failed. State: {sft_tuning_job.state}")
        return None

def test_fine_tuned_model(sft_tuning_job):
    """Test the fine-tuned model with a simple prompt."""
    if not sft_tuning_job:
        logger.error("No tuning job provided for testing")
        return
    
    logger.info("Testing the fine-tuned model...")
    from vertexai.generative_models import GenerativeModel
    
    tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)
    test_prompt = "What is machine learning?"
    
    response = tuned_model.generate_content(test_prompt)
    logger.info(f"Model response: {response.text}")

def main():
    """Main function to run the fine-tuning pipeline."""
    try:
        setup_env()
        tuning_job = fine_tune_gemini()
        if tuning_job:
            test_fine_tuned_model(tuning_job)
        
        logger.info("Fine-tuning process completed!")
    except Exception as e:
        logger.error(f"Error occurred during fine-tuning: {str(e)}")

if __name__ == "__main__":
    main() 