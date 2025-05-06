#!/usr/bin/env python3

"""
Streamlined Gemini Model Fine-Tuning Script
------------------------------------------
This script handles the fine-tuning of Gemini 2.0 Flash model using the
Vertex AI TuningJob API. It includes functionality to prepare the GCS bucket,
start the fine-tuning job, and test the resulting model.
"""

import os
import logging
import time
import subprocess
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.tuning import TuningJob, SupervisedTuningSpec, HyperParameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get("PROJECT_ID") or "your-project-id"
LOCATION = os.environ.get("LOCATION") or "us-central1"  # Must use a region that supports Gemini fine-tuning
BUCKET_NAME = os.environ.get("BUCKET_NAME") or f"{PROJECT_ID}-gemini-tuning"

# GCS paths
GCS_BUCKET_URI = f"gs://{BUCKET_NAME}"
DATASET_FILENAME = "openassistant_best_replies_train_reformatted.jsonl"
LOCAL_DATASET_PATH = f"data/openassistant-guanaco/{DATASET_FILENAME}"
GCS_DATASET_URI = f"{GCS_BUCKET_URI}/data/{DATASET_FILENAME}"
OUTPUT_DIR = f"{GCS_BUCKET_URI}/output"

# Model settings
BASE_MODEL = "gemini-2.0-flash"  # Use gemini-2.0-flash or gemini-2.0-flash-lite
TUNED_MODEL_NAME = f"{BASE_MODEL}-finetuned-assistant"

def ensure_bucket_exists():
    """Ensure the GCS bucket exists, creating it if necessary."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        if not bucket.exists():
            logger.info(f"Creating bucket {BUCKET_NAME} in location {LOCATION}")
            bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
            logger.info(f"Bucket {bucket.name} created")
        else:
            logger.info(f"Bucket {BUCKET_NAME} already exists")
            
        return True
    except Exception as e:
        logger.error(f"Error ensuring bucket exists: {e}")
        return False

def upload_to_gcs(local_path, gcs_path):
    """Upload a file to Google Cloud Storage."""
    if not os.path.exists(local_path):
        logger.error(f"Local file not found: {local_path}")
        return False
        
    try:
        logger.info(f"Uploading {local_path} to {gcs_path}")
        result = subprocess.run(
            ["gsutil", "cp", local_path, gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Upload successful: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Upload failed: {e.stderr}")
        return False

def fine_tune_gemini():
    """Fine-tune the Gemini model using Supervised Tuning."""
    logger.info(f"Starting fine-tuning process for {BASE_MODEL}")
    logger.info(f"Using dataset: {GCS_DATASET_URI}")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Define hyperparameters for fine-tuning
        hyperparameters = HyperParameters(
            batch_size=8,
            learning_rate=1e-5,
            epochs=2
        )
        
        # Create a supervised tuning spec
        tuning_spec = SupervisedTuningSpec(
            training_dataset=GCS_DATASET_URI,
            output_directory=OUTPUT_DIR
        )
        
        # Create and run the tuning job
        job = TuningJob(
            display_name=TUNED_MODEL_NAME,
            base_model=BASE_MODEL,
            tuning_spec=tuning_spec,
            hyperparameters=hyperparameters
        )
        
        logger.info("Starting fine-tuning job...")
        model = job.run()
        
        logger.info(f"Fine-tuning completed successfully!")
        logger.info(f"Model name: {model.name}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        
        return model.name
    except GoogleAPIError as e:
        logger.error(f"API error during fine-tuning: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during fine-tuning: {e}")
        return None

def test_fine_tuned_model(model_name):
    """Test the fine-tuned model with a few prompts."""
    logger.info(f"Testing fine-tuned model: {model_name}")
    
    try:
        # Initialize the model
        model = GenerativeModel(model_name)
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain how fine-tuning works.",
            "Write a short poem about AI."
        ]
        
        for prompt in test_prompts:
            logger.info(f"Testing prompt: {prompt}")
            response = model.generate_content(prompt)
            logger.info(f"Response: {response.text}")
            
        return True
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def main():
    """Main execution function."""
    logger.info(f"Starting Gemini fine-tuning process")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"Location: {LOCATION}")
    logger.info(f"Base model: {BASE_MODEL}")
    
    # Ensure the GCS bucket exists
    if not ensure_bucket_exists():
        logger.error("Failed to ensure GCS bucket exists. Exiting.")
        return
    
    # Upload dataset to GCS if it exists locally
    if os.path.exists(LOCAL_DATASET_PATH):
        logger.info(f"Found local dataset at {LOCAL_DATASET_PATH}")
        if not upload_to_gcs(LOCAL_DATASET_PATH, GCS_DATASET_URI):
            logger.error("Failed to upload dataset to GCS. Exiting.")
            return
    else:
        logger.warning(f"Local dataset not found at {LOCAL_DATASET_PATH}")
        logger.info(f"Will use dataset directly from GCS at {GCS_DATASET_URI} if it exists")
    
    # Start fine-tuning
    tuned_model_name = fine_tune_gemini()
    
    if tuned_model_name:
        logger.info(f"Fine-tuning job completed successfully")
        logger.info(f"Testing the fine-tuned model")
        test_fine_tuned_model(tuned_model_name)
    else:
        logger.error("Fine-tuning job failed or was incomplete")
    
    logger.info("Process completed")

if __name__ == "__main__":
    main() 