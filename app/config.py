"""
Application Configuration

This module provides centralized configuration settings for the application,
with support for environment variables and different environments (dev, prod).
"""

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).parent.resolve()

# Environment: development, testing, or production
ENV = os.environ.get("CHATBOT_ENV", "development")

# Google Cloud settings
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "learningemini")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-west1")

# Model settings
class ModelConfig:
    # Gemini models
    GEMINI_BASE_MODEL_ID = os.environ.get("GEMINI_BASE_MODEL_ID", "gemini-2.0-flash")
    GEMINI_TUNED_MODEL_PROJECT_NUM = os.environ.get("GEMINI_TUNED_MODEL_PROJECT_NUM", "708208532564")
    GEMINI_TUNED_MODEL_REGION = os.environ.get("GEMINI_TUNED_MODEL_REGION", "us-west1")
    GEMINI_TUNED_MODEL_ID = os.environ.get("GEMINI_TUNED_MODEL_ID", "2279310694123831296")
    
    # Full resource name for the fine-tuned model
    GEMINI_TUNED_MODEL_NAME = os.environ.get(
        "GEMINI_TUNED_MODEL_NAME", 
        f"projects/{GEMINI_TUNED_MODEL_PROJECT_NUM}/locations/{GEMINI_TUNED_MODEL_REGION}/models/{GEMINI_TUNED_MODEL_ID}"
    )
    
    # Default to using the tuned model
    USE_TUNED_MODEL = os.environ.get("USE_TUNED_MODEL", "true").lower() == "true"
    
    # Gemma models
    GEMMA_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "/path/to/gemma/model")

# Storage settings
class StorageConfig:
    # Database settings (for Firebase/Firestore)
    FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "chat_sessions")
    
    # Local storage paths
    PROMPTS_FILE = os.path.join(BASE_DIR, "backend", "prompts", "prompts.json")
    LOG_FILE = os.path.join(BASE_DIR.parent, "logs", "chatbot_app.log")

# Logging settings
class LoggingConfig:
    # Whether to use Google Cloud Logging
    USE_CLOUD_LOGGING = ENV != "development"
    
    # Log levels
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    
    # Logger names
    APP_LOGGER = "chatbot_app"
    MODEL_LOGGER = "model_interactions"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR.parent, "logs"), exist_ok=True)

# Web app settings
class WebAppConfig:
    # Streamlit settings
    PAGE_TITLE = "AI Travel Assistant"
    PAGE_ICON = "🌎"
    
    # Default model
    DEFAULT_MODEL = "Gemini"
    
    # Static files
    STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")
    CSS_FILE = os.path.join(STATIC_DIR, "style.css")
    
    # Templates
    TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")

# Fine-tuning settings
class FineTuningConfig:
    # Dataset paths
    DATASET_DIR = os.path.join(BASE_DIR.parent, "fine_tuning", "data")
    OPENASSISTANT_DATASET_PATH = os.path.join(DATASET_DIR, "openassistant-guanaco")
    
    # GCS bucket settings for fine-tuning
    BUCKET_NAME = os.environ.get("BUCKET_NAME", f"{GCP_PROJECT_ID}-vertex-tuning")
    GCS_BUCKET_URI = f"gs://{BUCKET_NAME}"
    DATASET_FILENAME = "openassistant_best_replies_train_reformatted.jsonl"
    GCS_DATASET_URI = f"{GCS_BUCKET_URI}/data/{DATASET_FILENAME}"
    OUTPUT_DIR = f"{GCS_BUCKET_URI}/output"
    
    # Fine-tuning parameters
    BASE_MODEL = "gemini-2.0-flash"
    TUNED_MODEL_NAME = "gemini-travel-assistant"
    
    # Hyperparameters
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
    EPOCHS = int(os.environ.get("EPOCHS", "2"))

# Development-specific settings
if ENV == "development":
    # Override certain settings for development
    LoggingConfig.USE_CLOUD_LOGGING = False
    LoggingConfig.LOG_LEVEL = "DEBUG"

# Testing-specific settings
elif ENV == "testing":
    # Override certain settings for testing
    pass

# Production-specific settings
elif ENV == "production":
    # Override certain settings for production
    LoggingConfig.LOG_LEVEL = "WARNING" 