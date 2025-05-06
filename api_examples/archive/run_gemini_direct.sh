#!/bin/bash

# Set environment variables
export PROJECT_ID="learningemini"  # Your GCP project ID
export LOCATION="us-central1"      # Your region
export MODEL_ID="2279310694123831296"  # gemini-assistant-custom model
export ENDPOINT_ID="4660193172909981696"  # The endpoint ID for the model

# Uncomment and set this if you have a Gemini API key
# export GEMINI_API_KEY="your-api-key-here"

# Run the direct API script
echo "Running Gemini Direct API script..."
python3 gemini_direct_api.py 