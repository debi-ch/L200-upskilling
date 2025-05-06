#!/bin/bash
#
# Script to run the Gemini SDK example
#
# This script demonstrates how to use the Vertex AI SDK to interact with Gemini models.
# It sets the environment variables needed for the example and runs the Python script.
#
# To run this script:
# 1. Make sure you have authenticated with Google Cloud: gcloud auth application-default login
# 2. Adjust the environment variables below if needed
# 3. Run: ./run_gemini_sdk.sh

echo "Gemini SDK Example - Shows how to use Vertex AI SDK with Gemini models"
echo "==========================================================================="

# Set environment variables
export PROJECT_ID="learningemini"  # Your GCP project ID
export LOCATION="us-central1"      # Your region
export MODEL_ID="2279310694123831296"  # gemini-assistant-custom model
export ENDPOINT_ID="4660193172909981696"  # The endpoint ID for the model

# Print configuration
echo "Configuration:"
echo "- PROJECT_ID: $PROJECT_ID"
echo "- LOCATION: $LOCATION"
echo "- MODEL_ID: $MODEL_ID"
echo "- ENDPOINT_ID: $ENDPOINT_ID"
echo ""

# Run the SDK script
echo "Running Gemini SDK example..."
python3 gemini_sdk.py

echo ""
echo "Example completed!"
echo "This demonstrates the same approach used in the main application (app/backend/models/gemini_chat.py)" 