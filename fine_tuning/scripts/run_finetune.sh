#!/bin/bash

# Script to run the complete fine-tuning workflow
# Usage: ./run_finetune.sh [--skip-tuning] [--model MODEL_NAME] [--deploy] [--integrate]

set -e  # Exit on error

# Default values
SKIP_TUNING=false
MODEL_NAME=""
DEPLOY=false
INTEGRATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-tuning)
      SKIP_TUNING=true
      shift
      ;;
    --model)
      MODEL_NAME=$2
      shift 2
      ;;
    --deploy)
      DEPLOY=true
      shift
      ;;
    --integrate)
      INTEGRATE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./run_finetune.sh [--skip-tuning] [--model MODEL_NAME] [--deploy] [--integrate]"
      exit 1
      ;;
  esac
done

# Check required packages
echo "Checking required packages..."
pip install -U google-cloud-aiplatform vertexai datasets pandas numpy

# Configure Google Cloud if needed
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
  echo "Logging in to Google Cloud..."
  gcloud auth login
  gcloud auth application-default login
fi

# Step 1: Run the fine-tuning process
if [ "$SKIP_TUNING" = false ]; then
  echo "Step 1: Running fine-tuning process..."
  python finetune_gemini_streamlined.py
  
  # Extract model name from output if not provided
  if [ -z "$MODEL_NAME" ]; then
    echo "Please enter the model name from the fine-tuning output:"
    read MODEL_NAME
  fi
else
  echo "Skipping fine-tuning process..."
  
  # Require model name if skipping tuning
  if [ -z "$MODEL_NAME" ]; then
    echo "Please enter the model name to use:"
    read MODEL_NAME
  fi
fi

# Check if we have a model name
if [ -z "$MODEL_NAME" ]; then
  echo "Error: Model name is required to continue."
  exit 1
else
  echo "Using model: $MODEL_NAME"
fi

# Step 2: Test the model
echo -e "\nStep 2: Testing the fine-tuned model..."
python test_tuned_model.py --model_name "$MODEL_NAME"

# Step 3: Deploy the model if requested
if [ "$DEPLOY" = true ]; then
  echo -e "\nStep 3: Deploying the model to an endpoint..."
  python deploy_model.py --model_name "$MODEL_NAME"
  
  # Extract endpoint name from output
  echo "Please enter the endpoint name from the deployment output:"
  read ENDPOINT_NAME
else
  echo -e "\nSkipping model deployment..."
fi

# Step 4: Integrate with chatbot app if requested
if [ "$INTEGRATE" = true ]; then
  echo -e "\nStep 4: Integrating with chatbot application..."
  python integrate_model.py --model_name "$MODEL_NAME"
  
  echo -e "\nIntegration complete. You can now run the chatbot app with:"
  echo "python chatbot_app.py"
else
  echo -e "\nSkipping integration with chatbot app..."
fi

echo -e "\nWorkflow complete!"
echo "Summary:"
echo "- Model name: $MODEL_NAME"
if [ "$DEPLOY" = true ] && [ ! -z "$ENDPOINT_NAME" ]; then
  echo "- Endpoint: $ENDPOINT_NAME"
fi
echo "- Fine-tuning completed: $([ "$SKIP_TUNING" = false ] && echo "Yes" || echo "Skipped")"
echo "- Model deployed: $([ "$DEPLOY" = true ] && echo "Yes" || echo "No")"
echo "- App integration: $([ "$INTEGRATE" = true ] && echo "Yes" || echo "No")"

echo -e "\nYou can now use your fine-tuned model in your applications!" 