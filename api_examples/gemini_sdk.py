"""
Gemini SDK Example

This file demonstrates how to use the Vertex AI SDK to interact with Gemini models,
including both base models and fine-tuned models. This approach is used in the main
application code in app/backend/models/gemini_chat.py.

Use this as a reference for:
1. Initializing Vertex AI with your project and location
2. Listing available models and endpoints
3. Using both base and fine-tuned Gemini models
4. Generating content with proper error handling
"""

import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Content, Part
from google.cloud import aiplatform

# Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "learningemini")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_ID = os.environ.get("MODEL_ID", "2279310694123831296")  # gemini-assistant-custom model
ENDPOINT_ID = os.environ.get("ENDPOINT_ID", "4660193172909981696")  # endpoint ID

def initialize_vertexai():
    """
    Initialize the Vertex AI SDK.
    
    This needs to be done once per session before using any Vertex AI services.
    """
    print(f"Initializing Vertex AI with project: {PROJECT_ID}, location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
def list_models():
    """
    List available models in the project.
    
    This helps you find what models are available to use, including fine-tuned models.
    """
    # Initialize Vertex AI for API calls
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    print("Listing Vertex AI models...")
    models = aiplatform.Model.list()
    print(f"Found {len(models)} models")
    for model in models:
        print(f"- {model.display_name} (ID: {model.name})")
    
    print("\nListing Vertex AI endpoints...")
    endpoints = aiplatform.Endpoint.list()
    print(f"Found {len(endpoints)} endpoints")
    for endpoint in endpoints:
        print(f"- {endpoint.display_name} (ID: {endpoint.name})")
        # Get deployed models for this endpoint
        try:
            deployed_models = endpoint.gca_resource.deployed_models
            for dm in deployed_models:
                model_id = dm.model.split('/')[-1]
                print(f"  - Deployed Model: {model_id}")
        except Exception as e:
            print(f"  Could not get deployed models: {e}")

def check_tuning_job():
    """
    Check if there are any tuning jobs.
    
    This is useful for monitoring the status of fine-tuning jobs.
    """
    from google.cloud.aiplatform.tuning import TuningJob
    
    print("Listing tuning jobs...")
    try:
        tuning_jobs = TuningJob.list()
        for job in tuning_jobs:
            print(f"- {job.display_name} (Status: {job.state})")
    except Exception as e:
        print(f"Error listing tuning jobs: {e}")

def get_model_response(prompt, use_tuned_model=True):
    """
    Generate a response from the model using the Vertex AI SDK.
    
    Args:
        prompt (str): The prompt text to send to the model
        use_tuned_model (bool): Whether to use the fine-tuned model or the base model
        
    Returns:
        str: The model's response text
    """
    try:
        # Initialize Vertex AI (idempotent)
        initialize_vertexai()
        
        if use_tuned_model:
            # Try using the fine-tuned model
            print(f"\nUsing fine-tuned model: {MODEL_ID}")
            model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}"
        else:
            # Use a base model
            print("\nUsing base model: gemini-1.5-pro")
            model_name = "gemini-1.5-pro"
            
        print(f"Loading model: {model_name}")
        model = GenerativeModel(model_name=model_name)
        
        # Create structured content for the prompt (same approach as in the app)
        structured_prompt = [
            Content(role="user", parts=[Part.from_text(prompt)])
        ]
        print(f"Sending structured prompt: {structured_prompt}")
        
        # Generate response
        response = model.generate_content(structured_prompt)
        
        if response.text:
            print("Successfully generated content!")
            return response.text
        else:
            print("No text in response")
            return None
            
    except Exception as e:
        print(f"Error during prediction: {type(e).__name__} - {str(e)}")
        
        # If using the fine-tuned model failed, try falling back to the base model
        if use_tuned_model:
            print("Falling back to base model due to error...")
            return get_model_response(prompt, use_tuned_model=False)
        else:
            # If the base model also failed
            return f"An error occurred: {str(e)}"

def chat_example():
    """
    Demonstrate using a chat session with the model.
    
    This shows how to have a multi-turn conversation with the model.
    """
    try:
        print("\nStarting a chat session...")
        
        # Initialize a model and chat session
        model = GenerativeModel("gemini-1.5-pro")
        chat = model.start_chat(history=[])
        
        # First message
        print("\nUser: Hello, can you help me learn about machine learning?")
        response1 = chat.send_message("Hello, can you help me learn about machine learning?")
        print(f"AI: {response1.text}")
        
        # Second message (follows up on the conversation)
        print("\nUser: What's the difference between supervised and unsupervised learning?")
        response2 = chat.send_message("What's the difference between supervised and unsupervised learning?")
        print(f"AI: {response2.text}")
        
        # You can access the history if needed
        print("\nChat history:")
        for message in chat.history:
            role = message.role
            text = message.parts[0].text
            print(f"{role.capitalize()}: {text[:50]}...")
            
        return True
    except Exception as e:
        print(f"Error in chat session: {e}")
        return False

def main():
    """Run the example code to demonstrate Vertex AI SDK usage."""
    print(f"Using Project ID: {PROJECT_ID}")
    print(f"Using Location: {LOCATION}")
    print(f"Using Model ID: {MODEL_ID}")
    print(f"Using Endpoint ID: {ENDPOINT_ID}")
    
    # Initialize Vertex AI
    initialize_vertexai()
    
    # List available models and endpoints
    list_models()
    
    # Check tuning jobs
    check_tuning_job()
    
    # Example prompt
    prompt = "What is machine learning?"
    
    print(f"\nGenerating response for prompt: '{prompt}'")
    print("\n=== FINE-TUNED MODEL RESPONSE ===")
    fine_tuned_response = get_model_response(prompt, use_tuned_model=True)
    
    if fine_tuned_response:
        print("\nGenerated Text from fine-tuned model:")
        print(fine_tuned_response)
    else:
        print("\nFailed to generate a response from the fine-tuned model.")
    
    print("\n=== BASE MODEL RESPONSE ===")
    base_response = get_model_response(prompt, use_tuned_model=False)
    
    if base_response:
        print("\nGenerated Text from base model:")
        print(base_response)
    else:
        print("\nFailed to generate a response from the base model.")
    
    # Demonstrate chat functionality
    print("\n=== CHAT SESSION EXAMPLE ===")
    chat_example()

if __name__ == "__main__":
    main() 