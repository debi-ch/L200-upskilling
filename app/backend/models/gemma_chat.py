from google.cloud import aiplatform
from google.cloud import logging
import re
import time

# Initialize Cloud Logging
logging_client = logging.Client()
logger = logging_client.logger('gemma_chat_logs')

# **IMPORTANT:**  Replace these with your actual Project ID, Region, and Endpoint ID from Vertex AI!
PROJECT_ID = "learningemini"  # <---- PUT YOUR GCP PROJECT ID HERE
REGION = "us-west1"        # <---- PUT YOUR GCP REGION HERE (e.g., "us-central1")
ENDPOINT_ID = "5091895522835300352"  # <---- PUT YOUR VERTEX AI ENDPOINT ID HERE

def chat_with_gemma(prompt_text):
    """Sends a prompt to the Gemma model endpoint and gets a response."""
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    # Simplified prompt structure
    instances = [{
        "prompt": prompt_text,
        "max_tokens": 2048  # Try setting it in the instance
    }]

    parameters = {
        "maxOutputTokens": 2048,
        "temperature": 0.7,
        "topP": 0.95,
        "topK": 40
    }

    try:
        prediction = endpoint.predict(instances=instances, parameters=parameters)
        
        print("\n=== DEBUG INFO ===")
        print("Raw prediction object:", prediction)
        print("Prediction type:", type(prediction))
        print("Predictions array length:", len(prediction.predictions))
        print("First prediction length:", len(prediction.predictions[0]))
        print("First prediction:", prediction.predictions[0])
        print("Parameters used:", parameters)
        print("=== END DEBUG INFO ===\n")

        response_text = prediction.predictions[0]
        if isinstance(response_text, str):
            # Clean up the response if needed
            response_text = response_text.replace("Prompt:", "").replace(prompt_text, "").strip()
            if response_text.startswith("Output:"):
                response_text = response_text[7:].strip()
        return response_text

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"An error occurred: {str(e)}"

# Main function to run the chatbot in the CLI - only used when running gemma_chat.py directly 
if __name__ == "__main__":
    print("Welcome to the Deb Chat!")
    print("Type 'poopie' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "poopie":
            print("Goodbye!")
            break

        print("Gemma is thinking...")
        gemma_response = chat_with_gemma(user_input)
        print(f"Gemma: {gemma_response}")
        print("-" * 40) # Separator for readability
