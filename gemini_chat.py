from google import genai
from google.genai.types import HttpOptions
from google.cloud import logging

# Initialize Cloud Logging
logging_client = logging.Client()
logger = logging_client.logger('gemini_chat_logs')

# Configuration for Gemini
PROJECT_ID = "learningemini"
REGION = "us-west1"

def chat_with_gemini(prompt_text):
    """Sends a prompt to the Gemini model and gets a response."""
    
    # Initialize Gemini client with Vertex AI
    client = genai.Client(
        vertexai=True,  # Enable Vertex AI
        project=PROJECT_ID,
        location=REGION,
        http_options=HttpOptions(api_version="v1")
    )
    
    # Generate response
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt_text
        )
        
        print("\n=== DEBUG INFO ===")
        print("Raw response:", response)
        print("Response type:", type(response))
        print("=== END DEBUG INFO ===\n")

        return response.text

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return f"An error occurred: {str(e)}"

# Test function
if __name__ == "__main__":
    print("Welcome to the Gemini Chat!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("Gemini is thinking...")
        response = chat_with_gemini(user_input)
        print(f"Gemini: {response}")
        print("-" * 40) 