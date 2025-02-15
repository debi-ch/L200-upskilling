from google.cloud import aiplatform
import re

# **IMPORTANT:**  Replace these with your actual Project ID, Region, and Endpoint ID from Vertex AI!
PROJECT_ID = "learningemini"  # <---- PUT YOUR GCP PROJECT ID HERE
REGION = "us-west1"        # <---- PUT YOUR GCP REGION HERE (e.g., "us-central1")
ENDPOINT_ID = "5091895522835300352"  # <---- PUT YOUR VERTEX AI ENDPOINT ID HERE

def chat_with_gemma(prompt_text):
    """Sends a prompt to the Gemma model endpoint and gets a response."""

    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    instances = [{"prompt": prompt_text}] #inputs as expected by gemma 

     # Add the 'maxOutputTokens' parameter here:
    parameters = {
        "maxOutputTokens": 500  # You can adjust this value
    }

    prediction = endpoint.predict(instances=instances, parameters=parameters)

    print("--- Prediction object (for debugging) ---") # ADDED
    print(prediction) # ADDED
    print("--- End Prediction object ---") # ADDED

    print(len(prediction.predictions[0]))

    full_prediction = prediction.predictions[0]
    match = re.search(r"Output:\s*(.*)", full_prediction, re.DOTALL) # finds everything after output
    if match:
        response_text = match.group(1).strip()
    else:
        response_text = full_prediction

    print(response_text)
    return response_text

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
