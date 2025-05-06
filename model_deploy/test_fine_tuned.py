import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

# Configuration - copied from gemini_chat.py
PROJECT_ID = "learningemini"
TUNED_MODEL_PROJECT_NUM = "708208532564"  # Fine-tuned model project number
TUNED_MODEL_REGION = "us-central1"  # Fine-tuned model region
# Full resource name for the fine-tuned model
TUNED_MODEL_NAME = f"projects/{TUNED_MODEL_PROJECT_NUM}/locations/{TUNED_MODEL_REGION}/models/2279310694123831296"

def test_tuned_model():
    """Test the fine-tuned model with a simple prompt."""
    print("=== Testing Fine-Tuned Model ===")
    print(f"Model Name: {TUNED_MODEL_NAME}")
    print(f"Project: {TUNED_MODEL_PROJECT_NUM}")
    print(f"Region: {TUNED_MODEL_REGION}")
    
    # Initialize Vertex AI
    print("\nInitializing Vertex AI...")
    vertexai.init(project=TUNED_MODEL_PROJECT_NUM, location=TUNED_MODEL_REGION)
    
    try:
        # Load the fine-tuned model
        print("\nLoading the model...")
        model = GenerativeModel(model_name=TUNED_MODEL_NAME)
        
        # Test prompt
        test_prompt = "What is machine learning?"
        print(f"\nTest Prompt: '{test_prompt}'")
        
        # Create structured content for the prompt
        structured_prompt = [
            Content(role="user", parts=[Part.from_text(test_prompt)])
        ]
        
        # Generate response
        print("\nGenerating response...")
        response = model.generate_content(structured_prompt)
        
        # Print response details
        print("\n=== Response Details ===")
        print(f"Response type: {type(response)}")
        print(f"Raw response: {response}")
        
        # Print the generated text
        print("\n=== Generated Text ===")
        print(response.text)
        
        return True
    except Exception as e:
        print(f"\nError testing fine-tuned model: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tuned_model()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed.")
        
    # Let's also try using a different model name format to see if that works
    print("\n\n=== Trying Alternative Model Format ===")
    try:
        # Initialize Vertex AI
        vertexai.init(project=TUNED_MODEL_PROJECT_NUM, location=TUNED_MODEL_REGION)
        
        # Just using the model ID directly
        alternative_model_name = "2279310694123831296"
        print(f"Using alternative model name: {alternative_model_name}")
        
        alt_model = GenerativeModel(model_name=alternative_model_name)
        
        # Test prompt
        test_prompt = "What is machine learning?"
        print(f"Test Prompt: '{test_prompt}'")
        
        # Create structured content for the prompt
        structured_prompt = [
            Content(role="user", parts=[Part.from_text(test_prompt)])
        ]
        
        # Generate response
        print("Generating response...")
        response = alt_model.generate_content(structured_prompt)
        
        # Print the generated text
        print("\nGenerated Text:")
        print(response.text)
        
        print("\n✅ Alternative approach succeeded!")
    except Exception as e:
        print(f"\n❌ Alternative approach failed: {type(e).__name__} - {str(e)}") 