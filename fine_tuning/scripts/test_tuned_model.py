import os
import argparse
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

# Configuration - copied from gemini_chat.py
PROJECT_ID = "learningemini"
TUNED_MODEL_PROJECT_NUM = "708208532564"  # Fine-tuned model project number
TUNED_MODEL_REGION = "us-central1"  # Fine-tuned model region
# Full resource name for the fine-tuned model
TUNED_MODEL_NAME = f"projects/{TUNED_MODEL_PROJECT_NUM}/locations/{TUNED_MODEL_REGION}/models/2279310694123831296"

# Function to initialize Vertex AI
def init_vertex_ai():
    """Initialize Vertex AI with project and location settings"""
    project_id = "learningemini"
    location = "us-central1"
    
    print(f"Initializing Vertex AI with project: {project_id}, location: {location}")
    vertexai.init(project=project_id, location=location)

# Function to test the fine-tuned model
def test_model(model_name, prompt):
    """Test the fine-tuned model with a given prompt"""
    print(f"Loading model: {model_name}")
    model = GenerativeModel(model_name=model_name)
    
    print(f"\nTesting model with prompt: '{prompt}'")
    response = model.generate_content(prompt)
    
    print("\nModel response:")
    print("="*80)
    print(response.text)
    print("="*80)
    
    return response.text

# Main function to run the test
def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Gemini model")
    parser.add_argument("--model_name", type=str, help="The name of the fine-tuned model")
    parser.add_argument("--prompt", type=str, default="What are the advantages of fine-tuning language models?", 
                        help="The prompt to test with")
    args = parser.parse_args()
    
    if not args.model_name:
        print("Please specify the model name with --model_name")
        print("Example: python test_tuned_model.py --model_name projects/learningemini/locations/us-central1/models/your-model-id")
        return
    
    # Initialize Vertex AI
    init_vertex_ai()
    
    # Test the model
    response = test_model(args.model_name, args.prompt)
    
    # Compare with base model if needed
    if input("\nWould you like to compare with the base Gemini model? (y/n): ").lower() == 'y':
        print("\nTesting with base Gemini 2.0 Flash model:")
        base_model = GenerativeModel(model_name="gemini-2.0-flash-001")
        base_response = base_model.generate_content(args.prompt)
        
        print("\nBase model response:")
        print("="*80)
        print(base_response.text)
        print("="*80)

# Function to integrate with chatbot_app.py
def integrate_with_app(model_name):
    """Instructions to integrate the fine-tuned model with the existing app"""
    print("\nTo integrate your fine-tuned model with the chatbot app:")
    print("1. Open the chatbot_app.py file")
    print("2. Find where the model is initialized, usually something like:")
    print("   model = GenerativeModel('gemini-1.5-pro') or similar")
    print(f"3. Replace it with: model = GenerativeModel('{model_name}')")
    print("4. Save the file and run the app to test with your fine-tuned model")
    
    print("\nHere's a code snippet you can use:")
    print("-"*80)
    print(f"""from vertexai.generative_models import GenerativeModel

# Initialize the fine-tuned model
model = GenerativeModel(model_name="{model_name}")

# Generate a response
def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text
""")
    print("-"*80)

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
    main()
    
    print("\nAfter testing, you can integrate this model with your chatbot app.")
    model_name = input("Enter the model name to generate integration code (or press Enter to skip): ")
    if model_name:
        integrate_with_app(model_name)
    
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