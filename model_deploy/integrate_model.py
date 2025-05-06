import os
import argparse
import re
import vertexai
from vertexai.generative_models import GenerativeModel

# Constants
PROJECT_ID = "learningemini"
LOCATION = "us-central1"
CHATBOT_APP_PATH = "chatbot_app.py"

def init_vertex_ai():
    """Initialize Vertex AI"""
    print(f"Initializing Vertex AI with project: {PROJECT_ID}, location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def validate_model(model_name):
    """Validate the model exists and can be used"""
    print(f"Validating model: {model_name}")
    try:
        model = GenerativeModel(model_name=model_name)
        response = model.generate_content("Hello, are you working?")
        print(f"Model validation successful. Sample response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False

def update_chatbot_app(model_name):
    """Update the chatbot app to use the fine-tuned model"""
    print(f"Updating {CHATBOT_APP_PATH} to use model: {model_name}")
    
    try:
        # Read the current file
        with open(CHATBOT_APP_PATH, 'r') as file:
            content = file.read()
        
        # Create a backup
        with open(f"{CHATBOT_APP_PATH}.backup", 'w') as file:
            file.write(content)
            print(f"Created backup at {CHATBOT_APP_PATH}.backup")
        
        # Find and replace the model initialization
        model_pattern = r"(GenerativeModel\(['\"])(.*?)(['\"])"
        
        if re.search(model_pattern, content):
            # Replace existing model with fine-tuned model
            updated_content = re.sub(model_pattern, f"\\1{model_name}\\3", content)
            
            # Write the updated content
            with open(CHATBOT_APP_PATH, 'w') as file:
                file.write(updated_content)
            
            print(f"Successfully updated {CHATBOT_APP_PATH}")
            return True
        else:
            print(f"Could not find GenerativeModel initialization in {CHATBOT_APP_PATH}")
            print("You may need to manually update the file.")
            return False
    
    except Exception as e:
        print(f"Error updating chatbot app: {e}")
        return False

def check_integration():
    """Check if integration was successful by validating code changes"""
    try:
        with open(CHATBOT_APP_PATH, 'r') as file:
            content = file.read()
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "GenerativeModel" in line:
                print("\nFound model initialization at line", i+1)
                print(">" + line)
                
                # Show surrounding context
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print("\nContext:")
                for j in range(start, end):
                    prefix = "  " if j != i else "* "
                    print(f"{prefix}{j+1}: {lines[j]}")
                
                return True
        
        print("Could not find model initialization in the updated file.")
        return False
    
    except Exception as e:
        print(f"Error checking integration: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Integrate a fine-tuned Gemini model with the chatbot app")
    parser.add_argument("--model_name", required=True, type=str, 
                        help="The name of the fine-tuned model to use")
    args = parser.parse_args()
    
    # Initialize Vertex AI
    init_vertex_ai()
    
    # Validate the model exists
    if not validate_model(args.model_name):
        print("Model validation failed. Please check the model name and try again.")
        return
    
    # Update the chatbot app
    if update_chatbot_app(args.model_name):
        # Check the integration
        check_integration()
        
        print("\nIntegration complete! You can now run the chatbot app with your fine-tuned model:")
        print(f"python {CHATBOT_APP_PATH}")
    else:
        print("\nIntegration failed. You may need to manually update the chatbot app.")
        print(f"Add this code to {CHATBOT_APP_PATH}:")
        print("-"*80)
        print(f"from vertexai.generative_models import GenerativeModel")
        print(f"model = GenerativeModel(model_name='{args.model_name}')")
        print("-"*80)

if __name__ == "__main__":
    main() 