"""
Gemini Model Integration

This module provides integration with Google's Gemini models,
including both base and fine-tuned versions.
"""

from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

# Import configuration and logging
from app.config import GCP_PROJECT_ID, GCP_LOCATION, ModelConfig
from app.utils.logging_utils import ChatbotLogger

# Initialize logger
logger = ChatbotLogger("gemini_model")

# Flag to control which model to use
USE_TUNED_MODEL = ModelConfig.USE_TUNED_MODEL

def chat_with_gemini(prompt_text):
    """
    Sends a prompt to the selected Gemini model and gets a response.
    
    Args:
        prompt_text (str): The prompt text to send to the model
        
    Returns:
        str: The model's response text
    """
    if USE_TUNED_MODEL:
        # Try the fine-tuned model (gemini-assistant-custom)
        return chat_with_model_internal(
            ModelConfig.GEMINI_TUNED_MODEL_NAME, 
            ModelConfig.GEMINI_TUNED_MODEL_PROJECT_NUM, 
            ModelConfig.GEMINI_TUNED_MODEL_REGION, 
            prompt_text, 
            "fine-tuned"
        )
    else:
        # Use the base model
        return chat_with_model_internal(
            ModelConfig.GEMINI_BASE_MODEL_ID, 
            GCP_PROJECT_ID, 
            GCP_LOCATION, 
            prompt_text, 
            "base"
        )

def chat_with_model_internal(model_id, project_id, location, prompt_text, model_type_str):
    """
    Internal function to call a specific Gemini model.
    
    Args:
        model_id (str): The model ID or resource name
        project_id (str): The Google Cloud project ID
        location (str): The Google Cloud region
        prompt_text (str): The prompt text to send to the model
        model_type_str (str): A string describing the model type (for logging)
        
    Returns:
        str: The model's response text
    """
    try:
        logger.info(
            f"Using {model_type_str} model",
            model_id=model_id,
            project_id=project_id,
            location=location
        )
        
        # Initialize Vertex AI (idempotent)
        vertexai.init(project=project_id, location=location)
        
        # Load the model
        model = GenerativeModel(model_name=model_id)
        
        # Create structured content for the prompt
        structured_prompt = [
            Content(role="user", parts=[Part.from_text(prompt_text)])
        ]
        logger.debug("Sending structured prompt", prompt=str(structured_prompt))
        
        # Generate response
        start_time = __import__('time').time()
        response = model.generate_content(structured_prompt)
        end_time = __import__('time').time()
        
        logger.info(
            f"Model response generated",
            model_type=model_type_str,
            response_time=end_time - start_time,
            response_length=len(response.text) if hasattr(response, 'text') else 0
        )
        
        return response.text
            
    except Exception as e:
        logger.error(
            f"Error during prediction with {model_type_str} model",
            error_type=type(e).__name__,
            error_message=str(e)
        )
        
        # If the fine-tuned model failed, try falling back to the base model
        if model_type_str == "fine-tuned":
            logger.warning("Falling back to base model due to error")
            set_model_preference(False)  # Change the preference
            # Call the base model directly, avoiding recursion into chat_with_gemini
            return chat_with_model_internal(
                ModelConfig.GEMINI_BASE_MODEL_ID, 
                GCP_PROJECT_ID, 
                GCP_LOCATION, 
                prompt_text, 
                "base (fallback)"
            )
        else:
            # If the base model (or fallback) also failed
            return f"An error occurred with the {model_type_str} model: {str(e)}"

def set_model_preference(use_tuned_model=True):
    """
    Set whether to use the tuned model or base model.
    
    Args:
        use_tuned_model (bool): True to use the tuned model, False to use the base model
        
    Returns:
        bool: The new model preference
    """
    global USE_TUNED_MODEL
    USE_TUNED_MODEL = use_tuned_model
    model_name = "Tuned Model (gemini-assistant-custom)" if USE_TUNED_MODEL else "Base Model"
    logger.info(f"Model preference set to: {model_name}")
    return USE_TUNED_MODEL

def get_current_model_name():
    """
    Returns the name of the currently active model.
    
    Returns:
        str: A descriptive name of the current model
    """
    if USE_TUNED_MODEL:
        return "Gemini (Fine-tuned: gemini-assistant-custom)"
    else:
        return "Gemini (Base)"

# Test function to run when this module is executed directly
if __name__ == "__main__":
    print("Welcome to the Gemini Chat!")
    print("Type 'exit' to quit.")
    print("Type 'switch' to toggle between base and fine-tuned models.")
    
    print(f"Currently using: {get_current_model_name()}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "switch":
            # Use the function to set the preference
            set_model_preference(not USE_TUNED_MODEL)
            print(f"Switched to: {get_current_model_name()}")
            continue

        print(f"{get_current_model_name()} is thinking...")
        response = chat_with_gemini(user_input)
        print(f"Response: {response}")
        print("-" * 40) 