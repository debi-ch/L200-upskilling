"""
Gemini Model Integration

This module provides integration with Google's Gemini models.
It re-exports functions from gemini_chat_refactored.py for compatibility.
"""

# Import all necessary functions from the refactored module
from app.backend.models.gemini_chat_refactored import (
    chat_with_gemini,
    set_model_preference,
    get_current_model_name,
    USE_TUNED_MODEL
)

# For direct execution testing
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