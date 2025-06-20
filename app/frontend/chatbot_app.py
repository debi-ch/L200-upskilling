"""
Main Streamlit application for the chatbot interface.
"""

import sys
import os
from pathlib import Path
import time
import base64

# Add the project root to sys.path
APPLICATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if APPLICATION_ROOT not in sys.path:
    sys.path.insert(0, APPLICATION_ROOT)

import streamlit as st
from app.backend.models.gemma_chat import chat_with_gemma
from app.backend.models.gemini_chat import chat_with_gemini, set_model_preference, get_current_model_name
from app.backend.models.gemini_rag import chat_with_gemini_rag
from app.backend.models.gemini_memory_rag import chat_with_memory_rag, get_memory_rag_status, add_user_interaction
from google.cloud import logging_v2
from app.backend.storage.db_operations import ChatDatabase
from app.backend.prompts.prompt_manager import PromptManager
from app.backend.models.weather_chat import chat_with_weather, get_weather_tool

# Initialize Cloud Logging
logging_client = logging_v2.Client()
logger = logging_client.logger('streamlit_chatbot_logs')

# Initialize database and prompt manager
db = ChatDatabase()
prompt_manager = PromptManager()

# Set page config
st.set_page_config(
    page_title="Travel & Weather Assistant",
    page_icon="🌍",
    layout="wide"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent.joinpath("static/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = db.create_session()
    logger.log_text(f"New chat session created: {st.session_state.session_id}")

# Initialize model choice if not present
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Gemma"

# Initialize fine-tuned model preference
if 'use_fine_tuned' not in st.session_state:
    st.session_state.use_fine_tuned = False

# Initialize RAG mode preference
if 'use_rag_mode' not in st.session_state:
    st.session_state.use_rag_mode = False

# Initialize Memory RAG mode preference
if 'use_memory_rag_mode' not in st.session_state:
    st.session_state.use_memory_rag_mode = False

# Initialize user identification
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{st.session_state.session_id}"

# Initialize weather preference
if "use_weather" not in st.session_state:
    st.session_state.use_weather = True

# Initialize messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a sidebar for session management, model selection, and prompt management
with st.sidebar:
    st.title("💬 Chat Settings")
    
    # Model selection with custom styling
    st.subheader("🤖 Model Selection")
    model_choice = st.radio(
        "Choose your AI companion:",
        ["Gemma", "Gemini"],
        index=0 if st.session_state.model_choice == "Gemma" else 1,
        help="Gemma is our Mexican pirate, Gemini is our local culture expert"
    )
    st.session_state.model_choice = model_choice
    
    # Add fine-tuned model toggle when Gemini is selected
    if model_choice == "Gemini":
        use_fine_tuned = st.checkbox(
            "Use fine-tuned model", 
            value=st.session_state.use_fine_tuned,
            help="Toggle between the base Gemini model and your fine-tuned version"
        )
        
        if use_fine_tuned != st.session_state.use_fine_tuned:
            st.session_state.use_fine_tuned = use_fine_tuned
            set_model_preference(use_fine_tuned)
            st.success(f"Switched to {get_current_model_name()}")
        
        # Add RAG mode toggle when Gemini is selected
        use_rag = st.checkbox(
            "Enable RAG (Travel Knowledge Base)",
            value=st.session_state.use_rag_mode,
            help="Enhance responses with specific travel document knowledge."
        )
        if use_rag != st.session_state.use_rag_mode:
            st.session_state.use_rag_mode = use_rag
            st.success(f"RAG Mode {'Enabled' if use_rag else 'Disabled'}")
        
        # Add Memory RAG mode toggle when Gemini is selected
        use_memory_rag = st.checkbox(
            "Enable Memory RAG (Personalized AI)",
            value=st.session_state.use_memory_rag_mode,
            help="Enable personalized responses based on your conversation history and preferences."
        )
        if use_memory_rag != st.session_state.use_memory_rag_mode:
            st.session_state.use_memory_rag_mode = use_memory_rag
            st.success(f"Memory RAG Mode {'Enabled' if use_memory_rag else 'Disabled'}")
            
            # Initialize Memory RAG when enabled
            if use_memory_rag:
                from app.backend.models.gemini_memory_rag import _ensure_memory_rag_initialized
                with st.spinner("Initializing Memory RAG system..."):
                    if _ensure_memory_rag_initialized():
                        st.success("✅ Memory RAG system initialized successfully!")
                    else:
                        st.error("❌ Failed to initialize Memory RAG system")
            
            # Show Memory RAG status
            if use_memory_rag:
                status = get_memory_rag_status()
                if status["ready"]:
                    st.info("🧠 Memory RAG system is ready for personalization!")
                else:
                    st.warning("⚠️ Memory RAG system is initializing...")
            
            # Add reload button for troubleshooting
            if st.button("🔄 Reload Memory RAG", help="Force reload if getting generic responses"):
                from app.backend.models.gemini_memory_rag import force_memory_rag_reload
                with st.spinner("Reloading Memory RAG..."):
                    reload_result = force_memory_rag_reload()
                    if reload_result.get('rag_context_length', 0) > 0:
                        st.success("✅ Memory RAG reloaded successfully!")
                    else:
                        st.warning("⚠️ Memory RAG reloaded but content retrieval may still have issues")
                    st.json(reload_result)
    
    # Weather integration toggle
    st.session_state.use_weather = st.checkbox(
        "Include Weather Information",
        value=True,
        help="Get real-time weather data for travel destinations"
    )
    
    # User identification section
    st.subheader("👤 User Profile")
    current_user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Your unique identifier for personalized responses"
    )
    if current_user_id != st.session_state.user_id:
        st.session_state.user_id = current_user_id
        st.success(f"User ID updated to: {current_user_id}")
    
    # Show Memory RAG status if enabled
    if st.session_state.use_memory_rag_mode:
        with st.expander("🧠 Memory RAG Status"):
            status = get_memory_rag_status()
            st.json(status)
    
    # Prompt Management
    st.subheader("📝 Prompt Management")
    current_model = st.session_state.model_choice.lower()
    
    # Display current prompt in a cleaner way
    with st.expander("Current Prompt", expanded=False):
        current_prompt = prompt_manager.get_latest_prompt(current_model)
        st.code(current_prompt, language="markdown")
    
    # Add new prompt version with better organization
    with st.expander("✨ Add New Prompt Version"):
        new_prompt = st.text_area(
            "Enter new prompt",
            placeholder="Write your new prompt here...",
            height=150
        )
        cols = st.columns([3, 1])
        with cols[0]:
            prompt_description = st.text_input(
                "Description",
                placeholder="Brief description of changes"
            )
        with cols[1]:
            if st.button("Save", use_container_width=True):
                if new_prompt and prompt_description:
                    prompt_manager.add_prompt_version(current_model, new_prompt, prompt_description)
                    st.success("✅ Saved!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please fill all fields")
    
    # Session management with visual enhancement
    st.subheader("📅 Chat Sessions")
    if st.button("🆕 Start New Chat", use_container_width=True):
        st.session_state.session_id = db.create_session()
        st.session_state.messages = []
        st.rerun()
    
    # Recent sessions with better visualization
    st.write("📚 Recent Sessions:")
    for session_id, session_data in db.list_sessions():
        if st.button(
            f"🕒 {session_data['created_at'][:16]}",
            key=session_id,
            use_container_width=True
        ):
            st.session_state.session_id = session_id
            st.session_state.messages = db.get_chat_history(session_id)
            st.rerun()

# Update title to include model type information
model_display_name = st.session_state.model_choice
if st.session_state.model_choice == "Gemini":
    if st.session_state.use_fine_tuned:
        model_display_name = "Gemini (Fine-tuned)"
    if st.session_state.use_memory_rag_mode:
        model_display_name += " [Memory RAG]"
    elif st.session_state.use_rag_mode:
        model_display_name += " [RAG]"
    elif not st.session_state.use_fine_tuned:
        model_display_name = "Gemini (Base)"

# Main chat interface
st.title(f"🌍 Travel & Weather Assistant ({model_display_name})")
st.write("Your personal guide to exploring the world's cultures and destinations!")

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = db.get_chat_history(st.session_state.session_id)

# Display chat messages with enhanced styling
for message in st.session_state.messages:
    icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=icon):
        st.markdown(message["content"])

# Chat input with custom placeholder
if prompt := st.chat_input("Ask about travel destinations, local culture, weather, or anything you'd like to know!"):
    # Apply the current fine-tuned model preference
    if st.session_state.model_choice == "Gemini":
        set_model_preference(st.session_state.use_fine_tuned)
    
    # Log user input
    logger.log_struct({
        'event_type': 'user_input',
        'session_id': st.session_state.session_id,
        'model': model_display_name,
        'message': prompt,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Show typing indicator
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            # Get model response
            start_time = time.time()
            current_model = st.session_state.model_choice.lower()
            system_prompt = prompt_manager.get_latest_prompt(current_model)
            
            # Pass only the user's direct prompt to RAG, RAGEngine handles its own templating
            user_direct_prompt = prompt
            
            # Define the tools to be used based on the session state
            tools = [get_weather_tool] if st.session_state.use_weather else None

            if st.session_state.model_choice == "Gemini" and st.session_state.use_memory_rag_mode:
                logger.log_text(f"Calling Memory RAG model for session {st.session_state.session_id}")
                full_response = chat_with_memory_rag(
                    user_prompt=user_direct_prompt,
                    user_id=st.session_state.user_id,
                    tools=tools
                )
            elif st.session_state.model_choice == "Gemini" and st.session_state.use_rag_mode:
                logger.log_text(f"Calling RAG model for session {st.session_state.session_id}")
                full_response = chat_with_gemini_rag(user_direct_prompt, tools=tools)
            elif current_model == "gemma":
                # Note: Function calling is typically a feature of more advanced models like Gemini.
                # Gemma might not support the 'tools' parameter as effectively.
                # For this example, we'll keep its original logic.
                full_prompt_for_gemma = f"{system_prompt}\\n\\nUser: {user_direct_prompt}\\nAssistant:"
                full_response = chat_with_gemma(full_prompt_for_gemma)
            else:  # Default to Gemini (base or fine-tuned, non-RAG)
                full_prompt_for_gemini = f"{system_prompt}\\n\\nUser: {user_direct_prompt}\\nAssistant:"
                full_response = chat_with_gemini(full_prompt_for_gemini, tools=tools)
            
            response_time = time.time() - start_time
            
            # Log response
            logger.log_struct({
                'event_type': 'model_response',
                'model': model_display_name,
                'session_id': st.session_state.session_id,
                'prompt_length': len(prompt),
                'response_length': len(full_response),
                'response_time': response_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Save and display response
            db.save_message(st.session_state.session_id, "assistant", full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.markdown(full_response)
            
            # Record interaction for Memory RAG if enabled
            if st.session_state.model_choice == "Gemini" and st.session_state.use_memory_rag_mode:
                add_user_interaction(
                    user_id=st.session_state.user_id,
                    user_message=prompt,
                    assistant_response=full_response
                )

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Made with ❤️ by Debi | Using Gemini & Gemma Models
    </div>
    """,
    unsafe_allow_html=True
)