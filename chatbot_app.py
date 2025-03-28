import streamlit as st
from gemma_chat import chat_with_gemma
from gemini_chat import chat_with_gemini
from google.cloud import logging_v2
from db_operations import ChatDatabase
from prompt_manager import PromptManager
import time

# Initialize Cloud Logging
logging_client = logging_v2.Client()
logger = logging_client.logger('streamlit_chatbot_logs')

# Initialize database and prompt manager
db = ChatDatabase()
prompt_manager = PromptManager()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = db.create_session()
    logger.log_text(f"New chat session created: {st.session_state.session_id}")

# Initialize model choice if not present
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Gemma"

# Add a sidebar for session management, model selection, and prompt management
with st.sidebar:
    st.title("Chat Settings")
    
    # Model selection
    st.subheader("Model Selection")
    model_choice = st.radio(
        "Choose your model:",
        ["Gemma", "Gemini"],
        index=0 if st.session_state.model_choice == "Gemma" else 1
    )
    st.session_state.model_choice = model_choice
    
    # Prompt Management
    st.subheader("Prompt Management")
    current_model = st.session_state.model_choice.lower()
    
    # Display current prompt
    st.write("Current Prompt:")
    current_prompt = prompt_manager.get_latest_prompt(current_model)
    st.text_area("Latest Prompt", current_prompt, height=100, disabled=True, key="current_prompt")
    
    # Add new prompt version
    with st.expander("Add New Prompt Version"):
        new_prompt = st.text_area("New Prompt", "", height=100, key="new_prompt")
        prompt_description = st.text_input("Description", key="prompt_description")
        if st.button("Save New Version"):
            if new_prompt:
                prompt_manager.add_prompt_version(current_model, new_prompt, prompt_description)
                st.success(f"New prompt version saved for {current_model}!")
                st.rerun()
    
    # View prompt history
    with st.expander("Prompt History"):
        versions = prompt_manager.get_prompt_versions(current_model)
        for version in versions:
            st.write(f"**{version['version']} - {version['created_at'][:10]}**")
            st.write(f"*Description:* {version['description']}")
            st.text(version['content'])
            st.divider()
    
    st.title("Chat Sessions")
    if st.button("New Chat"):
        st.session_state.session_id = db.create_session()
        st.session_state.messages = []
        st.rerun()
    
    # List recent sessions
    st.write("Recent Sessions:")
    for session_id, session_data in db.list_sessions():
        if st.button(f"Session {session_data['created_at'][:16]}", key=session_id):
            st.session_state.session_id = session_id
            st.session_state.messages = db.get_chat_history(session_id)
            st.rerun()

# Main chat interface
st.title(f"Debi's Chatbot ({st.session_state.model_choice})")

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = db.get_chat_history(st.session_state.session_id)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Que pasa, Mufasa?"):
    # Log user input
    logger.log_struct({
        'event_type': 'user_input',
        'session_id': st.session_state.session_id,
        'model': st.session_state.model_choice,
        'message': prompt,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Save and display user message
    db.save_message(st.session_state.session_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response based on selection
    start_time = time.time()
    current_model = st.session_state.model_choice.lower()
    system_prompt = prompt_manager.get_latest_prompt(current_model)
    
    # Combine system prompt with user prompt
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
    if current_model == "gemma":
        full_response = chat_with_gemma(full_prompt)
    else:
        full_response = chat_with_gemini(full_prompt)
    response_time = time.time() - start_time

    # Log model response
    logger.log_struct({
        'event_type': 'model_response',
        'model': st.session_state.model_choice,
        'session_id': st.session_state.session_id,
        'prompt_length': len(prompt),
        'response_length': len(full_response),
        'response_time': response_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Save and display assistant message
    db.save_message(st.session_state.session_id, "assistant", full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)