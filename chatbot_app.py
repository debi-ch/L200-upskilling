import streamlit as st
from gemma_chat import chat_with_gemma
from google.cloud import logging_v2
from db_operations import ChatDatabase
import time

# Initialize Cloud Logging
logging_client = logging_v2.Client()
logger = logging_client.logger('streamlit_chatbot_logs')

# Initialize database
db = ChatDatabase()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = db.create_session()
    logger.log_text(f"New chat session created: {st.session_state.session_id}")

# Add a sidebar for session management
with st.sidebar:
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
st.title("Debi's Chatbot")

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
        'message': prompt,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Save and display user message
    db.save_message(st.session_state.session_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    start_time = time.time()
    full_response = chat_with_gemma(prompt)
    response_time = time.time() - start_time

    # Log model response
    logger.log_struct({
        'event_type': 'model_response',
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