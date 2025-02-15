import streamlit as st
from gemma_chat import chat_with_gemma  # Import your existing chatbot function
from google.cloud import logging_v2
import time

# Initialize Cloud Logging
logging_client = logging_v2.Client()
logger = logging_client.logger('streamlit_chatbot_logs')

# Add logging to your main app
st.title("Debi's Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.log_text("New chat session started")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Que pasa, Mufasa?"):
    # Log user input
    logger.log_struct({
        'event_type': 'user_input',
        'message': prompt,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get reply from the model
    start_time = time.time()
    full_response = chat_with_gemma(prompt)
    response_time = time.time() - start_time

    # Log model response and timing
    logger.log_struct({
        'event_type': 'model_response',
        'prompt_length': len(prompt),
        'response_length': len(full_response),
        'response_time': response_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_response)