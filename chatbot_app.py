import streamlit as st
from gemma_chat import chat_with_gemma  # Import your existing chatbot function

st.title("Debi's Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Que pasa, Mufasa?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get reply from the model
    full_response = chat_with_gemma(prompt)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # Display assistant message in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_response)