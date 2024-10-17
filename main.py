import warnings
from pathlib import Path

from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole
import streamlit as st

# General configuration.
warnings.simplefilter('ignore')

# Self-host Nvidia NIM microservice.
PORT=8000
MODEL_ID = "facebook/opt-125m" # "meta/llama-3.1-8b-instruct"
llm = NVIDIA(
    model=MODEL_ID,
    base_url=f"http://localhost:{PORT}/v1",
    max_tokens=64,
)

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content=("You are a helpful assistant.")
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=("who are you? please elaborate in less then 100 words."),
    ),
]

# Sanity check.
chat_response = llm.chat(messages)


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "facebook/opt-125m"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        chat_response = llm.chat(messages)
        response = st.write(chat_response.message.content)
    st.session_state.messages.append({"role": "assistant", "content":
                                      chat_response.message.content})


