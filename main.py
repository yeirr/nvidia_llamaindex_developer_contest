import asyncio
import warnings
from pathlib import Path

import streamlit as st
import uvloop
from llama_index.llms.nvidia import NVIDIA

# General configuration.
warnings.simplefilter("ignore")
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Initialize Nvidia NIM.
PORT = 8000
# MODEL_ID = "facebook/opt-125m"
MODEL_ID = "meta/llama-3.1-8b-instruct"
llm = NVIDIA(
    api_key=Path("/home/yeirr/secret/ngc_personal_key.txt")
    .read_text()
    .replace("\n", ""),
    model=MODEL_ID,
    base_url=f"http://localhost:{PORT}/v1",
    max_tokens=32,
)
prompt = "who are you? please elaborate in less then 100 words."

# Sanity check.
# chat_response = llm.complete(prompt)


async def main() -> None:
    # Set a default model.
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "facebook/opt-125m"

    # Initialize chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input.
    if prompt := st.chat_input("Message"):
        # Add user message to chat history.
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container.
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container.
        with st.chat_message("assistant"):
            chat_response = await llm.acomplete(prompt)
            st.write(chat_response.text)

        # Write to history.
        st.session_state.messages.append(
            {"role": "assistant", "content": chat_response.text}
        )


if __name__ == "__main__":
    asyncio.run(main())
