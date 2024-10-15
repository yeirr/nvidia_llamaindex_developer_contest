import warnings
from pathlib import Path

from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

# General configuration.
warnings.simplefilter('ignore')

# Self-host Nvidia NIM microservice.
llm = NVIDIA(
    model="meta/llama-3.1-8b-instruct",
    base_url="http://localhost:8000/v1",
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

llm.chat(messages)
