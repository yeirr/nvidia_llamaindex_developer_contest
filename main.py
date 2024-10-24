import asyncio
import concurrent.futures
import json
import os
import typing
import warnings

import psycopg
import streamlit as st
import uvloop
import yaml
from dotenv import dotenv_values
from dspy.predict import aggregation
from dspy.primitives.prediction import Completions
from duckduckgo_search import AsyncDDGS
from duckduckgo_search.exceptions import RatelimitException
from jinja2 import Environment, FileSystemLoader
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.nvidia import NVIDIA
from openai import AsyncOpenAI

# General configuration.
warnings.simplefilter("ignore")
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
config = dotenv_values(dotenv_path=".env")

# Postgres DB.
DB_URL = "postgresql://postgres@localhost:5432/postgres?sslmode=disable"
GRAPH_NAME = "age_dev"


# Utility functions.
def read_system_templates(
    filepath: str = "./prompts/templates/expert-identity",
    system_type: str = "generic",
) -> typing.List[str]:
    """Read and render system prompt templates for expert identities from directory.

    Example
    =======
        system_type = "generic"
        list_system_prompt = read_system_templates_expert_identity(system_type=system_type)
        print(list_system_prompt[0])

    Returns
    =======
        A list of rendered system prompt templates.
    """
    prompts = []
    environment = Environment(loader=FileSystemLoader(filepath))

    templates = [os.path.join(filepath, file.name) for file in os.scandir(filepath)]
    for idx in templates:
        if system_type in os.path.split(idx)[-1]:
            template = environment.get_template(os.path.split(idx)[1])
            render = template.render()
            prompts.append(yaml.safe_load(render)["template"])

    return prompts


async def call_agent_endpoint(
    query: str,
    agent: str,
    expert_identity: str,
    timeout: int = 30,
) -> str:
    # Parse expert prompt and query for agent inference.
    message = f"""
        [Identity Background]
        {expert_identity}

        Now given the above identity background, please answer the following query in
        paragraph form with no conclusion or summary: {query}
        """

    agent_messages = []

    try:
        ddgs_agent_message = await AsyncDDGS(timeout=timeout).achat(
            message,
            model=agent,
        )
        agent_messages.append(ddgs_agent_message)
    except RatelimitException as e:
        raise e

    return agent_messages[-1]


# Initialization.
async def init_openai_client() -> AsyncOpenAI:
    client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="token-abc123")

    # Warm up local vllm engine.
    await client.chat.completions.create(
        model="yeirr/llama3_2-1B-instruct-awq-g128-4bit",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=False,
        temperature=0.1,
        max_tokens=32,
        stop=["<|eot_id|>", "<|im_end|>", "</s>", "<|end|>"],
    )

    return client


async def init_workflow() -> typing.Any:
    # Initialize Nvidia NIM with workflow.
    workflow = StatefulWorkflow(timeout=30, verbose=False)

    # Sanity check.
    # from llama_index.utils.workflow import draw_all_possible_flows
    # draw_all_possible_flows(DefaultWorkflow, filename=/tmp/default_workflow.html)

    return workflow


async def init_ma_reasoning(
    openai_client: AsyncOpenAI, message: str, timeout: int
) -> str:
    ma_messages: typing.List[str] = []

    ddgs_chat_agent_types = [
        "gpt-4o-mini",
        "claude-3-haiku",
        "llama-3.1-70b",  # default
        "mixtral-8x7b",
    ]

    # Identity expert identities.
    classify_expert_identities = await openai_client.chat.completions.create(
        model="yeirr/llama3_2-1B-instruct-awq-g128-4bit",
        messages=[
            {
                "role": "user",
                "content": f"{read_system_templates(system_type='classify_identities')[0] + message}",
            }
        ],
        stream=False,
        temperature=0.1,
        max_tokens=128,
        extra_body={
            "guided_json": expert_identities_schema,
        },
    )

    expert_identities: typing.List[str] = json.loads(
        str(classify_expert_identities.choices[0].message.content)
    )["expert_identities"]

    # Add multi-agent reasoning.
    workers = len(set(expert_identities))

    # Spawn N number of multi-agentic expert responses.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                call_agent_endpoint,
                message,
                ddgs_chat_agent_types[2],
                read_system_templates(system_type=expert_identities[i])[0],
            )
            for i in range(workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            ma_messages.append(await future.result(timeout=timeout))

    # Convert agents responses to dspy prediction format for aggregation via
    # majority voting(most common response).
    dspy_preds = Completions([{"answer": message} for message in ma_messages])
    # Do not store dynamic reasoning into chat history.
    ma_reasoning = aggregation.majority(dspy_preds)["answer"]

    return str(ma_reasoning)


# Custom events.
class SetUpEvent(Event):
    message: str = Field(description="End user query in natural language.")
    ma_reasoning: str = Field(
        description="Multi-agent reasoning with consensus reached via majority voting."
    )


class KGStopEvent(Event):
    kg_query_response: str = Field(description="Cypher query response.")


class StatefulWorkflow(Workflow):
    llm = NVIDIA(
        api_key=config["NGC_API_KEY"],
        model=config["MODEL_ID"],
        base_url=f"http://localhost:{config['PORT']}/v1",
        max_tokens=config["MAX_TOKENS"],
    )

    @step
    async def setup_step(self, ctx: Context, ev: StartEvent) -> SetUpEvent:
        # Load data into global context.
        await ctx.set("message", ev.message)
        await ctx.set("ma_reasoning", ev.ma_reasoning)

        return SetUpEvent(message=ev.message, ma_reasoning=ev.ma_reasoning)

    @step
    async def query_kg_step(self, ctx: Context, ev: SetUpEvent) -> KGStopEvent:
        cypher_query = """
            MATCH p=(a)-[b]->(c)
            RETURN DISTINCT a.type, label(b), c.type
        """
        cypher_fields = "a agtype, string_result agtype, c agtype"

        # Query prebuilt knowledge graph(kg).
        kg_responses: typing.List[tuple[str, str, str]] = []
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                try:
                    # Cypher commands.
                    template = """
                        SELECT * 
                        FROM cypher('{graph_name}', $$ 
                            {cypher_query}
                        $$) AS ({cypher_fields})"""

                    # Mandatory.
                    cur.execute("""SET search_path = ag_catalog, "$user", public""")
                    cur.execute(
                        template.format(
                            graph_name=GRAPH_NAME,
                            cypher_query=cypher_query,
                            cypher_fields=cypher_fields,
                        )
                    )
                    for row in cur:
                        print(row)
                        kg_responses.append(row)
                except Exception as e:
                    print(e)
                    conn.rollback()

        # Parse kg tuple responses into list of strings.
        kg_response_parsed = "\n".join(
            [" ".join(map(str, idx)).replace('"', "") for idx in kg_responses]
        )

        return KGStopEvent(kg_query_response=kg_response_parsed)

    @step
    async def llm_step(self, ctx: Context, ev: KGStopEvent) -> StopEvent:
        message = await ctx.get("message")
        ma_reasoning = await ctx.get("ma_reasoning")

        # Run inference here.
        chat_response = await self.llm.astream_chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=config["SYSTEM_MESSAGE"]),
                ChatMessage(role=MessageRole.ASSISTANT, content=ma_reasoning),
                ChatMessage(role=MessageRole.TOOL, content=ev.kg_query_response),
                ChatMessage(role=MessageRole.USER, content=message),
            ],
            timeout=30,
        )

        # Return a generator.
        return StopEvent(result=chat_response)


# Schemas.
expert_identities_enum = [
    "biology",
    "calculus",
    "chemistry",
    "generic",
    "macroeconomics",
    "medicine",
    "microeconomics",
    "philosophy",
    "physics",
    "probability",
]
expert_identities_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "deprecated": False,
    "readOnly": True,
    "writeOnly": False,
    "title": "Expert identities",
    "required": ["expert_identities"],
    "type": "object",
    "properties": {
        "expert_identities": {
            "description": "Generate multiple expert identities for multi-agents reasoning.",
            "type": "array",
            "uniqueItems": True,
            "minItems": 1,
            "maxItems": 5,
            "unevaluatedItems": False,
            "items": {"type": "string", "enum": expert_identities_enum},
        }
    },
}


async def main(timeout: int = 30) -> None:
    openai_client = await init_openai_client()
    workflow = await init_workflow()

    # Set a default model.
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = config["MODEL_ID"]

    # Initialize chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=config["SYSTEM_MESSAGE"],
            )
        )

    # Display chat messages from history on app rerun.
    for message in st.session_state.messages:
        if dict(message)["role"] == "system" or dict(message)["content"] == "":
            st.empty()
        else:
            with st.chat_message(dict(message)["role"]):
                st.markdown(dict(message)["content"])

    # Placeholders.
    text_buffer: typing.List[str] = []

    # Accept user input.
    message = st.chat_input("Message")
    if message:
        # Add user message to chat history.
        st.session_state.messages.append(
            ChatMessage(role=MessageRole.USER, content=message)
        )

        # Display user message in chat message container.
        with st.chat_message("user"):
            st.markdown(dict(st.session_state.messages[-1])["content"])

        # Display assistant response in chat message container.

        with st.chat_message("assistant"):
            with st.status("Reasoning..."):
                ma_reasoning = await init_ma_reasoning(
                    openai_client=openai_client,
                    message=message,
                    timeout=timeout,
                )
                st.write(ma_reasoning)

            # Use llama-index messages format and custom defined workflow.
            handler = workflow.run(
                message=message,
                ma_reasoning=ma_reasoning,
            )

            # Typewriter effect: replace each displayed chunk.
            with st.empty():
                async for ev in handler.stream_events():
                    if isinstance(ev, StopEvent):
                        async for chunk in ev.result:
                            if chunk.raw.choices[0].finish_reason != "stop":
                                text_buffer.append(chunk.delta)
                                st.write("".join(text_buffer))

    # Write buffered response to history and current session.
    assistant_response = "".join(text_buffer)
    st.session_state.messages.append(
        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response)
    )


if __name__ == "__main__":
    asyncio.run(main())
