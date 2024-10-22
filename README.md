# Introduction

A Nvidia and LlamaIndex project on multi-agent reasoning with knowledge graph, focusing on human nutrition.

## Instructions

Run following commands from project root directory.

* Ensure [Docker](https://docs.docker.com/engine/install/) is installed on host machine.
* Ensure [Compose Plugin](https://docs.docker.com/compose/install/#scenario-two-install-the-compose-plugin) is installed on host machine.

* Install python dependencies.

```python
pip install -r requirements.txt
```

* Set API keys in '.env' file.
* Initialize backend local instances.

```bash
docker compose --env-file=.env --profile pg16age --profile nim up -d

# Takes around >5mins to build model workspace in nim container.
```

```bash
bash scripts/run_vllm

# Wait for backend instances to be initialized before running next command.
```

* Initialize frontend.

```pythong
streamlit run main.py
```

## Limitations

Running local instances of llm models requires at least 20GB of VRAM.
