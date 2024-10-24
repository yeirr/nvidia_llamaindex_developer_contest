# Introduction

A Nvidia and LlamaIndex project on multi-agent reasoning with knowledge graph, focusing on human nutrition.

## Instructions

Run following commands from project root directory.

### Prerequisites

* Supports Ubuntu LTS(2204/2404) host only.
* Ensure [Docker](https://docs.docker.com/engine/install/) is installed on host machine.
* Ensure [Compose Plugin](https://docs.docker.com/compose/install/#scenario-two-install-the-compose-plugin) is installed on host machine.

* Install python dependencies.

```python
pip install -r requirements.txt
```

* Set API keys in '.env' file.

### Backend

Initialize local instances.

```bash
docker compose --env-file=.env --profile pg16age --profile nim up -d

# Takes around >5mins to build model workspace in nim container.
```

```bash
docker exec -it contest-pg16age psql -U postgres -d postgres -f /tmp/load_kg.sql

# Meanwhile wait for 'contest-pg16age' to initialize before importing graph data from CSV files by running mounted SQL script.
```

```bash
bash scripts/run_vllm

# Wait for backend instances to be initialized before running next command.
```

### Frontend

Initialize UI.

```pythong
streamlit run main.py
```

## Limitations

* Running local instances of llm models requires at least 24GB of VRAM.
* Initial inference require building of finite state machine(FSM) for structured output.
* Lacks data persistence on host filesystem.
