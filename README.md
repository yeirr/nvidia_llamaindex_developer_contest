# Introduction

A Nvidia and LlamaIndex project on multi-agent reasoning with knowledge graph, focusing on human nutrition.

## Instructions

Run following commands from project root directory.

* Ensure [Docker](https://docs.docker.com/engine/install/) is installed on host machine.

* Install python dependencies.

```python
pip install -r requirements.txt
```

* Initialize backend with personal API keys.

```bash
bash scripts/run_nim
```

* Wait(<2min) for backend to be initialized before running next command.

```console
INFO 2024-10-19 06:24:23.237 server.py:214] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

* Initialize frontend.

```pythong
streamlit run main.py
```
