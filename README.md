# Introduction

A Nvidia and LlamaIndex project on multi-agent reasoning with knowledge graph, focusing on human nutrition.

## Instructions

Run following commands from project root directory.

* Ensure [Docker](https://docs.docker.com/engine/install/) is installed on host machine.

* Install python dependencies.

```python
pip install -r requirements.txt
```

* Initialize backend local instances with personal API keys.

```bash
bash scripts/run_nim
```

```bash
bash scripts/run_vllm
```

* Wait(<2min) for both local instances to be initialized before running next command.

```console
Started server process.
Waiting for application startup.
Application startup complete.
```

* Initialize frontend.

```pythong
streamlit run main.py
```

## Limitations

Running local instances of llm models requires at least 20GB of VRAM.
