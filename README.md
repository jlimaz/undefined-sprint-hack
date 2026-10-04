# RF signal pipeline

A short local path: signal measurements go to Laya, which picks a type, then to Ollama for one plain-language summary.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ollama must already be running with `deepseek-r1:14b`. The first pipeline run downloads the English Laya checkpoint (about 808 MB) from Hugging Face.

## Run

From this directory:

```bash
python -m pipeline
```

Each signal is printed as one JSON object, then one operator summary. A reference line shows how many of Laya's labels match `expected`. Set `OLLAMA_HOST` if Ollama is not at `http://127.0.0.1:11434`. Set `OLLAMA_MODEL` to use a different local model; the default is `deepseek-r1:14b`. The run keeps that model loaded.

Signal types live in `data/signal_types.json`. Signal records live in `data/signals.json`.
