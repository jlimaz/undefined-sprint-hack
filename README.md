# RF signal pipeline

A short local path: radio observations go to Laya, which assigns each one a signal type, then to a local Ollama model that answers one plain-language question about the result.

## Current state

Laya is asked which of eight shortwave signal types an observation is, or `other`. The base checkpoint has to be fine-tuned before the pipeline output means anything: on an earlier, different question it scored at chance. The fine-tuning setup for the current question is ready but has not been run; see [finetune/README.md](finetune/README.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:4b-instruct
```

Ollama must be running. The first pipeline run downloads the English Laya checkpoint (about 808 MB) from Hugging Face.

## Run

From this directory:

```bash
python -m pipeline
```

Each observation is printed as one JSON object with the signal type Laya chose and its confidence. Then comes the Ollama answer in green, and a reference line showing how many of Laya's labels match `label_mode`. Step timings go to stderr.

To see only the answer and the reference line:

```bash
python -m pipeline | tail -n 2
```

### Asking a different question

The default question is "Which observations are Morse code, and how many are not?". Pass another one on the command line:

```bash
python -m pipeline "How many weather fax signals were seen?"
```

The Ollama model is given the count and observation ids per signal type, and nothing else. It is not told what any signal type means.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Laya checkpoint | `LAYA_CHECKPOINT` environment variable (a Hugging Face id or a local folder) | `convaiinnovations/laya` |
| Ollama model | `OLLAMA_MODEL` environment variable | `qwen3:4b-instruct` |
| Ollama address | `OLLAMA_HOST` environment variable | `http://127.0.0.1:11434` |
| Observations per run | `MAX_OBSERVATIONS` in `pipeline/__main__.py` | 100 |

Use a non-thinking Ollama model. `qwen3:4b` (without `-instruct`) reasons at length regardless of settings and runs out of tokens before it answers.

## Data

- `data/test_observations.jsonl`: 774 observations set aside from the training data. Each has a centre frequency, bandwidth, modulation and SNR, plus the expected `label_mode`. This is the test set, and what the pipeline reads.
- `data/questions_20.json`: the question Laya answers, with one description per signal type.
- `data/train/`: observations built from open data (Panoradio HF and sigidwiki), the source of both the training and the test rows.

Laya is shown three fields of each observation, as text: centre frequency (with the name of its band), bandwidth and modulation.

## How a run works

Laya and the Ollama model do not both fit on a small GPU, so they take turns:

1. The Ollama model is unloaded.
2. Laya loads and classifies every observation in one batch. It uses CUDA when at least 2 GiB of GPU memory is free, otherwise Apple MPS, otherwise CPU.
3. Laya is released and the Ollama model is loaded again and kept loaded.
4. Ollama answers the question.

On a 4 GB GPU, 100 observations take about 5 seconds to classify. Reloading Ollama takes 4 to 30 seconds.

Laya's default settings can cut the option descriptions short when there are many of them. `pipeline/laya.py` raises the budget so the full descriptions fit, and stops with an error if they ever do not.

## Fine-tuning

Fine-tuning is separate from the pipeline and lives in `finetune/`. It produces a checkpoint folder; the pipeline loads it through `LAYA_CHECKPOINT`:

```bash
LAYA_CHECKPOINT=models/laya-rf-modes python -m pipeline
```

The step-by-step guide is in [finetune/README.md](finetune/README.md).
