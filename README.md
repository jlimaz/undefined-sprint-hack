# RF signal pipeline

A short local path: radio observations go to Laya, which assigns each one an emitter family, then to a local Ollama model that answers one plain-language question about the result.

## Current state

The base Laya checkpoint does not classify these observations: it scores 8.6% on the 2,000 mock observations (12 families, so chance is about 8%) and answers `amateur_radio` for almost everything. Laya has to be fine-tuned before the pipeline output means anything. A first fine-tuning run reached 18.7%; a second run on improved training data is prepared. See [finetune/README.md](finetune/README.md).

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

Each observation is printed as one JSON object with the family Laya chose and its confidence. Then comes the Ollama answer in green, and a reference line showing how many of Laya's labels match `label_family`. Step timings go to stderr.

To see only the answer and the reference line:

```bash
python -m pipeline | tail -n 2
```

### Asking a different question

The default question is "Which observations are signal jammers, and how many are not?". Pass another one on the command line:

```bash
python -m pipeline "How many radars were seen?"
```

The Ollama model is given the count and observation ids per family, and nothing else. It is not told what any family means.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Laya checkpoint | `LAYA_CHECKPOINT` environment variable (a Hugging Face id or a local folder) | `convaiinnovations/laya` |
| Ollama model | `OLLAMA_MODEL` environment variable | `qwen3:4b-instruct` |
| Ollama address | `OLLAMA_HOST` environment variable | `http://127.0.0.1:11434` |
| Observations per run | `MAX_OBSERVATIONS` in `pipeline/__main__.py` | 100 |

Use a non-thinking Ollama model. `qwen3:4b` (without `-instruct`) reasons at length regardless of settings and runs out of tokens before it answers.

## Data

- `data/mock_observations_20.jsonl`: 2,000 simulated observations of 175 catalogued signals. Each has a centre frequency, bandwidth, modulation, SNR and duty cycle, plus the expected `label_family` and `label_signal`. The pipeline reads its first 100.
- `data/test_observations.jsonl`: the 788 mock observations kept for testing fine-tuned checkpoints. The rest of the mock file is used for training; see [finetune/README.md](finetune/README.md).
- `data/questions_20.json`: the question Laya answers, with one description per family.
- `data/train/`: training observations built from open data and from the mock file, used only for fine-tuning.

Laya is shown three fields of each observation, as text: centre frequency, bandwidth and modulation.

## How a run works

Laya and the Ollama model do not both fit on a small GPU, so they take turns:

1. The Ollama model is unloaded.
2. Laya loads and classifies every observation in one batch. It uses CUDA when at least 2 GiB of GPU memory is free, otherwise Apple MPS, otherwise CPU.
3. Laya is released and the Ollama model is loaded again and kept loaded.
4. Ollama answers the question.

On a 4 GB GPU, 100 observations take about 5 seconds to classify. Reloading Ollama takes 4 to 30 seconds.

Laya's default settings cut each family description to about 13 tokens when there are 12 of them. `pipeline/laya.py` raises the budget so the full descriptions fit, and stops with an error if they ever do not.

## Fine-tuning

Fine-tuning is separate from the pipeline and lives in `finetune/`. It produces a checkpoint folder; the pipeline loads it through `LAYA_CHECKPOINT`:

```bash
LAYA_CHECKPOINT=models/laya-rf-stage1 python -m pipeline
```

The step-by-step guide is in [finetune/README.md](finetune/README.md).
