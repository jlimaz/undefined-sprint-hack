# Fine-tuning Laya

How to fine-tune Laya on the RF training data on an Apple Silicon Mac (written for an M4 Pro with 24 GB). Run every command from the repository root, with the virtual environment active.

## Results so far

| Run | Test set | Accuracy |
|---|---|---|
| Base Laya | old: all 2,000 mock rows, every signal unseen | 8.6% |
| Stage 1, first run (4 epochs, `--grad-accum 16`) | old | 18.7% |
| Base Laya | new: 788 rows | 8.9% overall, 8.5% seen, 9.1% unseen |
| Stage 1, second run (5 epochs, `--grad-accum 4`) | new | not run yet |

The first run did not reach the 30% bar. It never predicted four families, because the training data had only 3 or 4 signals for some of them. The training data and test set were then changed (see "The test set"), so old and new numbers are not directly comparable.

## Rerun stage 1 (if you already did the first run)

```bash
git pull
python -m finetune.build_dataset
python finetune/train.py \
  --items finetune/out/stage1/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage1-v2 \
  --epochs 5 --micro-batch 2 --grad-accum 4
python -m finetune.evaluate --checkpoint models/laya-rf-stage1-v2 --baseline models/laya-rf-stage1
```

- The builder should report 2,000 and 12,000 items from 464 signals.
- The trainer should print `Device: mps` and `Training items: 1800; calibration items: 200`.
- With `--grad-accum 4` the model updates every 8 items instead of every 32, so the loss lines will swing more than in the first run. Expect the run to take about a quarter longer (one extra epoch).
- The last command compares the new checkpoint with the first run's. Both are scored on the new test file; neither was trained on its unseen signals.

## From scratch

### 1. Set up

```bash
git clone https://github.com/jlimaz/undefined-sprint-hack.git
cd undefined-sprint-hack
git checkout laya
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the training items

```bash
python -m finetune.build_dataset
```

This takes a few seconds and writes `finetune/out/stage1/train_items.pt`, `finetune/out/stage2/train_items.pt` and `finetune/out/test.jsonl`.

### 3. Measure the baseline

```bash
python -m finetune.evaluate
```

Expect 8.9% overall, with nearly every answer the same family.

### 4. Train stage 1

```bash
python finetune/train.py \
  --items finetune/out/stage1/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage1-v2 \
  --epochs 5 --micro-batch 2 --grad-accum 4
```

The first run downloads the base checkpoint (about 800 MB) into `models/laya-base`. A loss line is printed every 100 steps; about 2.5 is guessing level and lower is better.

### 5. Check the result

```bash
python -m finetune.evaluate --checkpoint models/laya-rf-stage1-v2 --baseline convaiinnovations/laya
```

It prints accuracy overall, on seen signals and on unseen signals, then a table per family. The last line says whether the bar for stage 2 is met: at least 30% on the unseen rows, and no single family making up more than half of the answers. The thresholds are constants at the top of `finetune/evaluate.py`.

### 6. Train stage 2, only if the bar was met

Same command as step 4 with `stage2` in the items path and a new output folder:

```bash
python finetune/train.py \
  --items finetune/out/stage2/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage2 \
  --epochs 5 --micro-batch 2 --grad-accum 4
```

Stage 2 has 12,000 items (1,000 per family) against stage 1's 2,000, and contains all of stage 1. Evaluate it as in step 5.

### 7. Use the checkpoint

```bash
LAYA_CHECKPOINT=models/laya-rf-stage1-v2 python -m pipeline
```

To use it on another machine, copy the whole checkpoint folder (about 800 MB) and point `LAYA_CHECKPOINT` at it. `models/` is not committed.

## If something goes wrong

- **Out of memory:** use `--micro-batch 1 --grad-accum 8`.
- **Interrupted run:** the trainer saves `checkpoint_latest/` inside the output folder after every epoch. It cannot resume; evaluate that folder or start again.
- **`Device: cpu` on the Mac:** MPS is not available in that Python environment. Training on CPU will take many hours.

## The test set

`data/test_observations.jsonl` holds 788 of the 2,000 mock observations. The mock file's 175 signal types are split per family:

- **Unseen (482 rows):** 43 signal types are held out completely. No training file contains them. This measures how the model handles signals it has never met. The names are in `data/held_out_signals.json`.
- **Seen (306 rows):** the other 132 signal types are shared with training. A fifth of their rows are kept here for testing; the model trained on other observations of the same signals.

The remaining 1,212 mock rows are training data.

What to expect: a small reference classifier trained on the stage 1 rows scores about 76% on the seen rows and about 35% on the unseen rows. Unseen accuracy is limited by the data itself: different families overlap in frequency, bandwidth and modulation. The 30% bar on unseen rows is therefore close to the ceiling.

## The training data

`data/train/` holds four files, all in the same format as the mock observations:

| File | Rows | Source |
|---|---|---|
| `sigid.jsonl` | 13,279 | [Artemis-DB](https://github.com/AresValley/Artemis-DB) v74, an export of sigidwiki.com. 424 catalogued signals, excluding the held-out ones. Observations are drawn from each signal's listed frequencies, bandwidths and modulations. |
| `panoradio_hf.jsonl` | 6,238 | [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/) (S. Scholl, 2019). Bandwidth and duty cycle are measured from the IQ vectors of 13 shortwave modes; SNR comes from the dataset. The centre frequency is assigned from where each mode is really operated. |
| `drone_links.jsonl` | 1,216 | Hand-written table of 19 drone control and video links. Approximate figures, not measurements. |
| `mock_train.jsonl` | 1,212 | The mock observations of shared signal types that are not in the test set. |

Known weak spots:

- **Few signals behind some families.** Jamming has 6 training signals, interference 11, marine 14, navigation and time 15. The rows are balanced, but the variety is not.
- **Panoradio is a small share.** It only covers amateur radio and marine modes, so after balancing it supplies about 280 of the 12,000 stage 2 items.
- **Drone links are not open data.** The catalogue has none.
- **Licences are unconfirmed** for Panoradio and the sigidwiki data. Check before publishing anything built on them.

### Rebuilding the training data

Only needed to change the sources or the split. `fetch` downloads about 6 GB into `data/raw/`.

```bash
python -m finetune.sources.fetch
python -m finetune.sources.split
python -m finetune.sources.sigid
python -m finetune.sources.panoradio
python -m finetune.sources.drone_links
python -m finetune.build_dataset
```

To add other data, write it in the same JSONL format and pass every training file to the builder:

```bash
python -m finetune.build_dataset --train data/train/*.jsonl data/my_file.jsonl
```

The builder refuses to continue if a held-out signal, or a test row, appears in the training data.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Turns training observations into the items the trainer loads, tokenised exactly as the pipeline does at inference. |
| `train.py` | Laya's official fine-tuning script (Apache-2.0), vendored, with CUDA added as a device. |
| `evaluate.py` | Accuracy overall, on seen and unseen signals, and per family, for any checkpoint. |
| `sources/` | Downloaders and converters for the open data, and the train/test split of the mock file. |
