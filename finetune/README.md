# Fine-tuning Laya

How to fine-tune Laya on the RF training data on an Apple Silicon Mac (written for an M4 Pro with 24 GB). Run every command from the repository root.

**Not yet tested:** `finetune/train.py` has never completed a run with this data. Everything else below has been run. Watch the first minutes of step 4, and expect that training time on the Mac is unknown.

## The plan

Training is done in two stages from the same base checkpoint. Stage 2 is only worth running if stage 1 clears the bar in step 5.

| Stage | Training items | Per family |
|---|---|---|
| 1 | 2,000 | about 167 |
| 2 | 12,000 | 1,000 |

Stage 1 is a subset of stage 2. Both are tested on `data/mock_observations_20.jsonl`, whose signals never appear in training.

## Steps

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

This takes a few seconds and writes `finetune/out/stage1/train_items.pt`, `finetune/out/stage2/train_items.pt` and `finetune/out/test.jsonl`. It should report 2,000 and 12,000 items.

### 3. Measure the baseline

```bash
python -m finetune.evaluate
```

Expect about 8.6% overall, with `amateur_radio` as nearly every answer.

### 4. Train stage 1

```bash
python finetune/train.py \
  --items finetune/out/stage1/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage1 \
  --epochs 4 --micro-batch 2 --grad-accum 16
```

The first run downloads the base checkpoint (about 800 MB) into `models/laya-base`. It should print `Device: mps` and `Training items: 1800; calibration items: 200`, then a loss line every 100 steps.

### 5. Check the result

```bash
python -m finetune.evaluate --checkpoint models/laya-rf-stage1 --baseline convaiinnovations/laya
```

The last line says whether the bar is met: at least 30% overall, and no single family making up more than half of the answers. The thresholds are constants at the top of `finetune/evaluate.py`.

### 6. Train stage 2, only if the bar was met

Same command as step 4 with `stage2` in both paths:

```bash
python finetune/train.py \
  --items finetune/out/stage2/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage2 \
  --epochs 4 --micro-batch 2 --grad-accum 16
```

Then evaluate it as in step 5.

### 7. Use the checkpoint

```bash
LAYA_CHECKPOINT=models/laya-rf-stage2 python -m pipeline
```

To use it on another machine, copy the whole checkpoint folder (about 800 MB) and point `LAYA_CHECKPOINT` at it. `models/` is not committed.

## If something goes wrong

- **Out of memory:** use `--micro-batch 1 --grad-accum 32`.
- **Interrupted run:** the trainer saves `checkpoint_latest/` inside the output folder after every epoch. It cannot resume; evaluate that folder or start again.
- **`Device: cpu` on the Mac:** MPS is not available in that Python environment. Training on CPU will take many hours.

## The training data

`data/train/` holds three files, all in the same format as the mock observations:

| File | Rows | Source |
|---|---|---|
| `sigid.jsonl` | 13,346 | [Artemis-DB](https://github.com/AresValley/Artemis-DB) v74, an export of sigidwiki.com. About 300 catalogued signals that the test file does not use. Observations are drawn from each signal's listed frequencies, bandwidths and modulations. |
| `panoradio_hf.jsonl` | 6,238 | [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/) (S. Scholl, 2019). Bandwidth and duty cycle are measured from the IQ vectors of 13 shortwave modes; SNR comes from the dataset. The centre frequency is assigned from where each mode is really operated. |
| `drone_links.jsonl` | 1,216 | Hand-written table of 19 drone control and video links. Approximate figures, not measurements. |

Known weak spots:

- **Few signals behind some families.** Jamming and radar come from 3 signals each, aviation from 4, marine from 6, navigation and time from 7. The rows are balanced, but the variety is not.
- **Panoradio is a small share.** It only covers amateur radio and marine modes, so after balancing it supplies about 500 of the 12,000 stage 2 items.
- **Drone links are not open data.** The catalogue has none.
- **Licences are unconfirmed** for Panoradio and the sigidwiki data. Check before publishing anything built on them.

### Rebuilding the training data

Only needed to change the sources. It downloads about 6 GB into `data/raw/`.

```bash
python -m finetune.sources.fetch
python -m finetune.sources.sigid
python -m finetune.sources.panoradio
python -m finetune.sources.drone_links
python -m finetune.build_dataset
```

To train on other data, write it in the same JSONL format and pass it to the builder:

```bash
python -m finetune.build_dataset --train data/train/my_file.jsonl
```

The builder refuses to continue if any training signal also appears in the test file.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Turns training observations into the items the trainer loads, tokenised exactly as the pipeline does at inference. |
| `train.py` | Laya's official fine-tuning script (Apache-2.0), vendored, with CUDA added as a device. |
| `evaluate.py` | Accuracy per family for any checkpoint on the test set. |
| `sources/` | Downloaders and converters for the open data. |
