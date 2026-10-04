# Fine-tuning Laya

How to fine-tune Laya on the RF training data on an Apple Silicon Mac (written for an M4 Pro with 24 GB). Run every command from the repository root.

## Results so far

| Run | Test set | Accuracy |
|---|---|---|
| Base Laya | first: all 2,000 mock rows | 8.6% |
| Stage 1, first run (4 epochs, `--grad-accum 16`) | first | 18.7% |
| Stage 1, first run | second: 482 rows of unseen signal types | 19.3% |
| Stage 1, second run (5 epochs, `--grad-accum 4`) | second | 21.0% |
| Stage 2 (12,000 items, 3 epochs) | third: 726 rows, seen and unseen | not run yet |

The three test sets differ, so their numbers are not directly comparable.

The second run stayed far below the target, for two reasons:

- **Laya was not learning what the data holds.** A nearest-neighbour classifier on centre frequency, bandwidth and modulation, given the same 2,000 training rows, scores about 38% on the second test set. Laya found 0 of 41 jamming rows and 1 of 42 radar rows, which that classifier gets almost all right.
- **The second test set had a ceiling near 40%.** Its interference rows were one signal (HP Laptop, 50.9 kHz) and its navigation rows three signals at 8 to 450 MHz. Nothing in training resembles them and no classifier gets any of those 80 rows.

The same nearest-neighbour classifier scores about 64% on the third test set (about 80% seen, 50% unseen), so the data supports the 50% target. Whether Laya reaches it is not known until the run.

## What changed for this run

- **Test set** (see "The test set").
- **Training selection:** every stage now trains on all 1,274 mock training rows; the open data fills the rest of each family's share.
- **Size:** the 12,000-item set, for 3 epochs.
- **Observation text:** Laya is now told the band as well, e.g. `shortwave (HF) band, centered at 9.04 MHz, 10.2 kHz wide, PPM modulation.` Checkpoints trained before this change saw the text without the band and will score differently now.

These four changed together, so a gain cannot be attributed to any one of them.

## Run it (if you already did the earlier runs)

Before pulling, measure how well the last checkpoint fits its own training rows. This takes minutes and cannot be done afterwards:

```bash
source .venv/bin/activate
python -m finetune.evaluate --checkpoint models/laya-rf-stage1-v2 --test finetune/out/stage1/train_rows.jsonl
```

Keep the accuracy it prints. Below about 50% means the model was not fitting its training data; above about 80% means it fitted but did not carry over to new signals.

Then:

```bash
git pull
python -m finetune.build_dataset
python finetune/train.py \
  --items finetune/out/stage2/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage2 \
  --epochs 3 --micro-batch 8 --grad-accum 1 --no-checkpointing
python -m finetune.evaluate --checkpoint models/laya-rf-stage2 --baseline convaiinnovations/laya
```

- The builder should report 402 unseen and 324 seen test rows, 1,274 rows added to training, then 2,000 and 12,000 items.
- The trainer should print `Device: mps` and `Training items: 11600; calibration items: 400`.
- Compare against the base checkpoint, not the earlier runs: they trained on rows that are now seen test rows.

### How long it takes

The run is 34,800 item passes (11,600 items, 3 epochs). The second stage 1 run was 9,000 passes, so with the same flags this one takes about 3.9 times as long: about 8 hours if that run took 2, about 12 if it took 3. Five epochs would be 6.4 times as long.

`--micro-batch 8 --grad-accum 1 --no-checkpointing` updates every 8 items, as before, and should be faster than the earlier flags on 24 GB. This has not been measured. For a firm figure, time the first loss line (100 batches of 8 items) and multiply by 43.5.

The trainer saves `checkpoint_latest/` after every epoch, so a run can be stopped and that folder evaluated.

## The test set

The builder splits `data/mock_observations_20.jsonl` (175 signal types, 2,000 rows) three ways:

| Part | Rows | What it is |
|---|---|---|
| Unseen test rows | 402 | Every row of 39 signal types, a quarter of each family's types. No training data contains them. |
| Seen test rows | 324 | A fifth of the rows of the other 136 signal types. |
| Training rows | 1,274 | The rest of the rows of those 136 types. |

Interference and navigation and time have no unseen test rows: all their signal types are shared. Their mock signals have little in common with each other, so a held-out one cannot be recognised from the rest (see "Results so far"). The model is therefore not tested on new kinds of interference or navigation signals.

`finetune/evaluate.py` reports overall, seen and unseen accuracy. The bar is at least 50% overall, with no single family making up more than half of the answers; both thresholds are constants at the top of that file.

## Steps from scratch

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

This takes a few seconds and writes `finetune/out/stage1/train_items.pt` (2,000 items), `finetune/out/stage2/train_items.pt` (12,000 items, 1,000 per family) and `finetune/out/test.jsonl`. Stage 1 is the smaller set for a quick trial.

### 3. Measure the baseline

```bash
python -m finetune.evaluate
```

Expect a figure close to chance (about 8%), with nearly every answer the same family.

### 4. Train

```bash
python finetune/train.py \
  --items finetune/out/stage2/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-stage2 \
  --epochs 3 --micro-batch 8 --grad-accum 1 --no-checkpointing
```

The first run downloads the base checkpoint (about 800 MB) into `models/laya-base`. It prints a loss line every 100 batches.

### 5. Check the result

```bash
python -m finetune.evaluate --checkpoint models/laya-rf-stage2 --baseline convaiinnovations/laya
```

The last line says whether the bar is met.

### 6. Use the checkpoint

```bash
LAYA_CHECKPOINT=models/laya-rf-stage2 python -m pipeline
```

To use it on another machine, copy the whole checkpoint folder (about 800 MB) and point `LAYA_CHECKPOINT` at it. `models/` is not committed.

## If something goes wrong

- **Out of memory:** use `--micro-batch 4 --grad-accum 2`. If that fails too, use `--micro-batch 2 --grad-accum 4` and leave out `--no-checkpointing`.
- **Interrupted run:** the trainer saves `checkpoint_latest/` inside the output folder after every epoch. It cannot resume; evaluate that folder or start again.
- **`Device: cpu` on the Mac:** MPS is not available in that Python environment. Training on CPU will take many hours.

## The training data

`data/train/` holds three files, all in the same format as the mock observations. The builder adds the mock training rows to them.

| File | Rows | Source |
|---|---|---|
| `sigid.jsonl` | 13,346 | [Artemis-DB](https://github.com/AresValley/Artemis-DB) v74, an export of sigidwiki.com. About 300 catalogued signals that the mock file does not use. Observations are drawn from each signal's listed frequencies, bandwidths and modulations. |
| `panoradio_hf.jsonl` | 6,238 | [Panoradio HF](https://panoradio-sdr.de/radio-signal-classification-dataset/) (S. Scholl, 2019). Bandwidth and duty cycle are measured from the IQ vectors of 13 shortwave modes; SNR comes from the dataset. The centre frequency is assigned from where each mode is really operated. |
| `drone_links.jsonl` | 1,216 | Hand-written table of 19 drone control and video links. Approximate figures, not measurements. |

Known weak spots:

- **Few signals behind some families.** With the shared mock signals, jamming comes from 6 signals, interference from 12, marine from 14, navigation and time from 18. The rows are balanced, but the variety is not.
- **Panoradio is a small share.** It only covers amateur radio and marine modes, so after balancing it supplies about 430 of the 12,000 stage 2 items.
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

The mock training rows are always added. The builder refuses to continue if any of these files contains an unseen test signal.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Turns training observations into the items the trainer loads, tokenised exactly as the pipeline does at inference. |
| `train.py` | Laya's official fine-tuning script (Apache-2.0), vendored, with CUDA added as a device. |
| `evaluate.py` | Accuracy per family for any checkpoint on the test set. |
| `sources/` | Downloaders and converters for the open data. |
