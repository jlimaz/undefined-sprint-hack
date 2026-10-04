"""Convert training observations into the items Laya's fine-tuning script loads.

Writes two balanced training sets, a small one that is a subset of the large one,
plus the test file both are evaluated on. The mock file is shared between the two:
a quarter of its signal types are held out as the unseen test rows, a fifth of the
observations of the other types are kept as the seen test rows, and the rest are
added to the training data. Every item is tokenised exactly as the pipeline
tokenises an observation at inference time.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from laya.common import QTYPES, build_sequence
from transformers import AutoTokenizer

from finetune.sources.common import MOCK, ROOT, TRAIN, read_jsonl
from pipeline.laya import HEAD_MAX_LEN, MAX_LEN, describe

# Fine-tuning always starts from the published checkpoint.
BASE_CHECKPOINT = "convaiinnovations/laya"
QUESTIONS = ROOT / "data" / "questions_20.json"
OUT = ROOT / "finetune" / "out"
DEFAULT_TRAIN = [TRAIN / "sigid.jsonl", TRAIN / "panoradio_hf.jsonl", TRAIN / "drone_links.jsonl"]
STAGES = {"stage1": 2000, "stage2": 12000}
QUESTION = "family"
SEED = 20
# Share of each family's mock signal types kept out of training, as the unseen test rows.
HELD_OUT_SHARE = 0.25
# Families with no unseen test rows. Their mock signals have nothing in common
# (interference) or sit far from the family's usual bands (navigation_time), so a
# held-out one cannot be recognised from the others.
UNSEEN_EXCLUDED = {"interference", "navigation_time"}
# Share of each shared signal's mock rows kept out of training, as the seen test rows.
SEEN_TEST_SHARE = 0.2


def split_mock(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the mock rows into test rows, marked seen or unseen, and training rows."""
    by_family = collections.defaultdict(set)
    for row in rows:
        by_family[row["label_family"]].add(row["label_signal"])
    rng = random.Random(SEED)
    held_out = set()
    for family in sorted(by_family):
        names = sorted(by_family[family])
        rng.shuffle(names)
        if family not in UNSEEN_EXCLUDED:
            held_out.update(names[: max(1, round(HELD_OUT_SHARE * len(names)))])
    test = [{**row, "split": "unseen"} for row in rows if row["label_signal"] in held_out]
    by_signal = collections.defaultdict(list)
    for row in rows:
        if row["label_signal"] not in held_out:
            by_signal[row["label_signal"]].append(row)
    train = []
    for name in sorted(by_signal):
        shared = by_signal[name]
        rng.shuffle(shared)
        kept = max(1, round(SEEN_TEST_SHARE * len(shared))) if len(shared) > 1 else 0
        test.extend({**row, "split": "seen"} for row in shared[:kept])
        train.extend(shared[kept:])
    return test, train


def spread(rows: list[dict], rng: random.Random) -> list[dict]:
    """Order one family's rows so any prefix covers as many signals as possible."""
    by_signal = collections.defaultdict(list)
    for row in rows:
        by_signal[row["label_signal"]].append(row)
    queues = list(by_signal.values())
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    ordered = []
    while queues:
        ordered.extend(queue.pop() for queue in queues)
        queues = [queue for queue in queues if queue]
    return ordered


def select(by_family: dict[str, list[dict]], total: int) -> list[dict]:
    """Take an equal share of each family, in the order `spread` gave them."""
    quota = -(-total // len(by_family))
    chosen = []
    for family, rows in by_family.items():
        if len(rows) < quota:
            print(f"  {family}: only {len(rows)} rows for a quota of {quota}")
        chosen.extend(rows[:quota])
    return chosen[:total] if len(chosen) > total else chosen


def build_item(tokenizer, row: dict, question: dict, rng: random.Random) -> dict:
    """One training item, with the options in a random order."""
    labels = list(question["criteria"])
    rng.shuffle(labels)
    internal = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": {label: question["criteria"][label] for label in labels},
    }
    ids, markers = build_sequence(tokenizer, describe(row), internal, MAX_LEN, HEAD_MAX_LEN)
    if len(markers) != len(labels):
        raise SystemExit("An item lost options to truncation; check MAX_LEN and HEAD_MAX_LEN.")
    label = labels.index(row["label_family"])
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[question["type"]],
        "target": [float(index == label) for index in range(len(labels))],
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=MOCK)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    question = json.loads(args.questions.read_text())[QUESTION]
    families = list(question["criteria"])
    rows = [row for path in args.train for row in read_jsonl(path)]
    test, shared = split_mock(read_jsonl(args.test))
    unseen = {row["label_signal"] for row in test if row["split"] == "unseen"}
    leaked = {row["label_signal"] for row in rows} & unseen
    if leaked:
        raise SystemExit(f"{len(leaked)} test signals are in the training data, e.g. {sorted(leaked)[:3]}")
    if {row["id"] for row in test} & {row["id"] for row in shared}:
        raise SystemExit("Some mock rows are in both the test file and the training data.")
    splits = collections.Counter(row["split"] for row in test)
    print(
        f"mock file: {splits['unseen']} unseen test rows from {len(unseen)} signals, "
        f"{splits['seen']} seen test rows, {len(shared)} rows added to training"
    )

    rng = random.Random(SEED)
    # Mock rows come first so every stage trains on all of them; the open data fills the rest.
    by_family = {
        family: [row for row in shared if row["label_family"] == family]
        + spread([row for row in rows if row["label_family"] == family], rng)
        for family in families
    }
    tokenizer = AutoTokenizer.from_pretrained(Path(snapshot_download(BASE_CHECKPOINT)) / "tokenizer")
    for stage, total in STAGES.items():
        print(f"{stage}: {total} items")
        chosen = select(by_family, total)
        # A fixed seed per stage keeps the item order independent of the other stage.
        stage_rng = random.Random(f"{SEED}-{stage}")
        items = [build_item(tokenizer, row, question, stage_rng) for row in chosen]
        folder = args.out / stage
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(items, folder / "train_items.pt")
        (folder / "train_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in chosen))
        counts = collections.Counter(row["label_family"] for row in chosen)
        signals = len({row["label_signal"] for row in chosen})
        print(f"  wrote {len(items)} items from {signals} signals to {folder / 'train_items.pt'}")
        print(f"  per family: {dict(counts)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test))
    print(f"test set: {args.out / 'test.jsonl'}")


if __name__ == "__main__":
    main()
