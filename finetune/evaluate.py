"""Measure a Laya checkpoint on the test observations, optionally against a baseline."""

import argparse
import collections
import json
from pathlib import Path

import laya

from finetune.sources.common import ROOT, read_jsonl
from pipeline.laya import classify, pick_device, release

QUESTIONS = ROOT / "data" / "questions_20.json"
TEST = ROOT / "finetune" / "out" / "test.jsonl"
BASE_CHECKPOINT = "convaiinnovations/laya"
QUESTION = "family"
# A fine-tune is worth scaling up when it clears both bars. Accuracy is taken on
# the unseen rows: signals that no training file contains.
MIN_UNSEEN_ACCURACY = 0.30
MAX_SHARE_OF_ONE_FAMILY = 0.50
SCORES = (("overall", "overall"), ("seen signals", "seen"), ("unseen signals", "unseen"))


def accuracy(rows: list[dict], predicted: list[str], split: str | None = None) -> tuple[float, int]:
    """Share of correct answers, over all rows or over one split, and the row count."""
    pairs = [
        (row["label_family"], answer)
        for row, answer in zip(rows, predicted)
        # Rows without a split tag count as unseen, the stricter reading.
        if split is None or row.get("split", "unseen") == split
    ]
    if not pairs:
        return 0.0, 0
    return sum(truth == answer for truth, answer in pairs) / len(pairs), len(pairs)


def measure(checkpoint: str, rows: list[dict], questions: dict) -> dict:
    """Classify every row with one checkpoint and summarise the outcome."""
    agent = laya.load(checkpoint, device=pick_device())
    print(f"running {checkpoint} on {agent.device}")
    predicted = [answer[QUESTION][0] for answer in classify(agent, rows, questions)]
    release(agent)
    truth = [row["label_family"] for row in rows]
    per_family = {}
    for family in questions[QUESTION]["criteria"]:
        total = truth.count(family)
        hits = sum(p == t == family for p, t in zip(predicted, truth))
        per_family[family] = (hits, total, predicted.count(family))
    top_family, top_count = collections.Counter(predicted).most_common(1)[0]
    return {
        "overall": accuracy(rows, predicted),
        "seen": accuracy(rows, predicted, "seen"),
        "unseen": accuracy(rows, predicted, "unseen"),
        "per_family": per_family,
        "top_family": top_family,
        "top_share": top_count / len(rows),
    }


def report(name: str, result: dict) -> None:
    print(f"\n{name}")
    for label, key in SCORES:
        share, count = result[key]
        if count:
            print(f"  {label + ':':<16} {share:.1%} of {count} rows")
    print(f"  most predicted:  {result['top_family']} ({result['top_share']:.1%} of answers)")
    print(f"  {'family':<20} {'found':>11} {'predicted':>10}")
    for family, (hits, total, predicted) in result["per_family"].items():
        print(f"  {family:<20} {hits:>5}/{total:<5} {predicted:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    parser.add_argument("--baseline", help="a second checkpoint to compare against")
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, help="use only the first N test rows")
    args = parser.parse_args()

    rows = read_jsonl(args.test)[: args.limit]
    questions = json.loads(args.questions.read_text())
    result = measure(args.checkpoint, rows, questions)
    report(f"{args.checkpoint} on {len(rows)} test rows", result)
    if args.baseline:
        baseline = measure(args.baseline, rows, questions)
        report(f"baseline {args.baseline}", baseline)
        print()
        for label, key in SCORES:
            if result[key][1]:
                print(f"change on {label}: {result[key][0] - baseline[key][0]:+.1%}")
    passed = result["unseen"][0] >= MIN_UNSEEN_ACCURACY and result["top_share"] <= MAX_SHARE_OF_ONE_FAMILY
    print(
        f"\nscale-up bar (unseen accuracy >= {MIN_UNSEEN_ACCURACY:.0%}, no family above "
        f"{MAX_SHARE_OF_ONE_FAMILY:.0%} of answers): {'PASSED' if passed else 'not met'}"
    )


if __name__ == "__main__":
    main()
