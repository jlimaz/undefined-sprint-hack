"""Split the mock observations between training and test, by signal type.

A quarter of the signal types in each family are held out: all their rows are the
"unseen" test rows, and no training file may contain those signals. The other
signal types are shared: a fifth of their rows are the "seen" test rows and the
rest become training data.
"""

import collections
import json
import random

from finetune.sources.common import HELD_OUT, MOCK, TEST, TRAIN, read_jsonl, write_jsonl

OUTPUT = TRAIN / "mock_train.jsonl"
SEED = 20
HELD_OUT_SHARE = 0.25
SEEN_TEST_SHARE = 0.2


def main() -> None:
    rows = read_jsonl(MOCK)
    signals = collections.defaultdict(set)
    for row in rows:
        signals[row["label_family"]].add(row["label_signal"])

    rng = random.Random(SEED)
    held_out = set()
    print(f"{'family':<20} {'shared':>6} {'held out':>8}")
    for family in sorted(signals):
        names = sorted(signals[family])
        rng.shuffle(names)
        count = max(1, round(HELD_OUT_SHARE * len(names)))
        held_out.update(names[:count])
        print(f"{family:<20} {len(names) - count:>6} {count:>8}")

    by_signal = collections.defaultdict(list)
    for row in rows:
        by_signal[row["label_signal"]].append(row)
    test, train = [], []
    for name in sorted(by_signal):
        group = by_signal[name]
        if name in held_out:
            test.extend({**row, "split": "unseen"} for row in group)
            continue
        rng.shuffle(group)
        seen = round(SEEN_TEST_SHARE * len(group))
        test.extend({**row, "split": "seen"} for row in group[:seen])
        train.extend({**row, "mock_id": row["id"], "source": "mock"} for row in group[seen:])

    test.sort(key=lambda row: row["id"])
    TEST.write_text("".join(json.dumps(row) + "\n" for row in test))
    HELD_OUT.write_text(json.dumps(sorted(held_out), indent=2, ensure_ascii=False) + "\n")
    write_jsonl(OUTPUT, train)
    unseen = sum(row["split"] == "unseen" for row in test)
    print(f"held out {len(held_out)} of {len(by_signal)} signal types")
    print(f"test: {unseen} unseen rows and {len(test) - unseen} seen rows in {TEST}")
    print(f"training: {len(train)} rows in {OUTPUT}")


if __name__ == "__main__":
    main()
