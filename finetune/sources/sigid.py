"""Turn the Artemis signal catalogue (sigidwiki) into training observations.

Each catalogued signal has listed frequencies, bandwidths, modulations and
categories. Signals that the test file already uses are left out, so the test
file stays a test of signals never seen in training.
"""

import argparse
import collections
import random
import sqlite3

from finetune.sources.common import RAW, TRAIN, observe, test_signal_names, write_jsonl

DATABASE = RAW / "artemis" / "data.sqlite"
OUTPUT = TRAIN / "sigid.jsonl"
SEED = 20
ROWS_PER_FAMILY = 1200

# First match wins. A signal with several categories gets the most specific one.
FAMILY_BY_CATEGORY = [
    ("Radar", "radar"),
    ("Military", "military_comms"),
    ("Numbers Stations", "military_comms"),
    ("Amateur Radio", "amateur_radio"),
    ("Aviation", "aviation"),
    ("Marine", "marine"),
    ("Satellite", "satellite"),
    ("Time", "navigation_time"),
    ("Navigation", "navigation_time"),
    ("Commercial", "commercial_wireless"),
    ("Trunked Radio", "commercial_wireless"),
    ("Interfering", "interference"),
    ("Utility", "other_civil"),
]
# Names decide before categories: many catalogue entries carry no useful category.
FAMILY_BY_NAME = [
    ("jammer", "jamming"),
    ("interference", "interference"),
    ("noise", "interference"),
    ("rfi", "interference"),
    ("numbers station", "military_comms"),
    ("radiosonde", "other_civil"),
    ("weather balloon", "other_civil"),
    ("key fob", "other_civil"),
    ("keyfob", "other_civil"),
    ("car key", "other_civil"),
    ("keyless", "other_civil"),
    ("paging", "other_civil"),
    ("pager", "other_civil"),
    ("sensor", "other_civil"),
]


def family_of(name: str, categories: set[str]) -> str | None:
    for word, family in FAMILY_BY_NAME:
        if word in name.lower():
            return family
    for category, family in FAMILY_BY_CATEGORY:
        if category in categories:
            return family
    return None


def load_signals() -> list[dict]:
    """Every catalogued signal with the values needed to observe it."""
    if not DATABASE.exists():
        raise SystemExit(f"{DATABASE} is missing. Run: python -m finetune.sources.fetch")
    db = sqlite3.connect(DATABASE)

    def values(table: str) -> dict[int, list]:
        found = collections.defaultdict(list)
        for signal, value in db.execute(f"select SIG_ID, VALUE from {table}"):
            if value:
                found[signal].append(value)
        return found

    frequencies, bandwidths, modulations = values("frequency"), values("bandwidth"), values("modulation")
    categories = collections.defaultdict(set)
    for signal, label in db.execute(
        "select SIG_ID, VALUE from category join categorylabel using (CLB_ID)"
    ):
        categories[signal].add(label)
    return [
        {
            "name": name,
            "family": family_of(name, categories[signal]),
            "frequencies": frequencies[signal],
            "bandwidths": bandwidths[signal],
            "modulations": modulations[signal],
        }
        for signal, name in db.execute("select SIG_ID, NAME from signals order by SIG_ID")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows-per-family", type=int, default=ROWS_PER_FAMILY)
    args = parser.parse_args()

    held_out = test_signal_names()
    skipped = collections.Counter()
    by_family = collections.defaultdict(list)
    for signal in load_signals():
        if signal["name"] in held_out:
            skipped["used by the test file"] += 1
        elif signal["family"] is None:
            skipped["no family for its categories"] += 1
        elif not (signal["frequencies"] and signal["bandwidths"]):
            skipped["missing frequency or bandwidth"] += 1
        else:
            by_family[signal["family"]].append(signal)

    rng = random.Random(SEED)
    rows = []
    print(f"{'family':<20} {'signals':>7} {'rows':>6}")
    for family in sorted(by_family):
        signals = by_family[family]
        each = -(-args.rows_per_family // len(signals))
        for signal in signals:
            for _ in range(each):
                rows.append(
                    {
                        **observe(rng, signal["frequencies"], signal["bandwidths"], signal["modulations"] or ["unknown"]),
                        "label_family": family,
                        "label_signal": signal["name"],
                        "source": "sigidwiki",
                        "simulated": True,
                    }
                )
        print(f"{family:<20} {len(signals):>7} {each * len(signals):>6}")
    for reason, count in skipped.items():
        print(f"skipped {count} signals: {reason}")
    write_jsonl(OUTPUT, rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
