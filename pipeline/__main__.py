import json
import sys
import time
from pathlib import Path

from pipeline.laya import classify, load_agent
from pipeline.ollama import judge, warmup, warmup_seconds

ROOT = Path(__file__).resolve().parent.parent


def main():
    warmup()
    signals = json.loads((ROOT / "data" / "signals.json").read_text())
    criteria = json.loads((ROOT / "data" / "signal_types.json").read_text())
    timings = [("ollama warmup", warmup_seconds())]
    started = time.perf_counter()
    agent = load_agent()
    timings.append(("laya load", time.perf_counter() - started))
    classified = []
    correct = 0
    for signal in signals:
        started = time.perf_counter()
        signal_type, confidence = classify(agent, signal, criteria)
        timings.append((f"laya classify [{signal['id']}]", time.perf_counter() - started))
        if signal_type == signal["expected"]:
            correct += 1
        classified.append(
            {
                "id": signal["id"],
                "frequency_mhz": signal["frequency_mhz"],
                "bandwidth_mhz": signal["bandwidth_mhz"],
                "amplitude_dbm": signal["amplitude_dbm"],
                "signal_type": signal_type,
                "confidence": confidence,
            }
        )
    started = time.perf_counter()
    judgment = judge(classified)
    timings.append(("ollama judge", time.perf_counter() - started))
    jamming = [signal["id"] for signal in signals if signal["expected"] == "jamming"]
    for item in classified:
        print(json.dumps(item))
    print(f"\033[32m{json.dumps({'judgment': judgment})}\033[0m")
    print(
        f"laya {correct}/{len(signals)} "
        f"expected jamming: {', '.join(jamming)} "
        f"expected not jamming: {len(signals) - len(jamming)}"
    )
    for step, seconds in timings:
        print(f"{step}: {seconds:.2f}s", file=sys.stderr)
    print(f"total: {sum(seconds for _, seconds in timings):.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
