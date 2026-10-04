import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import laya

CHECKPOINT = "convaiinnovations/laya"


def load_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"laya: this checkpoint ships invalid temperatures",
            category=RuntimeWarning,
        )
        return laya.load(CHECKPOINT, device="mps")


def describe(signal):
    """Rewrite measurements in the same terms as the signal-type criteria."""
    frequency = signal["frequency_mhz"]
    bandwidth = signal["bandwidth_mhz"]
    amplitude = signal["amplitude_dbm"]
    if 2400 <= frequency <= 2500:
        band = "centered near 2.4 GHz"
    elif 5000 <= frequency <= 5900:
        band = "centered near 5 GHz"
    elif 2700 <= frequency <= 3100:
        band = "centered near 2.7 to 3.1 GHz"
    elif 8000 <= frequency <= 10000:
        band = "centered near 9 GHz"
    else:
        band = f"centered near {frequency / 1000:.1f} GHz"
    if bandwidth <= 1:
        width = "about 1 MHz wide"
    elif bandwidth < 20:
        width = "a few MHz wide"
    elif bandwidth <= 80:
        width = "20 to 80 MHz wide"
    else:
        width = "hundreds of MHz wide"
    level = (
        "much louder than a normal signal"
        if amplitude > -25
        else "moderate amplitude"
    )
    return f"{width}, {band}, {level}."


def classify(agent, signal, criteria):
    state = describe(signal)
    question = {
        "signal_type": {
            "type": "choice",
            "instructions": "What type of signal is this?",
            "criteria": criteria,
        }
    }
    answer = agent.predict(state, question)["answers"]["signal_type"]
    return answer["choice"], answer["confidence"]
