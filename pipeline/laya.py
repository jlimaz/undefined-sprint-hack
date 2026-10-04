import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import laya
import torch
from laya.common import build_head

CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "convaiinnovations/laya")


# The checkpoint is about 1.6 GiB of float32 weights, plus working memory.
CUDA_FREE_BYTES_NEEDED = 2 * 1024**3
BATCH_SIZE = 32
# Laya's default question budget (192 tokens) cuts each of the 12 family
# descriptions to about 13 tokens. The full question needs 414.
MAX_LEN = 512
HEAD_MAX_LEN = 448


def pick_device():
    """Use a GPU only when it can actually hold the model."""
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= CUDA_FREE_BYTES_NEEDED:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"laya: this checkpoint ships invalid temperatures",
            category=RuntimeWarning,
        )
        return laya.load(CHECKPOINT, device=pick_device())


def _scaled(hertz):
    """Format a value in Hz with the unit the question criteria would use."""
    for unit, size in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if hertz >= size:
            return f"{hertz / size:.3g} {unit}"
    return f"{hertz} Hz"


# Upper edge in Hz of each band, named as the question criteria name them.
BANDS = (
    (30e3, "very low frequency (VLF)"),
    (300e3, "low frequency (LF)"),
    (3e6, "medium wave (MF)"),
    (30e6, "shortwave (HF)"),
    (300e6, "VHF"),
    (3e9, "UHF"),
)


def _band(hertz):
    """Name the band a frequency falls in."""
    for upper, name in BANDS:
        if hertz < upper:
            return name
    return "microwave (SHF)"


def describe(observation):
    """Rewrite measurements in the same terms as the question criteria."""
    return (
        f"{_band(observation['center_frequency_hz'])} band, "
        f"centered at {_scaled(observation['center_frequency_hz'])}, "
        f"{_scaled(observation['bandwidth_hz'])} wide, "
        f"{observation['modulation']} modulation."
    )


def _check_untruncated(agent, questions):
    """Fail loudly if Laya would shorten any option description to fit its budget."""
    for name, question in questions.items():
        stats = build_head(agent.tok, agent._to_internal(question), HEAD_MAX_LEN)[2]
        if stats["tokens_per_option"] is not None:
            raise SystemExit(
                f"Question {name!r} does not fit in HEAD_MAX_LEN={HEAD_MAX_LEN}: "
                f"each option would be cut to {stats['tokens_per_option']} tokens."
            )


def classify(agent, observations, questions):
    """Answer every question for every observation, in input order."""
    _check_untruncated(agent, questions)
    states = [describe(observation) for observation in observations]
    results = agent.predict_batch(
        states,
        questions,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
    )
    return [
        {
            name: (answer["choice"], answer["confidence"])
            for name, answer in result["answers"].items()
        }
        for result in results
    ]


def release(agent):
    """Drop the model and hand its GPU memory back."""
    agent.model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
