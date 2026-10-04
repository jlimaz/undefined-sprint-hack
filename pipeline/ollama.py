"""Local Ollama client for the operator summary."""

import json
import os
import re
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")

_SYSTEM = (
    "Respond directly. Do not reason, and do not include think tags. "
    "Write a plain-language summary for an operator. "
    "Name every signal classified as jamming, and state how many are not jamming."
)
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")
_warmed = False
_warmup_s = 0.0


def strip_thinking(text: str) -> str:
    """Remove DeepSeek-R1 chain-of-thought blocks from model output."""
    text = _THINK_BLOCK.sub("", text)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


def warmup() -> None:
    """Load model weights and keep them resident. Safe to call more than once."""
    global _warmed, _warmup_s
    if _warmed:
        return
    started = time.perf_counter()
    _post("/api/generate", {"model": MODEL, "keep_alive": -1})
    _warmed = True
    _warmup_s = time.perf_counter() - started


def warmup_seconds() -> float:
    """Seconds spent on the warmup that actually loaded the model."""
    return _warmup_s


def judge(items: list[dict]) -> str:
    """Summarize which signals are jamming, and how many are not."""
    lines = [f"- {item['id']}: {item['signal_type']}" for item in items]
    prompt = (
        "Each line is a signal and the type Laya assigned.\n"
        + "\n".join(lines)
        + "\nWrite one short summary for an operator. "
        "Name every signal classified as jamming. "
        "State how many are not jamming. Do not revise or recount."
    )
    payload = _post(
        "/api/chat",
        {
            "model": MODEL,
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "num_predict": 256},
        },
    )
    message = payload.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise SystemExit(f"Ollama returned no text for model {MODEL}.")
    text = strip_thinking(content)
    if not text:
        raise SystemExit(f"Ollama returned no answer for model {MODEL}.")
    return text


def _post(path: str, payload: dict) -> dict:
    """POST JSON to the local Ollama server and return the decoded body."""
    request = urllib.request.Request(
        f"{HOST.rstrip('/')}/{path.lstrip('/')}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama is not reachable at {HOST}: {exc.reason}") from exc


warmup()
