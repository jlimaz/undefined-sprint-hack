"""Local Ollama client for the operator summary."""

import json
import os
import re
import time
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")

_SYSTEM = (
    "Respond directly. Do not reason, and do not include think tags. "
    "Answer the operator's question in plain language, using only the "
    "classification results provided. Do not revise or recount them."
)
DEFAULT_QUESTION = "Which observations are Morse code, and how many are not?"
_UNLOAD_TIMEOUT_S = 10.0
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


def unload() -> None:
    """Evict the model so its GPU memory is free, and wait until it is gone."""
    global _warmed
    _post("/api/generate", {"model": MODEL, "keep_alive": 0})
    _warmed = False
    deadline = time.perf_counter() + _UNLOAD_TIMEOUT_S
    while time.perf_counter() < deadline:
        loaded = _get("/api/ps").get("models") or []
        if all(entry.get("name") != MODEL for entry in loaded):
            return
        time.sleep(0.2)


def warmup_seconds() -> float:
    """Seconds spent on the warmup that actually loaded the model."""
    return _warmup_s


def judge(items: list[dict], modes: list[str], question: str = DEFAULT_QUESTION) -> str:
    """Answer one question about how Laya classified the observations."""
    ids = {mode: [] for mode in modes}
    for item in items:
        ids.setdefault(item["mode"], []).append(str(item["id"]))
    lines = [
        f"- {mode} ({len(members)}): {', '.join(members) or 'none'}"
        for mode, members in ids.items()
    ]
    prompt = (
        f"Laya classified {len(items)} observations. "
        "Signal type (count): observation ids\n"
        + "\n".join(lines)
        + f"\n\nQuestion: {question}"
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
            "options": {"temperature": 0.1, "num_predict": 512},
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


def _get(path: str) -> dict:
    """GET a JSON document from the local Ollama server."""
    return _send(urllib.request.Request(f"{HOST.rstrip('/')}/{path.lstrip('/')}"))


def _post(path: str, payload: dict) -> dict:
    """POST JSON to the local Ollama server and return the decoded body."""
    return _send(
        urllib.request.Request(
            f"{HOST.rstrip('/')}/{path.lstrip('/')}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


def _send(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama is not reachable at {HOST}: {exc.reason}") from exc
