"""HTTP service the dashboard uses to run Laya and read back what it found."""

import threading
import time

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import analysis
from pipeline.context import build_context, build_summary
from pipeline.inputs import InputError, parse_knowledge, parse_observations

app = FastAPI(title="Laya pipeline")

# One operator, one GPU: a single run at a time, and a single result kept.
_lock = threading.Lock()
_active = None


class Upload(BaseModel):
    name: str
    text: str


class ClassifyRequest(BaseModel):
    knowledge: Upload
    data: Upload


def _error(status, message, field=None):
    return JSONResponse({"error": message, "field": field}, status_code=status)


def _state():
    state = {"active": _active is not None, "running": _lock.locked()}
    return {**state, **_active} if _active else state


@app.exception_handler(RequestValidationError)
def _incomplete_request(request, exc):
    return _error(400, "Both a knowledge file and a data file are required.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/context")
def context():
    """The current classification, as the chat model reads it."""
    return _state()


@app.delete("/context")
def clear():
    global _active
    _active = None
    return _state()


@app.post("/classify")
def classify(request: ClassifyRequest):
    """Run Laya on the uploaded files and make the result the current one."""
    global _active
    try:
        questions = parse_knowledge(request.knowledge.text)
        observations, total = parse_observations(request.data.text)
    except InputError as exc:
        return _error(400, str(exc), exc.field)
    if not _lock.acquire(blocking=False):
        return _error(409, "Laya is already classifying. Wait for it to finish.")
    try:
        started = time.perf_counter()
        rows = analysis.run(questions, observations)
        seconds = time.perf_counter() - started
    except InputError as exc:
        return _error(400, str(exc), exc.field)
    except analysis.OllamaError as exc:
        return _error(502, str(exc))
    except Exception as exc:
        return _error(500, f"Laya failed: {exc}")
    finally:
        _lock.release()
    names = {"knowledge": request.knowledge.name, "data": request.data.name}
    summary = build_summary(names, questions, rows, total, seconds)
    _active = {"summary": summary, "context": build_context(names, questions, rows, total)}
    return {"summary": summary}
