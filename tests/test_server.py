import json

import pytest
from fastapi.testclient import TestClient

from pipeline import analysis, server
from pipeline.inputs import InputError

QUESTIONS = {
    "mode": {
        "type": "choice",
        "instructions": "Which signal type is this most likely?",
        "criteria": {"morse": "Morse code; OOK", "other": "any other signal"},
    }
}
ROWS = [
    {"id": 0, "center_frequency_hz": 7030000, "bandwidth_hz": 100, "modulation": "OOK"},
    {"id": 1, "center_frequency_hz": 14230000, "bandwidth_hz": 2700, "modulation": "FM"},
]


def fake_run(questions, observations):
    return [
        {
            "id": observation["id"],
            "state": f"state of {observation['id']}.",
            "answers": {"mode": ("morse" if observation["id"] == 0 else "other", 0.9)},
        }
        for observation in observations
    ]


def body(knowledge=None, data=None):
    return {
        "knowledge": {
            "name": "questions.json",
            "text": json.dumps(QUESTIONS) if knowledge is None else knowledge,
        },
        "data": {
            "name": "observations.json",
            "text": json.dumps(ROWS) if data is None else data,
        },
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(analysis, "run", fake_run)
    monkeypatch.setattr(server, "_active", None)
    return TestClient(server.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_no_context_before_a_run(client):
    assert client.get("/context").json() == {"active": False, "running": False}


def test_classify_returns_a_summary_and_activates_the_context(client):
    response = client.post("/classify", json=body())
    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["observations"] == 2
    assert summary["questions"][0]["counts"] == {"morse": 1, "other": 1}

    context = client.get("/context").json()
    assert context["active"] is True
    assert context["summary"] == summary
    assert "- morse (1): 0" in context["context"]


def test_clearing_the_context(client):
    client.post("/classify", json=body())
    assert client.delete("/context").json() == {"active": False, "running": False}
    assert client.get("/context").json()["active"] is False


def test_malformed_knowledge_is_a_400_on_that_field(client):
    response = client.post("/classify", json=body(knowledge="{nope"))
    assert response.status_code == 400
    assert response.json()["field"] == "knowledge"
    assert "line 1" in response.json()["error"]


def test_bad_data_is_a_400_on_that_field(client):
    response = client.post("/classify", json=body(data=json.dumps([{"id": 1}])))
    assert response.status_code == 400
    assert response.json()["field"] == "data"
    assert "Row 1" in response.json()["error"]


def test_missing_file_is_a_400(client):
    response = client.post("/classify", json={"knowledge": body()["knowledge"]})
    assert response.status_code == 400
    assert "error" in response.json()


def test_a_failed_run_keeps_the_previous_context(client, monkeypatch):
    client.post("/classify", json=body())
    monkeypatch.setattr(analysis, "run", lambda *_: (_ for _ in ()).throw(RuntimeError("out of memory")))
    response = client.post("/classify", json=body())
    assert response.status_code == 500
    assert "out of memory" in response.json()["error"]
    assert client.get("/context").json()["active"] is True


def test_ollama_down_is_a_502(client, monkeypatch):
    def unreachable(*_):
        raise analysis.OllamaError("Ollama is not reachable at http://127.0.0.1:11434: refused")

    monkeypatch.setattr(analysis, "run", unreachable)
    response = client.post("/classify", json=body())
    assert response.status_code == 502
    assert "Ollama is not reachable" in response.json()["error"]


def test_a_question_laya_cannot_fit_is_a_400_on_knowledge(client, monkeypatch):
    def too_long(*_):
        raise InputError("knowledge", "Question 'mode' does not fit in HEAD_MAX_LEN=448.")

    monkeypatch.setattr(analysis, "run", too_long)
    response = client.post("/classify", json=body())
    assert response.status_code == 400
    assert response.json()["field"] == "knowledge"


def test_a_second_run_while_one_is_going_is_a_409(client):
    assert server._lock.acquire(blocking=False)
    try:
        response = client.post("/classify", json=body())
        assert response.status_code == 409
        assert client.get("/context").json()["running"] is True
    finally:
        server._lock.release()
