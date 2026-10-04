import pytest

from pipeline import analysis


def test_ollama_being_down_is_reported_before_laya_loads(monkeypatch):
    def unreachable():
        raise SystemExit("Ollama is not reachable at http://127.0.0.1:11434: refused")

    monkeypatch.setattr(analysis, "unload", unreachable)
    with pytest.raises(analysis.OllamaError, match="Ollama is not reachable"):
        analysis.run({}, [])
