import pytest

from pipeline import checkpoint


def test_the_environment_variable_wins_even_without_the_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYA_CHECKPOINT", "convaiinnovations/laya")
    monkeypatch.setattr(checkpoint, "FINE_TUNED", tmp_path / "missing")
    assert checkpoint.resolve() == "convaiinnovations/laya"


def test_the_fine_tuned_folder_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LAYA_CHECKPOINT", raising=False)
    (tmp_path / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(checkpoint, "FINE_TUNED", tmp_path)
    assert checkpoint.resolve() == str(tmp_path)


def test_a_missing_fine_tuned_folder_is_an_error_naming_the_path(monkeypatch, tmp_path):
    monkeypatch.delenv("LAYA_CHECKPOINT", raising=False)
    monkeypatch.setattr(checkpoint, "FINE_TUNED", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "missing")):
        checkpoint.resolve()
