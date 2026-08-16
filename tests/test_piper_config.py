from pathlib import Path

import pytest
from pydantic import ValidationError

from vocalbin.piper.config import Config


def test_config_reads_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPER_MODEL_PATH", "voices/de.onnx")

    config = Config(_env_file=None)

    assert config.model_path == Path("voices/de.onnx")
    assert config.config_path is None


def test_config_reads_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPER_MODEL_PATH", "voices/de.onnx")
    monkeypatch.setenv("PIPER_CONFIG_PATH", "voices/de.onnx.json")

    config = Config(_env_file=None)

    assert config.config_path == Path("voices/de.onnx.json")


def test_config_requires_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPER_MODEL_PATH", raising=False)

    with pytest.raises(ValidationError, match="model_path"):
        Config(_env_file=None)
