import pytest
from pydantic import ValidationError

from vocalbin import OpenAICredentials, OpenAISpeechToText


def test_credentials_read_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    credentials = OpenAICredentials()

    assert credentials.api_key.get_secret_value() == "environment-key"


def test_credentials_require_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="api_key"):
        OpenAICredentials()


async def test_client_uses_environment_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    service = OpenAISpeechToText()

    try:
        assert service.client.api_key == "environment-key"
    finally:
        await service.aclose()


async def test_explicit_api_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    service = OpenAISpeechToText(api_key="explicit-key")

    try:
        assert service.client.api_key == "explicit-key"
    finally:
        await service.aclose()
